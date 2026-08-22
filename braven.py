"""
braven.py — Python SDK for the Braven experiment tracker.

DIRECT LOGGING (wandb-style)
─────────────────────────────
One-time setup:
    python -m braven login

Then in any script:
    import braven

    braven.init(name="My Experiment")
    braven.config("learning_rate", 0.001)
    braven.summary("accuracy", 0.94)
    braven.upload("plot.png")
    braven.finish()

QUERYING EXPERIMENTS
─────────────────────
    from braven import Braven

    b = Braven("https://api.bravenlab.com", api_key="braven_...")
    exp = b.get("high temp run")
    df  = exp.file("data.csv").as_dataframe()

PIPELINE SCRIPTS (run by the Braven worker)
─────────────────────────────────────────────
    import braven

    def process(braven=None):
        braven.log_config("lr", "0.001")
        braven.log_summary("acc", "0.94")
        braven.log_artifact("plot.png")

MULTIPLE DEVICES IN ONE PIPELINE RUN (ADR-0013)
──────────────────────────────────────────────
    # Tag values with a stable per-device key; the KPI name stays the same for
    # every device (no "SNR_dev1"), the device is a separate coordinate — and
    # in a pipeline script, each device gets its own child Experiment under
    # the run's (parent) Experiment.
    for sensor_id, snr in results.items():
        braven.get_device(sensor_id).log_summary("SNR", snr)
    braven.log_summary("max_device_mismatch", spread)   # parent-level

    # Or ambiently, for several calls against the same device:
    braven.set_device(sensor_id)
    braven.log_summary("SNR", snr)
    braven.upload(fig, name="spectrum.png")
    braven.set_device(None)   # back to parent-level
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Union

import requests

# Windows consoles often default to a non-UTF-8 codepage (e.g. cp1252). Since
# scripts built on this SDK commonly print µ/°/± and similar symbols, guard
# against UnicodeEncodeError crashing an otherwise-successful run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Credential store
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path.home() / ".braven" / "config.json"
_DEFAULT_API_URL = "https://api.bravenlab.com"
_SETTINGS_URL = "https://app.bravenlab.com/app/settings"


def login(api_url: str, api_key: str) -> None:
    """Save credentials to ~/.braven/config.json."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps({"api_url": api_url.rstrip("/"), "api_key": api_key}, indent=2),
        encoding="utf-8",
    )


def _prompt_login() -> dict:
    """wandb-style first-use prompt (wandb.init() does the same thing when
    you're not logged in): asks for a Watcher Key right here instead of
    crashing, so a fresh `pip install braven` only ever needs one
    braven.init(...)/collect(...) call — no separate `python -m braven
    login` step first. Only reached when stdin is a real terminal (see
    _load_config) — there's nobody to answer this in CI or a piped script."""
    print("You're not logged in to Braven yet.")
    print(f'Get your Watcher Key from {_SETTINGS_URL} (under "Watcher Keys" -> Generate Key).')
    api_key = getpass.getpass("Paste your Watcher Key (braven_...): ").strip()
    if not api_key:
        raise RuntimeError(
            "No API key entered — not logged in.\n"
            "Run:  python -m braven login\n"
            "Or:   braven.login(api_url, api_key)"
        )
    login(_DEFAULT_API_URL, api_key)
    print(f"Logged in. Credentials saved to {_CONFIG_PATH}\n")
    return {"api_url": _DEFAULT_API_URL, "api_key": api_key}


def _load_config() -> dict:
    """Return stored credentials. If none are saved yet, prompts once
    (wandb-style — see _prompt_login) when running interactively; raises a
    clear instruction otherwise (CI, a piped/redirected script, a notebook
    kernel with no real stdin)."""
    if not _CONFIG_PATH.exists():
        if sys.stdin.isatty():
            return _prompt_login()
        raise RuntimeError(
            "Not logged in to Braven.\n"
            "Run:  python -m braven login\n"
            "Or:   braven.login(api_url, api_key)"
        )
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Corrupted credentials file {_CONFIG_PATH}: {e}") from e


# ---------------------------------------------------------------------------
# Script capture — links an experiment to the exact script/version that
# produced it (content-hash identity, see docs/specs/01-script-linkage.md).
# ---------------------------------------------------------------------------

_SCRIPT_SIZE_CAP = 2 * 1024 * 1024  # 2MB — above this, hash+path are still sent, source is not


def _capture_script_info() -> dict:
    """Best-effort identification of the running script. Never raises — returns
    {} on any failure (REPL/Jupyter has no __main__.__file__, permissions
    errors, etc.), since a run should never fail because of this."""
    try:
        main_mod = sys.modules.get("__main__")
        script_path = getattr(main_mod, "__file__", None)
        if not script_path or not os.path.isfile(script_path):
            return {}
        script_path = os.path.abspath(script_path)
        raw = Path(script_path).read_bytes()
        info = {
            "script_hash": hashlib.sha256(raw).hexdigest(),
            "script_name": os.path.basename(script_path),
            "script_local_path": script_path,
            "script_byte_size": len(raw),
        }
        if len(raw) <= _SCRIPT_SIZE_CAP:
            try:
                info["script_content"] = raw.decode("utf-8")
            except UnicodeDecodeError:
                pass  # binary/undecodable — hash+path still sent, source skipped
        return info
    except Exception as e:
        print(f"[braven] script capture skipped: {e}")
        return {}


# ---------------------------------------------------------------------------
# Plot-series key convention (paths B & D, docs/specs/03-interactive-plots.md)
#
# Explicit plot_series() and matplotlib-derived series both live under
# category="series" metadata, distinguished only by key shape — no schema
# change. plot_series() uses `plot::{name}::{idx}::y|x|meta`; the `{idx}` is a
# per-name call counter (NOT in the original spec's literal `plot::{name}::y`)
# — without it, a second plot_series() call with the same `name` would
# overwrite the first trace's key, since both Run._metadata and the pipeline
# _metadata_queue dedupe by exact key.
# ---------------------------------------------------------------------------

_plot_series_counters: dict[str, int] = {}


def _next_plot_trace_idx(name: str) -> int:
    idx = _plot_series_counters.get(name, 0)
    _plot_series_counters[name] = idx + 1
    return idx


def _build_plot_series_entries(
    name: str,
    y: list,
    x: list | None,
    x_label: str | None,
    y_label: str | None,
    mode: str,
) -> list[tuple[str, str, str]]:
    """Build (key, value, category) triples for one plot_series() call. y_label also
    doubles as the trace's legend label when multiple traces share one `name`."""
    idx = _next_plot_trace_idx(name)
    prefix = f"plot::{name}::{idx}"
    # float() each element (not just list()) so numpy arrays — the common case for
    # scientific/hardware data — serialize cleanly; raw numpy scalars aren't JSON-safe.
    entries = [(f"{prefix}::y", json.dumps([float(v) for v in y]), "series")]
    if x is not None:
        entries.append((f"{prefix}::x", json.dumps([float(v) for v in x]), "series"))
    meta = {"mode": mode, "trace_label": y_label, "x_label": x_label, "y_label": y_label}
    entries.append((f"{prefix}::meta", json.dumps(meta), "series"))
    return entries


# ---------------------------------------------------------------------------
# Matplotlib figure introspection (path D, docs/specs/03-interactive-plots.md)
# ---------------------------------------------------------------------------


def _is_matplotlib_figure(obj) -> bool:
    """True if obj is a live matplotlib Figure. Matplotlib stays an optional
    dependency — this never imports it at module load time."""
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return False
    return isinstance(obj, Figure)


_PREVIEW_MAX_DIM = 420  # px — cards render ~206-480px wide; ≲100KB target
# MUST be kept in sync with PREVIEW_RASTER_EXTS in
# backend-ts/src/services/previews.ts — the backend only signs preview URLs
# for extensions this generator can actually produce a preview for.
_PREVIEW_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def _make_image_preview(file_path: Path) -> bytes | None:
    """Downscaled WebP preview bytes for a raster image, or None.

    List thumbnails: uploaded matplotlib figures are multi-MB full-res PNGs
    painted at ~206px on the experiments list. A ≤420px WebP beside the
    original lets the list page skip the full-res download entirely.
    Best-effort by contract: any failure (non-image, missing file, PIL
    without WebP support) returns None and the upload proceeds without a
    preview — the list falls back to the original.
    """
    if file_path.suffix.lower() not in _PREVIEW_IMAGE_EXTS:
        return None
    try:
        from PIL import Image  # ships with matplotlib (pillow)

        with Image.open(file_path) as im:
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if im.mode == "P" else "RGB")
            im.thumbnail((_PREVIEW_MAX_DIM, _PREVIEW_MAX_DIM))
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=80, method=4)
            return buf.getvalue()
    except Exception:
        return None


def _upload_file_parts(upload_name: str, fh, file_path: Path, content_type: str | None = None) -> dict:
    """The multipart `files=` dict for a /files upload: the file itself plus,
    for raster images, a generated `preview` part the backend stores beside
    the original in R2 (see _make_image_preview). Every upload path — direct
    Run.upload(), a pipeline's upload()/_pipeline_upload(), and
    Analysis.upload() — routes through this one function so a preview is
    generated the same way regardless of how the upload got here.
    `content_type` is passed through when the caller already resolved one
    (Analysis.upload() guesses from the filename so the backend doesn't
    default it); omitted elsewhere, matching the prior request behavior."""
    parts: dict = {"file": (upload_name, fh, content_type) if content_type else (upload_name, fh)}
    preview = _make_image_preview(file_path)
    if preview is not None:
        parts["preview"] = (f"{upload_name}.webp", preview, "image/webp")
    return parts


def _safe_filename_sdk(name: str) -> str:
    """Literal port of backend/services/file_service.py:_safe_filename — MUST be kept
    byte-for-byte in sync so mplplot:: keys line up with the FileRecord.filename the
    backend actually stores for the uploaded PNG (see the matching comment there)."""
    parts = name.replace("\\", "/").split("/")
    safe_parts = [re.sub(r"[^\w\-_. ]", "_", part, flags=re.ASCII) for part in parts if part]
    return "/".join(safe_parts) if safe_parts else "file"


def _to_float_list(values) -> list[float] | None:
    """Coerce an iterable (plain list, numpy array, etc.) to a list of Python floats.
    Returns None if any element can't convert — e.g. datetime/categorical axis data,
    which the walker should skip rather than guess at (matches the spec's explicit
    rejection of mpl_to_plotly()'s "non-numeric ticks silently become 0,1,2,…" gap)."""
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


# Interactive companion plots are previews — the PNG holds full fidelity.
# Without this cap, a figure plotting raw sensor data (hundreds of thousands
# of points per trace, 12+ subplots) turns into tens of millions of floats
# JSON-encoded into series metadata: minutes of CPU per figure, and a
# metadata flush large enough to blow the Worker's request/memory limits
# (observed live 2026-07-17: pipeline runs grinding for 10+ minutes on what
# the PNG-only version did in seconds).
_MAX_PLOT_POINTS_PER_TRACE = 2000


def _downsample_trace(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Stride-sample a trace down to <= _MAX_PLOT_POINTS_PER_TRACE points,
    always keeping the final point so the trace ends where the data ends."""
    n = len(xs)
    if n <= _MAX_PLOT_POINTS_PER_TRACE:
        return xs, ys
    step = -(-n // _MAX_PLOT_POINTS_PER_TRACE)  # ceil division
    xd, yd = xs[::step], ys[::step]
    if xd[-1] != xs[-1]:
        xd.append(xs[-1])
        yd.append(ys[-1])
    return xd, yd


def _walk_matplotlib_figure(fig) -> list[dict]:
    """Extract line/scatter series from a live matplotlib Figure. Never raises — returns
    [] on any failure or if no usable line/scatter artists are found (imshow, 3D,
    contour/quiver, annotate-only figures all fall through to this silently, matching
    the SDK's graceful-degradation philosophy used elsewhere, e.g. _capture_script_info).

    Returns one dict per Axes that yielded >=1 usable trace:
        {"axes_index", "x_label", "y_label", "x_scale", "y_scale",
         "traces": [{"label", "x": [...], "y": [...], "mode": "line"|"scatter"}]}
    """
    try:
        from matplotlib.collections import PathCollection
    except Exception:
        PathCollection = None  # matplotlib always available here (caller already imported it)

    groups: list[dict] = []
    try:
        axes_list = fig.get_axes()
    except Exception:
        return []
    for ax_idx, ax in enumerate(axes_list):
        try:
            if hasattr(ax, "zaxis"):  # 3D axes — not supported, PNG-only fallback
                continue
            traces = []
            for line in ax.lines:
                try:
                    xdata, ydata = _to_float_list(line.get_xdata()), _to_float_list(line.get_ydata())
                except Exception:
                    continue
                if not xdata or not ydata or len(xdata) != len(ydata):
                    continue
                label = line.get_label()
                # A Line2D with linestyle="None" (e.g. errorbar(fmt="o")) draws markers
                # only, no connecting stroke — tag it "scatter" so the frontend renders
                # discrete points instead of a misleading line through sparse samples.
                try:
                    is_marker_only = line.get_linestyle() in (None, "None", "none", "")
                except Exception:
                    is_marker_only = False
                xdata, ydata = _downsample_trace(xdata, ydata)
                traces.append({
                    "label": None if not label or label.startswith("_") else label,
                    "x": xdata,
                    "y": ydata,
                    "mode": "scatter" if is_marker_only else "line",
                })
            for coll in ax.collections:
                # Only true scatter collections (ax.scatter()) carry meaningful point
                # data via get_offsets() — other Collection subtypes (fill_between's
                # PolyCollection, errorbar's LineCollection, etc.) also expose
                # get_offsets() but it returns a meaningless default (often a single
                # point at the origin) rather than raising, which would otherwise inject
                # a spurious garbage trace instead of correctly falling back to PNG-only.
                if PathCollection is not None and not isinstance(coll, PathCollection):
                    continue
                try:
                    offsets = coll.get_offsets()
                except Exception:
                    continue
                if offsets is None or len(offsets) == 0:
                    continue
                xs = _to_float_list(p[0] for p in offsets)
                ys = _to_float_list(p[1] for p in offsets)
                if not xs or not ys:
                    continue
                label = coll.get_label()
                xs, ys = _downsample_trace(xs, ys)
                traces.append({
                    "label": None if not label or label.startswith("_") else label,
                    "x": xs,
                    "y": ys,
                    "mode": "scatter",
                })
            if traces:
                groups.append({
                    "axes_index": ax_idx,
                    "x_label": ax.get_xlabel() or None,
                    "y_label": ax.get_ylabel() or None,
                    "x_scale": ax.get_xscale(),
                    "y_scale": ax.get_yscale(),
                    "traces": traces,
                })
        except Exception:
            continue  # this Axes failed — skip it, not the whole figure
    return groups


# ---------------------------------------------------------------------------
# Run — direct experiment logger
# ---------------------------------------------------------------------------

class Run:
    """An active experiment being logged to directly.

    Returned by ``braven.init()``. You can use the module-level helpers
    (``braven.config()``, ``braven.summary()``, etc.) instead of calling
    methods on this object — both work identically.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        project_id: str,
        name: str | None = None,
        notes: str | None = None,
    ) -> None:
        self._base_url = api_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Project-Id": project_id,
        }
        # (device_key, key) → {key, value, category, device_key}. device_key is
        # None for experiment-level values; set for a device slice (Spec 04), so
        # two devices' "SNR" are distinct entries rather than one overwriting the
        # other. The backend resolves device_key → a persistent device record.
        self._metadata: dict[tuple[str | None, str], dict] = {}
        self._devices: dict[str, "Device"] = {}

        payload = {"name": name, "notes": notes}
        payload.update(_capture_script_info())

        resp = requests.post(
            f"{self._base_url}/experiments",
            json=payload,
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self.id: str = data["id"]
        self.name: str | None = data.get("name")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def config(self, key: str, value) -> None:
        """Log an input / configuration parameter (experiment-level)."""
        self._log(key, value, "config")

    def summary(self, key: str, value) -> None:
        """Log an output metric (experiment-level)."""
        self._log(key, value, "summary")

    def get_device(self, key: str, type: str | None = None) -> "Device":
        """Return a device-scoped handle within this experiment (Spec 04;
        renamed from device() in 0.2.0 — see the module-level device()'s
        docstring for why). Everything logged on it is tagged with `key` (a
        stable per-project device identifier); the backend resolves that to a
        persistent device record, auto-created on first sight. An optional
        `type` names the device type to create it under when it's new
        (ignored for an existing device — a device is never re-typed).
        Memoised per run, so repeated calls with the same key return the same
        handle (the first call's type wins).

        Note: unlike the module-level get_device()/set_device() (pipeline
        scripts, ADR-0013), a direct-logging Run has no parent/child
        Experiment concept — this still tags device-scoped values onto rows
        of the SAME Experiment (Spec 04), and Device.upload() isn't
        supported from a Run's device handle (pipeline-only)."""
        key = str(key)
        if key not in self._devices:
            self._devices[key] = Device(key, run=self, type=type)
        return self._devices[key]

    def plot_series(
        self,
        name: str,
        y: list,
        x: list | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        mode: str = "line",
    ) -> None:
        """Log a labeled data series as an interactive plot. Call multiple times with
        the same `name` and a distinct `y_label` to group traces onto one chart."""
        self._log_many(_build_plot_series_entries(name, y, x, x_label, y_label, mode))

    def upload(self, path_or_fig, name: str | None = None, interactive: bool = True) -> None:
        """Upload a file and attach it to this experiment.

        `path_or_fig` may be a file path, or a live matplotlib Figure — in the latter
        case the PNG is saved the same way `fig.savefig()` would, and (when
        `interactive` is True, the default) the SDK also attempts to extract an
        interactive companion series from the figure's line/scatter data
        (best-effort, never blocks the upload).

        Pass `interactive=False` to skip that extraction — it costs an extra
        metadata flush to the backend (ADR-0002's deferred opt-out; see the
        ADR for why per-line identity was rejected). Worth setting for a figure
        re-uploaded rapidly in a loop (e.g. a live-updating plot), where only
        the final frame's interactive data is likely to matter.
        """
        fig = path_or_fig if _is_matplotlib_figure(path_or_fig) else None
        tmp_path: Path | None = None
        if fig is not None:
            tmp_path = Path(tempfile.mktemp(suffix=".png"))
            fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
            file_path = tmp_path
            upload_name = name or "figure.png"
        else:
            file_path = Path(path_or_fig)
            if not file_path.exists():
                raise FileNotFoundError(f"upload: file not found: {file_path}")
            upload_name = name or file_path.name

        with open(file_path, "rb") as fh:
            resp = requests.post(
                f"{self._base_url}/files",
                files=_upload_file_parts(upload_name, fh, file_path),
                params={"experiment_id": self.id},
                headers=self._headers,
                timeout=120,
            )
        resp.raise_for_status()

        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if fig is not None and interactive:
            self._log_matplotlib_series(fig, upload_name)

    def finish(self) -> None:
        """Flush metadata (idempotent). Called automatically by braven.finish()."""
        self._flush()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, key: str, value, category: str, device_key: str | None = None, device_type: str | None = None) -> None:
        self._metadata[(device_key, key)] = {"key": key, "value": str(value), "category": category, "device_key": device_key, "device_type": device_type}
        self._flush()

    def _log_many(self, entries: list[tuple[str, str, str]], device_key: str | None = None, device_type: str | None = None) -> None:
        """Like _log(), but for multiple (key, value, category) triples flushed once —
        avoids one PUT per key when writing several related entries (e.g. a
        plot_series() call or a matplotlib figure's extracted traces)."""
        for key, value, category in entries:
            self._metadata[(device_key, key)] = {"key": key, "value": value, "category": category, "device_key": device_key, "device_type": device_type}
        self._flush()

    def _log_matplotlib_series(self, fig, image_name: str) -> None:
        """Best-effort: extract interactive series data from a matplotlib figure and log
        it alongside the PNG already uploaded. Never raises — a failure here must never
        affect the (already-successful) file upload."""
        try:
            groups = _walk_matplotlib_figure(fig)
            if not groups:
                return
            base = _safe_filename_sdk(image_name)
            entries: list[tuple[str, str, str]] = []
            for g in groups:
                for trace_idx, tr in enumerate(g["traces"]):
                    prefix = f"mplplot::{base}::{g['axes_index']}::{trace_idx}"
                    entries.append((f"{prefix}::y", json.dumps(tr["y"]), "series"))
                    entries.append((f"{prefix}::x", json.dumps(tr["x"]), "series"))
                    meta = {
                        "mode": tr["mode"],
                        "trace_label": tr["label"],
                        "x_label": g["x_label"],
                        "y_label": g["y_label"],
                        "x_scale": g["x_scale"],
                        "y_scale": g["y_scale"],
                    }
                    entries.append((f"{prefix}::meta", json.dumps(meta), "series"))
            self._log_many(entries)
        except Exception as e:
            print(f"[braven] matplotlib series extraction skipped: {e}")

    def _flush(self) -> None:
        if not self._metadata:
            return
        resp = requests.put(
            f"{self._base_url}/experiments/{self.id}/metadata",
            json=list(self._metadata.values()),
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()

    def __repr__(self) -> str:
        return f"Run(name={self.name!r}, id={self.id!r})"


# ---------------------------------------------------------------------------
# Device — a device-scoped logging handle within an experiment (Spec 04)
# ---------------------------------------------------------------------------

class Device:
    """A device within an experiment. Values logged here are tagged with the
    device's stable key and resolved by the backend to a persistent device
    record (auto-created on first sight), so a physical sensor accumulates
    history across experiments.

    Obtain one via ``run.get_device(key)`` (direct logging) or module-level
    ``braven.get_device(key)`` (pipeline scripts). The rule is simple:

        dev = run.get_device("SENSOR-4471")
        dev.log_summary("SNR", 14.2)     # this device's SNR
        run.log_summary("max_mismatch", 3.1)  # experiment-level (all devices)

    ``dev.*`` is device-scoped; ``run.*`` / ``braven.*`` stay experiment-level.
    The KPI name never changes ("SNR" stays "SNR" for every device) — the device
    is a separate coordinate, not a suffix.
    """

    def __init__(self, key: str, *, run: "Run | None" = None, type: str | None = None) -> None:
        # run set → direct-logging mode (writes via the Run); run None → pipeline
        # mode (writes onto the worker's pipeline metadata queue). `type` is an
        # optional device-type name, applied only if the device is created here.
        self.key = str(key)
        self._run = run
        self.type = type

    def log_config(self, key: str, value) -> None:
        """Log an input / configuration parameter for this device."""
        self._log(key, value, "config")

    def log_summary(self, key: str, value) -> None:
        """Log an output metric for this device."""
        self._log(key, value, "summary")

    def log_series(self, key: str, values: list) -> None:
        """Log a numeric data series for this device (stored as JSON)."""
        self._log(key, json.dumps(values), "series")

    # Direct-logging aliases, mirroring Run.config/Run.summary naming.
    config = log_config
    summary = log_summary

    def plot_series(
        self,
        name: str,
        y: list,
        x: list | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        mode: str = "line",
    ) -> None:
        """Log a labeled interactive plot series for this device."""
        entries = _build_plot_series_entries(name, y, x, x_label, y_label, mode)
        if self._run is not None:
            self._run._log_many(entries, device_key=self.key, device_type=self.type)
        else:
            for k, v, c in entries:
                _pipeline_log(k, v, c, device_key=self.key, device_type=self.type)

    def upload(self, path_or_fig, name: str | None = None, interactive: bool = True) -> None:
        """Upload a file (or a live matplotlib Figure — see Run.upload) attached
        to this device's own child Experiment (ADR-0013). Pipeline scripts only
        (via braven.get_device()) — a direct-logging Run's device handle
        (run.get_device()) still uses Spec 04's device-tagged-row model for
        every other method on this class, so a device-scoped upload spinning
        up a child Experiment nothing else about that Run knows about would be
        a silent, confusing split; this raises instead.

        `interactive=False` skips the companion-series extraction for a
        matplotlib Figure — see Run.upload()."""
        if self._run is not None:
            raise RuntimeError(
                "Device.upload() is only supported for pipeline scripts (braven.get_device()), "
                "not a direct-logging Run's device handle (run.get_device())."
            )
        _pipeline_upload(path_or_fig, name, device_key=self.key, device_type=self.type, interactive=interactive)

    def _log(self, key: str, value, category: str) -> None:
        if self._run is not None:
            self._run._log(key, value, category, device_key=self.key, device_type=self.type)
        else:
            _pipeline_log(key, str(value), category, device_key=self.key, device_type=self.type)

    def __repr__(self) -> str:
        return f"Device(key={self.key!r})"


# ---------------------------------------------------------------------------
# Module-level API (wandb-style)
# ---------------------------------------------------------------------------

_current_run: Run | None = None


def init(
    name: str | None = None,
    notes: str | None = None,
    project: str | None = None,
    company: str | None = None,
) -> Run:
    """Create a new experiment and set it as the active run.

    Credentials are read automatically from ~/.braven/config.json.
    Run ``python -m braven login`` once to set them up.

    Args:
        name:    Experiment name shown in the UI.
        notes:   Optional description.
        project: Project name (as shown in Braven). Required if you belong
                 to more than one project.
        company: Company name. Only needed to disambiguate projects with
                 the same name across companies.
    """
    global _current_run
    cfg = _load_config()
    project_id = _resolve_project_id(cfg, company=company, project=project)
    _current_run = Run(
        api_url=cfg["api_url"],
        api_key=cfg["api_key"],
        project_id=project_id,
        name=name,
        notes=notes,
    )
    return _current_run


def _resolve_project_id(cfg: dict, company: str | None, project: str | None) -> str:
    """Look up the project ID by calling the Braven API with the stored credentials.

    Raises a RuntimeError with a clear message if no match is found.
    """
    if project is None and company is None:
        raise RuntimeError(
            "Specify company and project in braven.init():\n"
            '  braven.init(name="...", company="My Company", project="My Project")'
        )

    api_url = cfg["api_url"]
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}

    # Fetch all companies the API key has access to
    try:
        resp = requests.get(f"{api_url}/companies", headers=headers, timeout=10)
        resp.raise_for_status()
        all_companies = resp.json()
    except requests.HTTPError as e:
        raise RuntimeError(
            f"Could not fetch companies from {api_url}: {e}\n"
            "Check that your API key is valid."
        ) from e

    # Filter by company name (case-insensitive)
    if company is not None:
        matching_companies = [c for c in all_companies if c["name"].lower() == company.lower()]
    else:
        matching_companies = all_companies

    if not matching_companies:
        available = [c["name"] for c in all_companies]
        raise RuntimeError(
            f"Company {company!r} not found.\n"
            f"Available companies: {available}"
        )

    # For each matching company, search for the project by name
    found: list[tuple[str, str, str]] = []  # (company_name, project_id, project_name)
    for comp in matching_companies:
        try:
            resp = requests.get(
                f"{api_url}/projects",
                headers=headers,
                params={"company_id": comp["id"]},
                timeout=10,
            )
            resp.raise_for_status()
            projs = resp.json()
        except Exception:
            continue

        for p in projs:
            if project is None or p["name"].lower() == project.lower():
                found.append((comp["name"], p["id"], p["name"]))

    if not found:
        raise RuntimeError(
            f"Project {project!r} not found"
            + (f" in company {company!r}" if company else "")
            + ".\n"
            "Make sure the company and project names match exactly what you see in the Braven UI."
        )
    if len(found) > 1:
        raise RuntimeError(
            f"Multiple projects match — add company= to disambiguate:\n"
            + "\n".join(f'  company="{c}", project="{pn}"' for c, _, pn in found)
        )
    return found[0][1]


def config(key: str, value) -> None:
    """Log an input / configuration parameter on the active run."""
    if _current_run is None:
        raise RuntimeError("No active run. Call braven.init() first.")
    _current_run.config(key, value)


def summary(key: str, value) -> None:
    """Log an output metric on the active run."""
    if _current_run is None:
        raise RuntimeError("No active run. Call braven.init() first.")
    _current_run.summary(key, value)


def upload(path_or_fig, name: str | None = None, interactive: bool = True) -> None:
    """Upload a file (or a live matplotlib Figure — see Run.upload) attached to the
    current experiment. Works from either a direct-log run (after braven.init()) or a
    pipeline script — same dual dispatch as plot_series(). In a pipeline script, goes
    to the ambient current device's own child Experiment (see set_device()) if one is
    set, else the parent.

    `interactive=False` skips the companion-series extraction for a matplotlib
    Figure — see Run.upload()."""
    if _current_run is not None:
        _current_run.upload(path_or_fig, name, interactive=interactive)
    else:
        _pipeline_upload(path_or_fig, name, device_key=_current_device_key, device_type=_current_device_type, interactive=interactive)


def plot_series(
    name: str,
    y: list,
    x: list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    mode: str = "line",
) -> None:
    """Log a labeled data series as an interactive plot. Works from either a direct-log
    run (after braven.init()) or a pipeline script — call multiple times with the same
    `name` and a distinct `y_label` to group traces onto one chart (multi-trace)."""
    if _current_run is not None:
        _current_run.plot_series(name, y, x=x, x_label=x_label, y_label=y_label, mode=mode)
    else:
        _pipeline_plot_series(name, y, x=x, x_label=x_label, y_label=y_label, mode=mode)


def get_device(key: str, type: str | None = None) -> "Device":
    """Return a device-scoped logging handle for `key`. Works from a direct run
    (after braven.init()) or a pipeline script — same dual dispatch as
    upload()/plot_series().

        braven.get_device("SENSOR-4471").log_summary("SNR", 14.2)
        braven.get_device("SENSOR-4471", "Photodiode").log_summary("SNR", 14.2)

    The optional `type` names the device type to create the device under when
    it's first seen; it's ignored for a device that already exists (a device is
    resolved by its key alone and never re-typed). In a direct run the handle is
    memoised on the run; in a pipeline it's a light stateless handle that writes
    onto the pipeline metadata queue.

    In a PIPELINE script (ADR-0013), this also resolves-or-creates the device's
    own single-device child Experiment under the pipeline's Experiment (the
    "parent") — log_config/log_summary/log_series/plot_series/upload on the
    returned handle land on the child's own data, not a device-tagged row on
    the parent. See set_device() for the ambient equivalent (no explicit
    handle needed for every call)."""
    if _current_run is not None:
        return _current_run.get_device(key, type)
    return Device(str(key), run=None, type=type)


def device(key: str, type: str | None = None) -> "Device":
    """REMOVED (0.2.0, ADR-0013) — use get_device(key, type) instead (same
    signature, same return value — a drop-in rename), or set_device(key,
    type) if you want subsequent bare log_config()/log_summary()/log_series()/
    plot_series()/upload() calls to apply to that device without threading an
    explicit handle through every call. Raises instead of silently keeping
    device()'s old behavior: in a pipeline script, device()-tagged values used
    to land on a shared row of the parent Experiment (Spec 04); get_device()
    now gives that device its own child Experiment, a real behavior change
    old scripts relying on the old fused-row shape (e.g. reading
    `max_device_mismatch` against every device's row on one Experiment) would
    otherwise silently break under."""
    raise RuntimeError(
        "braven.device() has been removed - use braven.get_device(key, type) for an explicit "
        "handle, or braven.set_device(key, type) to make subsequent log_config()/log_summary()/"
        "log_series()/plot_series()/upload() calls apply to that device ambiently. "
        "braven.set_device(None) returns to experiment-level logging."
    )


def set_device(key: str | None, type: str | None = None) -> None:
    """Set (or clear) the ambient current device for subsequent pipeline-script
    calls to log_config()/log_summary()/log_series()/plot_series()/upload() —
    the device-scoped equivalent of how braven.init() sets the ambient current
    run (ADR-0013). Pipeline scripts only (mirrors get_device()'s "parent
    Experiment" concept, which a direct-logging Run doesn't have).

        braven.set_device("SENSOR-4471")
        braven.log_summary("SNR", 14.2)   # this device's SNR, own child Experiment
        braven.log_summary("SNR", 9.8)    # still SENSOR-4471 -- ambient, not per-call
        braven.set_device(None)           # back to parent-level logging
        braven.log_summary("max_device_mismatch", 3.1)  # parent-level again

    `type` follows get_device()'s rule: applied only if the device is new,
    ignored (never re-typing) for one that already exists. Pass key=None (or
    call with no arguments) to clear back to parent-level logging — the same
    as never having called set_device() at all. Reset automatically at the
    start of each pipeline run (init_pipeline()), so a previous run's device
    selection can never leak into a new one."""
    global _current_device_key, _current_device_type
    _current_device_key = str(key) if key is not None else None
    _current_device_type = type if key is not None else None


def finish() -> None:
    """Flush metadata and close the active run."""
    global _current_run
    if _current_run is not None:
        _current_run.finish()
    _current_run = None


# ---------------------------------------------------------------------------
# Pipeline execution context (used inside worker-run scripts — unchanged)
# ---------------------------------------------------------------------------

braven_experiment_id: str | None = None
braven_api_url: str | None = None
braven_watcher_secret: str | None = None
braven_user_id: str | None = None
braven_pipeline_id: str | None = None

# Streaming mode (2026-08-22, default ON): when true, log_config()/
# log_summary() (Experiment-level, non-series) each send their own entry to
# the backend immediately instead of only joining the buffered queue below —
# see _pipeline_log() and _stream_one_entry(). log_series()/plot_series() and
# any get_device(...)-scoped call are deliberately excluded regardless of this
# flag — they stay exclusively on the buffered flush_metadata() path (see
# _stream_one_entry's docstring for why). Toggle with
# braven.init_pipeline(stream=False) / braven.init(stream=False).
braven_stream: bool = True

_metadata_queue: list[dict] = []

# ADR-0013: the ambient current device set by set_device() — consulted by
# log_config()/log_summary()/log_series()/plot_series()/upload() below so a
# pipeline script can write several values for one device without threading
# an explicit get_device() handle through every call. None ⇒ parent-level
# (the default, and what set_device(None) returns to).
_current_device_key: str | None = None
_current_device_type: str | None = None

# The run handle for the current pipeline (set by init_pipeline), so scripts can
# use the same `run = braven.init(); run.get_device(...)` object model as direct
# logging. See _PipelineRun and the adopt-first rule in init() (Spec 04 §6.1).
_pipeline_run: "_PipelineRun | None" = None

# Convenience dict populated by the executor: filename → local temp path
files: dict[str, str] = {}

# Extracted parameters populated by the executor before process() runs.
# Keys are canonical names (from field mappings); raw key used as fallback.
params: dict = {}

# Column rename map populated by the mapping layer before process() runs.
# Maps raw CSV column names to canonical names: {raw: canonical}
column_maps: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Local dry mode — pipeline scripts are normally written and iterated on
# locally (`python script.py`) before being copy-pasted into the webapp
# Pipeline editor. Historically, any pipeline-vocabulary call
# (log_config/log_summary/log_series/log_artifact/upload/plot_series/
# device()) raised RuntimeError the moment it ran outside the worker, since
# nothing had configured an experiment/API context — breaking that
# copy-paste loop. Now, when _ensure_pipeline_init() finds NEITHER
# braven_experiment_id NOR braven_api_url resolvable (not via explicit
# args, not via BRAVEN_* env vars), it returns ("", "") instead of raising,
# and every write function below prints what it would have done instead.
#
# This can never fire inside a real worker run: worker/executor.py always
# calls braven.init(experiment_id=..., api_url=..., ...) with explicit
# keyword arguments before user code executes, and explicit arguments
# already take precedence over every other source in init_pipeline()'s
# resolution order. A *partially* configured context (one field resolves,
# the other doesn't) is left alone — that's a real misconfiguration, not a
# local script, and still raises exactly as before.
#
# See braven-mvp's docs/adr/0008-local-pipeline-dry-mode.md and
# .scratch/local-pipeline-dry-mode/spec.md for the full design.
# ---------------------------------------------------------------------------

_LOCAL_OUTPUT_DIR = "braven_local_output"

_CATEGORY_TO_LOG_FN = {"config": "log_config", "summary": "log_summary", "series": "log_series"}


def _local_output_path(filename: str) -> Path:
    """Next available path under ./braven_local_output/ for `filename`,
    appending a numeric suffix on collision (figure_1.png, figure_2.png,
    ...) rather than overwriting a previous local run's output — local
    iteration commonly re-runs the same script repeatedly."""
    out_dir = Path(_LOCAL_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    n = 1
    while True:
        candidate = out_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


class _PipelineRun:
    """Run handle returned by braven.init() inside a worker-run pipeline. Its
    logging methods delegate to the module-level pipeline metadata queue, and
    get_device() returns a device-scoped handle — so pipeline scripts can use
    the same run/device object model as direct logging.

        run = braven.init()                 # adopts the worker's experiment
        run.log_summary("max_mismatch", 3)  # experiment-level
        run.get_device("SENSOR-1").log_summary("SNR", 14.2)
    """

    def __init__(self, experiment_id: str | None) -> None:
        self.id = experiment_id

    def get_device(self, key: str, type: str | None = None) -> "Device":
        return Device(str(key), run=None, type=type)

    def log_config(self, key: str, value) -> None:
        _pipeline_log(key, str(value), "config", device_key=_current_device_key, device_type=_current_device_type)

    def log_summary(self, key: str, value) -> None:
        _pipeline_log(key, str(value), "summary", device_key=_current_device_key, device_type=_current_device_type)

    def log_series(self, key: str, values: list) -> None:
        _pipeline_log(key, json.dumps(values), "series", device_key=_current_device_key, device_type=_current_device_type)

    config = log_config
    summary = log_summary

    def plot_series(self, name: str, y: list, x: list | None = None, x_label: str | None = None, y_label: str | None = None, mode: str = "line") -> None:
        _pipeline_plot_series(name, y, x=x, x_label=x_label, y_label=y_label, mode=mode)

    def upload(self, path_or_fig, name: str | None = None, interactive: bool = True) -> None:
        _pipeline_upload(path_or_fig, name, device_key=_current_device_key, device_type=_current_device_type, interactive=interactive)

    def __repr__(self) -> str:
        return f"Run(pipeline, id={self.id!r})"


def init_pipeline(
    experiment_id: str | None = None,
    api_url: str | None = None,
    watcher_secret: str | None = None,
    user_id: str | None = None,
    pipeline_id: str | None = None,
    stream: bool | None = None,
) -> "_PipelineRun":
    """Initialise the pipeline execution context (called by the worker executor).

    Prefer ``braven.init()`` for direct logging from your own scripts. Returns a
    run handle bound to the pipeline's experiment.

    ``stream`` (default: unchanged, i.e. stays at the module default of True)
    controls whether log_config()/log_summary() write immediately or only join
    the buffered end-of-run flush — see the ``braven_stream`` module doc above.
    Pass ``None`` (the default) to leave whatever's already set untouched, same
    as every other field here — only an explicit True/False changes it, so
    _ensure_pipeline_init()'s internal no-args re-entry can never silently
    reset a script's own stream=False choice back to the default.
    """
    global braven_experiment_id, braven_api_url, braven_watcher_secret
    global braven_user_id, braven_pipeline_id, braven_stream, _metadata_queue, column_maps, params, _pipeline_run
    global _current_device_key, _current_device_type
    if stream is not None:
        braven_stream = stream
    braven_experiment_id = experiment_id or braven_experiment_id or os.environ.get("BRAVEN_EXPERIMENT_ID")
    braven_api_url = api_url or braven_api_url or os.environ.get("BRAVEN_API_URL")
    braven_watcher_secret = (
        watcher_secret
        or braven_watcher_secret
        or os.environ.get("BRAVEN_WATCHER_SECRET")
        or os.environ.get("WATCHER_SECRET")
    )
    braven_user_id = user_id or braven_user_id or os.environ.get("BRAVEN_USER_ID")
    if pipeline_id is not None:
        braven_pipeline_id = pipeline_id
    _metadata_queue = []
    column_maps = {}
    params = {}
    # Reset only on a genuine new-run call (experiment_id explicitly passed —
    # the worker's real, once-per-run entry point, executor.py's braven.init(
    # experiment_id=..., ...)), not on _ensure_pipeline_init()'s internal
    # auto-resolve re-entry (no args). In local dry mode, braven_experiment_id
    # never resolves, so _ensure_pipeline_init() calls this on EVERY logging
    # call — an unconditional reset here would silently clear set_device()'s
    # ambient state after just one call, inside a single dry-mode script run.
    if experiment_id is not None:
        _current_device_key = None
        _current_device_type = None
    _pipeline_run = _PipelineRun(braven_experiment_id)
    return _pipeline_run


def _ensure_pipeline_init() -> tuple[str, str]:
    """Resolve the pipeline execution context, calling init_pipeline() first
    to pick up BRAVEN_* env vars if nothing's set yet.

    Returns (experiment_id, api_url) when a context is configured. Returns
    ("", "") — local dry mode, see the section above — only when NEITHER
    field resolves to anything; callers must check the returned
    experiment_id and, if empty, print instead of making a network call. A
    partially-configured context (one field resolves, the other doesn't)
    is a real misconfiguration, not a local script, and still raises.
    """
    if not braven_experiment_id or not braven_api_url:
        init_pipeline()
    if not braven_experiment_id and not braven_api_url:
        return "", ""
    if not braven_experiment_id:
        raise RuntimeError(
            "braven experiment_id not set. The worker should have called braven.init_pipeline(), "
            "or set BRAVEN_EXPERIMENT_ID in the environment."
        )
    if not braven_api_url:
        raise RuntimeError(
            "braven api_url not set. Set BRAVEN_API_URL in the environment."
        )
    return braven_experiment_id, braven_api_url


def _pipeline_auth_headers() -> dict:
    if not braven_watcher_secret:
        return {}
    return {"Authorization": f"Bearer {braven_watcher_secret}"}


def log_config(key: str, value: str) -> None:
    """Queue a config parameter for this pipeline run — for the ambient
    current device (see set_device()) if one is set, else parent-level."""
    _pipeline_log(key, value, "config", device_key=_current_device_key, device_type=_current_device_type)


def log_summary(key: str, value: str) -> None:
    """Queue a summary metric for this pipeline run — for the ambient current
    device (see set_device()) if one is set, else parent-level."""
    _pipeline_log(key, value, "summary", device_key=_current_device_key, device_type=_current_device_type)


def log(key: str, value: str) -> None:
    """REMOVED (2026-07-16) — the ambiguous log() alias is gone so scripts
    state their intent explicitly. Raises with migration guidance instead of
    an opaque AttributeError, since older stored pipeline scripts still call
    it."""
    raise RuntimeError(
        "braven.log() has been removed - use braven.log_summary(key, value) "
        "for output metrics or braven.log_config(key, value) for input parameters."
    )


def log_metadata(key: str, value: str) -> None:
    """Alias for log_config() (kept for backwards compatibility)."""
    _pipeline_log(key, value, "config", device_key=_current_device_key, device_type=_current_device_type)


def path_params() -> dict:
    """Return parameters extracted deterministically from experiment file paths.

    Scans the filenames in ``braven.files`` for common scientific naming
    conventions (e.g. ``30C``, ``3v3``, ``run_5``) and returns them as a dict.
    """
    import re as _re
    _PATH_PATTERNS = [
        (_re.compile(r"(\d+(?:\.\d+)?)\s*[Cc](?:\b|_|$)"),  "temperature", float),
        (_re.compile(r"(\d+(?:\.\d+)?)\s*[Vv](?:\b|_|$)"),  "vdd",         float),
        (_re.compile(r"(\d+(?:\.\d+)?)\s*m[Aa](?:\b|_|$)"), "current_ma",  float),
        (_re.compile(r"(\d+(?:\.\d+)?)\s*[Aa](?:\b|_|$)"),  "current_a",   float),
        (_re.compile(r"[Rr]un[_-]?(\d+)"),                   "run_index",   int),
        (_re.compile(r"[Ss]tep[_-]?(\d+)"),                  "step",        int),
        (_re.compile(r"[Ss]ample[_-]?(\d+)"),                "sample",      int),
        (_re.compile(r"(\d+)\s*deg(?:rees?)?(?:\b|_)"),      "angle_deg",   float),
        (_re.compile(r"(\d+(?:\.\d+)?)\s*[Mm][Hh]z"),       "freq_mhz",    float),
        (_re.compile(r"(\d+(?:\.\d+)?)\s*[Kk][Hh]z"),       "freq_khz",    float),
    ]
    result: dict = {}
    for filename in files.keys():
        for segment in filename.replace("\\", "/").split("/"):
            for pattern, key, cast in _PATH_PATTERNS:
                if key in result:
                    continue
                m = pattern.search(segment)
                if m:
                    try:
                        result[key] = cast(m.group(1))
                    except (ValueError, IndexError):
                        pass
    return result


def log_series(key: str, values: list) -> None:
    """Queue a numeric data series (stored as JSON string) — for the ambient
    current device (see set_device()) if one is set, else parent-level."""
    _pipeline_log(key, json.dumps(values), "series", device_key=_current_device_key, device_type=_current_device_type)


def _pipeline_plot_series(
    name: str,
    y: list,
    x: list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    mode: str = "line",
) -> None:
    """Pipeline-context implementation backing the module-level plot_series()
    dispatcher — for the ambient current device (see set_device()) if one is
    set, else parent-level."""
    for key, value, category in _build_plot_series_entries(name, y, x, x_label, y_label, mode):
        _pipeline_log(key, value, category, device_key=_current_device_key, device_type=_current_device_type)


def _pipeline_log(key: str, value: str, category: str, device_key: str | None = None, device_type: str | None = None) -> None:
    experiment_id, api_url = _ensure_pipeline_init()
    if not experiment_id:  # local dry mode — see the section above
        fn = _CATEGORY_TO_LOG_FN.get(category, "log_config")
        suffix = f", device={device_key!r}" if device_key else ""
        print(f"[braven:local] would {fn}({key!r}, {value!r}{suffix})", flush=True)
        return
    global _metadata_queue
    # Dedup by (key, device_key): a device's value only replaces the same device's
    # earlier value for that key, never another device's (Spec 04 long format).
    _metadata_queue = [e for e in _metadata_queue if not (e["key"] == key and e.get("device_key") == device_key)]
    _metadata_queue.append({"key": key, "value": str(value), "category": category, "device_key": device_key, "device_type": device_type})
    # Streaming mode (default on): also write this one entry immediately —
    # scoped to Experiment-level, non-series entries only. See
    # _stream_one_entry's docstring for why device-tagged/series entries are
    # excluded regardless of braven_stream, and why this is safe/cheap to do
    # per call (unlike calling flush_metadata() itself here would be).
    if braven_stream and device_key is None and category != "series":
        _stream_one_entry(experiment_id, api_url, key, str(value), category)


def _stream_one_entry(experiment_id: str, api_url: str, key: str, value: str, category: str) -> None:
    """Best-effort immediate write for one Experiment-level, non-series entry
    (streaming mode). Hits a separate, narrower endpoint than flush_metadata()
    — PATCH .../pipeline-metadata/{id}, an incremental upsert scoped to just
    this key — rather than that endpoint's PUT, which re-derives and rewrites
    the WHOLE accumulated flush every call (O(total entries so far) per call,
    O(n²) over a run — fine once at the end, not once per log statement).
    Device-tagged and series entries are excluded from streaming and stay
    exclusively on the buffered flush_metadata() path: device entries need
    real per-flush resolution work (company/child-Experiment routing) worth
    doing once, and Series lives in R2 (ADR-0001) as a read-modify-write
    object, not a row — streaming those would mean a full object write per
    log_series()/plot_series() call.

    Never raises: a transient failure here just means this value shows up
    once the run finishes instead of live — flush_metadata() at run end is
    still the source of truth and rewrites it correctly regardless.
    """
    if not braven_pipeline_id:
        return
    try:
        resp = requests.patch(
            f"{api_url.rstrip('/')}/experiments/{experiment_id}/pipeline-metadata/{braven_pipeline_id}",
            json=[{"key": key, "value": value, "category": category}],
            headers=_pipeline_auth_headers(),
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[braven] streaming flush skipped for {key!r}: {e}", flush=True)


def _describe_device_report(report: dict) -> str:
    """One-line human summary of what a flush created / flagged (Spec device-policy
    ticket 06), e.g. 'created 2 new devices: S-1, S-2; 1 type mismatch: …'."""
    parts = []
    keys = report.get("createdDeviceKeys") or []
    types = report.get("createdTypeNames") or []
    mism = report.get("typeMismatches") or []
    if keys:
        parts.append(f"created {len(keys)} new device(s): {', '.join(keys)}")
    if types:
        parts.append(f"created {len(types)} new type(s): {', '.join(types)}")
    if mism:
        joined = "; ".join(f"{m.get('key')} said '{m.get('requested')}' but is '{m.get('stored')}'" for m in mism)
        parts.append(f"{len(mism)} device type mismatch(es): {joined}")
    return "; ".join(parts)


def flush_metadata(pipeline_id: str | None = None) -> dict | None:
    """Atomically replace all pipeline-scoped metadata (called by the executor).
    Returns the backend's device created-report ({createdDeviceKeys,
    createdTypeNames, typeMismatches}) so the worker can fold it into the run
    result, and prints a human line to stdout when the run created/flagged
    devices. Returns None when there's nothing to flush."""
    experiment_id, api_url = _ensure_pipeline_init()
    if not experiment_id:  # local dry mode — see the section above
        print("[braven:local] flush_metadata() — nothing to flush (local dry mode)", flush=True)
        return None
    entries = list(_metadata_queue)
    if not entries:
        return None
    pid = pipeline_id or braven_pipeline_id
    if not pid:
        raise RuntimeError("flush_metadata: pipeline_id is required")
    resp = requests.put(
        f"{api_url.rstrip('/')}/experiments/{experiment_id}/pipeline-metadata/{pid}",
        json=entries,
        headers=_pipeline_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    try:
        report = resp.json().get("device_report")
    except Exception:
        report = None
    if report:
        desc = _describe_device_report(report)
        if desc:
            print(f"[braven] {desc}", flush=True)
    return report


def log_plot(fig, name: str) -> None:
    """REMOVED (2026-07-16) — Plotly figure upload is gone; matplotlib is
    the plotting API. Raises with migration guidance instead of an opaque
    AttributeError, since older auto-generated pipeline scripts still call
    it."""
    raise RuntimeError(
        "braven.log_plot() (Plotly) has been removed - use "
        "braven.upload(fig, name) with a matplotlib Figure instead."
    )


def _pipeline_upload(path_or_fig, name: str | None = None, device_key: str | None = None, device_type: str | None = None, interactive: bool = True) -> None:
    """Pipeline-mode twin of Run.upload(): upload a file (or a live
    matplotlib Figure) attached to the current pipeline experiment — or, with
    `device_key` set (Device.upload()/the ambient current device, ADR-0013),
    to that device's own child Experiment; the backend resolves the child
    from `device_key` server-side, same as it does for metadata's device_key.
    For a Figure, the PNG is saved the same way fig.savefig() would, and
    (when `interactive` is True, the default) an interactive companion series
    is extracted from the figure's line/scatter data (best-effort — queued as
    series metadata and written by the final flush_metadata(), never blocking
    the upload itself)."""
    fig = path_or_fig if _is_matplotlib_figure(path_or_fig) else None
    if fig is not None:
        upload_name = name or "figure.png"
    else:
        file_path = Path(path_or_fig)
        if not file_path.exists():
            raise FileNotFoundError(f"upload: file not found: {file_path}")
        upload_name = name or file_path.name

    experiment_id, api_url = _ensure_pipeline_init()
    if not experiment_id:  # local dry mode — see the section above
        suffix = f", device={device_key!r}" if device_key else ""
        if fig is not None:
            out_path = _local_output_path(upload_name)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[braven:local] saved figure to {out_path} (upload skipped{suffix})", flush=True)
        else:
            print(f"[braven:local] would upload({upload_name!r}{suffix})", flush=True)
        return

    tmp_path: Path | None = None
    if fig is not None:
        tmp_path = Path(tempfile.mktemp(suffix=".png"))
        fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
        file_path = tmp_path

    params: dict = {"experiment_id": experiment_id}
    if braven_pipeline_id:
        params["pipeline_id"] = braven_pipeline_id
    if device_key:
        params["device_key"] = device_key
        if device_type:
            params["device_type"] = device_type
    with open(file_path, "rb") as fh:
        resp = requests.post(
            f"{api_url.rstrip('/')}/files",
            files=_upload_file_parts(upload_name, fh, file_path),
            params=params,
            headers=_pipeline_auth_headers(),
            timeout=120,
        )
    resp.raise_for_status()

    if tmp_path is not None:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if fig is not None and interactive:
        try:
            groups = _walk_matplotlib_figure(fig)
            base = _safe_filename_sdk(upload_name)
            for g in groups:
                for trace_idx, tr in enumerate(g["traces"]):
                    prefix = f"mplplot::{base}::{g['axes_index']}::{trace_idx}"
                    _pipeline_log(f"{prefix}::y", json.dumps(tr["y"]), "series", device_key=device_key, device_type=device_type)
                    _pipeline_log(f"{prefix}::x", json.dumps(tr["x"]), "series", device_key=device_key, device_type=device_type)
                    meta = {
                        "mode": tr["mode"],
                        "trace_label": tr["label"],
                        "x_label": g["x_label"],
                        "y_label": g["y_label"],
                        "x_scale": g["x_scale"],
                        "y_scale": g["y_scale"],
                    }
                    _pipeline_log(f"{prefix}::meta", json.dumps(meta), "series", device_key=device_key, device_type=device_type)
        except Exception as e:
            print(f"[braven] matplotlib series extraction skipped: {e}")


def log_artifact(path: Union[str, Path], name: str | None = None) -> None:
    """Upload a file attached to the current pipeline experiment."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"log_artifact: file not found: {file_path}")
    upload_name = name or file_path.name

    experiment_id, api_url = _ensure_pipeline_init()
    if not experiment_id:  # local dry mode — see the section above
        print(f"[braven:local] would log_artifact({upload_name!r})", flush=True)
        return

    params: dict = {"experiment_id": experiment_id}
    if braven_pipeline_id:
        params["pipeline_id"] = braven_pipeline_id
    with open(file_path, "rb") as fh:
        resp = requests.post(
            f"{api_url.rstrip('/')}/files",
            files=_upload_file_parts(upload_name, fh, file_path),
            params=params,
            headers=_pipeline_auth_headers(),
            timeout=120,
        )
    resp.raise_for_status()


# Keep the old `init` name working for pipeline scripts that call braven.init()
# with keyword arguments matching the pipeline signature.
_direct_init = init


def init(*args, **kwargs):  # type: ignore[misc]
    """Unified init: routes to init_pipeline() for pipeline scripts, or the
    direct-logging init for standalone scripts.

    Pipeline usage (called by worker executor):
        braven.init(experiment_id="...", api_url="...", ...)

    Direct-logging usage:
        braven.init(name="My Experiment")

    Adopt-first rule (Spec 04 §6.1): inside a worker-run pipeline the experiment
    is pre-created by the worker, so a bare ``braven.init()`` in a user script
    ADOPTS that existing experiment (returning its run handle) instead of POSTing
    a new one — no credentials/project needed, and existing scripts that never
    call init() are unaffected. (Creating additional experiments from one pipeline
    is not supported yet; a repeat init() returns the same adopted run.)
    """
    pipeline_keys = {"experiment_id", "api_url", "watcher_secret", "user_id", "pipeline_id", "stream"}
    if args or (kwargs and pipeline_keys.intersection(kwargs)):
        return init_pipeline(*args, **kwargs)
    # Bare init() while a pipeline context is active → adopt the pre-created run.
    if braven_experiment_id and _pipeline_run is not None:
        return _pipeline_run
    return _direct_init(**kwargs)


# ---------------------------------------------------------------------------
# Query API — read existing experiments
# ---------------------------------------------------------------------------


def _parse_iso(s: str) -> datetime:
    """datetime.fromisoformat() only accepts a trailing 'Z' from Python 3.11
    on — the backend's JSON timestamps use it (JS Date.toISOString()), and
    this SDK declares no Python version floor, so normalize it ourselves
    rather than assuming a modern interpreter."""
    return datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith("Z") else s)


class BravenFile:
    def __init__(self, file_id: str, filename: str, uploaded_at: str, base_url: str, headers: dict, timeout: int):
        self.id = file_id
        self.filename = filename
        self.uploaded_at = _parse_iso(uploaded_at)
        self._base_url = base_url
        self._headers = headers
        self._timeout = timeout

    def download(self) -> bytes:
        """Download and return raw bytes."""
        r = requests.get(f"{self._base_url}/files/{self.id}/download", headers=self._headers, timeout=self._timeout)
        r.raise_for_status()
        return r.content

    def save(self, dest: Union[str, Path]) -> None:
        """Download and save to disk."""
        Path(dest).write_bytes(self.download())

    def as_dataframe(self):
        """Parse as CSV and return a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")
        if not self.filename.lower().endswith(".csv"):
            raise ValueError(f"{self.filename!r} is not a CSV file")
        return pd.read_csv(io.BytesIO(self.download()))

    def __repr__(self) -> str:
        return f"BravenFile(filename={self.filename!r}, id={self.id!r})"


class Experiment:
    def __init__(self, data: dict, base_url: str, headers: dict, timeout: int):
        self.id: str = data["id"]
        self.name: str | None = data.get("name")
        self.notes: str | None = data.get("notes")
        self.created_at: datetime = _parse_iso(data["created_at"])
        self.metadata: dict[str, str | None] = {m["key"]: m["value"] for m in data.get("metadata", [])}
        self.files: list[BravenFile] = [
            BravenFile(f["id"], f["filename"], f["uploaded_at"], base_url, headers, timeout)
            for f in data.get("files", [])
        ]

    def file(self, filename: str) -> BravenFile:
        for f in self.files:
            if f.filename == filename:
                return f
        raise FileNotFoundError(
            f"No file {filename!r} in experiment {self.name or self.id!r}. "
            f"Available: {[f.filename for f in self.files]}"
        )

    def __repr__(self) -> str:
        return f"Experiment(name={self.name!r}, id={self.id!r})"


class ExperimentSummary:
    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.name: str | None = data.get("name")
        self.notes: str | None = data.get("notes")
        self.created_at: datetime = _parse_iso(data["created_at"])
        self.file_count: int = data.get("file_count", 0)
        self.metadata: dict[str, str | None] = {m["key"]: m["value"] for m in data.get("metadata", [])}

    def __repr__(self) -> str:
        return f"ExperimentSummary(name={self.name!r}, id={self.id!r})"


class Braven:
    """Read-only client for querying the Braven API.

    Args:
        url:     Backend URL, e.g. "https://api.bravenlab.com"
        api_key: Your ``braven_`` API key.
        timeout: Request timeout in seconds.
    """

    def __init__(self, url: str, api_key: str = "", timeout: int = 30):
        self._base_url = url.rstrip("/")
        self._timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def experiments(self) -> list[ExperimentSummary]:
        """Return lightweight summaries of all experiments."""
        return [ExperimentSummary(d) for d in self._get("/experiments")]

    def get(self, name_or_names: Union[str, list[str]]) -> Union[Experiment, list[Experiment]]:
        """Fetch one or more experiments by partial, case-insensitive name."""
        summaries = self._get("/experiments")
        if isinstance(name_or_names, list):
            return [self._resolve_one(n, summaries) for n in name_or_names]
        return self._resolve_one(name_or_names, summaries)

    def _resolve_one(self, name: str, summaries: list[dict]) -> Experiment:
        needle = name.lower()
        matches = [s for s in summaries if s.get("name") and needle in s["name"].lower()]
        if not matches:
            raise ValueError(f"No experiment matching {name!r}. Available: {[s.get('name') for s in summaries if s.get('name')]}")
        if len(matches) > 1:
            raise ValueError(f"Multiple experiments match {name!r}: {[s.get('name') for s in matches]}. Use a more specific name.")
        detail = self._get(f"/experiments/{matches[0]['id']}")
        return Experiment(detail, self._base_url, self._headers, self._timeout)

    def _get(self, path: str):
        r = requests.get(self._base_url + path, headers=self._headers, timeout=self._timeout)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Analysis — cross-Experiment egress with SDK loop-back
#
# collect() is meant to be called only from local, direct-logging usage — an
# Analysis is built from local work done OUTSIDE Braven (a script on your own
# machine, reading data pulled out via collect()), so it's not part of the
# pipeline-script execution context (server-side, run by the Braven worker
# against uploaded files). Nothing in this module enforces that boundary;
# it's a design intent, not a guard.
# ---------------------------------------------------------------------------


def _decode_filter_token(token: str) -> dict:
    """Decodes the opaque `filter_token` the webapp's "Copy as code" action
    embeds in a generated snippet (ticket 04): base64url(JSON({project_id,
    filter})). Opaque by convention, not by security — there is nothing
    secret in a filter description, this just keeps it out of the way of a
    hand-editable experiment_ids=[...] call."""
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid filter_token: {e}") from e


class AnalysisExperiment:
    """One of an Analysis's frozen source Experiments. Config/Summary metadata
    is already loaded (Analysis fetched it eagerly at collect()-time); Series
    and the raw uploaded source files are fetched lazily, on first access —
    large payloads a local script may never touch."""

    def __init__(self, experiment_id: str, data: dict, analysis_id: str, base_url: str, headers: dict, timeout: int) -> None:
        self.id = experiment_id
        self.name: str | None = data.get("name")
        self.notes: str | None = data.get("notes")
        self.metadata: dict[str, str | None] = {m["key"]: m["value"] for m in data.get("metadata", [])}
        self._file_records = data.get("files", [])
        self._analysis_id = analysis_id
        self._base_url = base_url
        self._headers = headers
        self._timeout = timeout
        self._series: dict[str, str | None] | None = None

    @property
    def series(self) -> dict[str, str | None]:
        """Parsed measurement arrays for this Experiment: {key: JSON-encoded
        value}. Fetched once, on first access, and cached."""
        if self._series is None:
            r = requests.get(
                f"{self._base_url}/analyses/{self._analysis_id}/experiments/{self.id}/series",
                headers=self._headers,
                timeout=self._timeout,
            )
            r.raise_for_status()
            self._series = {e["key"]: e["value"] for e in r.json().get("entries", [])}
        return self._series

    def files(self) -> list[BravenFile]:
        """This Experiment's originally uploaded source files (optional —
        only fetched/downloaded if you call .download()/.save() on one)."""
        return [
            BravenFile(f["id"], f["filename"], f["uploaded_at"], self._base_url, self._headers, self._timeout)
            for f in self._file_records
        ]

    def file(self, filename: str) -> BravenFile:
        for f in self.files():
            if f.filename == filename:
                return f
        raise FileNotFoundError(
            f"No file {filename!r} in experiment {self.name or self.id!r}. "
            f"Available: {[f['filename'] for f in self._file_records]}"
        )

    def __repr__(self) -> str:
        return f"AnalysisExperiment(name={self.name!r}, id={self.id!r})"


class Analysis:
    """Returned by braven.collect(...) — both the extracted data (read) and
    the object results are logged back onto (write), so the record of what
    this Analysis is about can never drift from what was actually read.

    Logging mirrors pipeline-mode vocabulary (log_config/log_summary/
    log_artifact/upload), matching the ADR's "same output vocabulary" intent
    — NOT Run's own .config()/.summary() names. Each call writes immediately;
    there is no buffered finish() to forget.
    """

    def __init__(self, data: dict, api_url: str, headers: dict, timeout: int = 30) -> None:
        self.id: str = data["analysis_id"]
        self.name: str = data["name"]
        self.company_id: str = data["company_id"]
        self.experiment_ids: list[str] = list(data["experiment_ids"])
        self._base_url = api_url.rstrip("/")
        self._headers = headers
        self._timeout = timeout

        # Eager Config/Summary fetch for every resolved Experiment (small) —
        # Series/files stay lazy on each AnalysisExperiment (large).
        self._experiments: dict[str, AnalysisExperiment] = {}
        for experiment_id in self.experiment_ids:
            detail = self._get(f"/analyses/{self.id}/experiments/{experiment_id}")
            self._experiments[experiment_id] = AnalysisExperiment(
                experiment_id, detail, self.id, self._base_url, self._headers, self._timeout
            )

    def experiments(self) -> list[AnalysisExperiment]:
        """All resolved source Experiments, in the order collect() froze them."""
        return [self._experiments[eid] for eid in self.experiment_ids]

    def experiment(self, experiment_id: str) -> AnalysisExperiment:
        return self._experiments[experiment_id]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_config(self, key: str, value) -> None:
        """Log an input / configuration parameter on this Analysis."""
        self._log_metadata(key, value, "config")

    def log_summary(self, key: str, value) -> None:
        """Log an output metric (e.g. accuracy) on this Analysis."""
        self._log_metadata(key, value, "summary")

    def log_artifact(self, path: Union[str, Path], name: str | None = None) -> None:
        """Upload a file (e.g. a serialized model) attached to this Analysis."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"log_artifact: file not found: {file_path}")
        self._upload_file(file_path, name or file_path.name)

    def upload(self, path_or_fig, name: str | None = None, interactive: bool = True) -> None:
        """Upload a file and attach it to this Analysis.

        `path_or_fig` may be a file path, or a live matplotlib Figure — in the
        latter case the PNG is saved the same way `fig.savefig()` would, and
        (when `interactive` is True, the default) an interactive companion
        series is extracted from the figure's line/scatter data (best-effort,
        never blocks the upload) — the same upload() pattern Runs already use.
        """
        fig = path_or_fig if _is_matplotlib_figure(path_or_fig) else None
        tmp_path: Path | None = None
        if fig is not None:
            tmp_path = Path(tempfile.mktemp(suffix=".png"))
            fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
            file_path = tmp_path
            upload_name = name or "figure.png"
        else:
            file_path = Path(path_or_fig)
            if not file_path.exists():
                raise FileNotFoundError(f"upload: file not found: {file_path}")
            upload_name = name or file_path.name

        self._upload_file(file_path, upload_name)

        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if fig is not None and interactive:
            self._log_matplotlib_series(fig, upload_name)

    def _upload_file(self, file_path: Path, upload_name: str) -> None:
        # A bare (filename, fileobj) 2-tuple omits the part's Content-Type
        # header entirely — verified the backend then stores an incorrect
        # default rather than guessing from the filename itself, so the
        # mimetype is resolved here instead.
        content_type = mimetypes.guess_type(upload_name)[0] or "application/octet-stream"
        with open(file_path, "rb") as fh:
            resp = requests.post(
                f"{self._base_url}/analyses/{self.id}/files",
                files=_upload_file_parts(upload_name, fh, file_path, content_type=content_type),
                headers=self._headers,
                timeout=120,
            )
        resp.raise_for_status()

    def _log_metadata(self, key: str, value, category: str) -> None:
        resp = requests.put(
            f"{self._base_url}/analyses/{self.id}/metadata",
            json=[{"key": key, "value": str(value), "category": category}],
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()

    def _log_matplotlib_series(self, fig, image_name: str) -> None:
        """Best-effort: extract interactive series data from a matplotlib figure and log
        it alongside the PNG already uploaded. Never raises — a failure here must never
        affect the (already-successful) file upload. Mirrors Run._log_matplotlib_series."""
        try:
            groups = _walk_matplotlib_figure(fig)
            if not groups:
                return
            base = _safe_filename_sdk(image_name)
            entries: list[dict] = []
            for g in groups:
                for trace_idx, tr in enumerate(g["traces"]):
                    prefix = f"mplplot::{base}::{g['axes_index']}::{trace_idx}"
                    entries.append({"key": f"{prefix}::y", "value": json.dumps(tr["y"]), "category": "series"})
                    entries.append({"key": f"{prefix}::x", "value": json.dumps(tr["x"]), "category": "series"})
                    meta = {
                        "mode": tr["mode"],
                        "trace_label": tr["label"],
                        "x_label": g["x_label"],
                        "y_label": g["y_label"],
                        "x_scale": g["x_scale"],
                        "y_scale": g["y_scale"],
                    }
                    entries.append({"key": f"{prefix}::meta", "value": json.dumps(meta), "category": "series"})
            resp = requests.put(
                f"{self._base_url}/analyses/{self.id}/metadata",
                json=entries,
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"[braven] matplotlib series extraction skipped: {e}")

    def _get(self, path: str):
        r = requests.get(self._base_url + path, headers=self._headers, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def __repr__(self) -> str:
        return f"Analysis(name={self.name!r}, id={self.id!r}, experiments={len(self.experiment_ids)})"


def collect(
    experiment_ids: list[str] | None = None,
    filter_token: str | None = None,
    name: str | None = None,
) -> Analysis:
    """Pull a cross-Experiment dataset out of Braven and get back the object
    used to log results (accuracy, a plot, a model file) back onto it.

    Provide exactly one of:
        experiment_ids: an explicit list of Experiment ids — frozen exactly
                         as given.
        filter_token:   an opaque token copied from the webapp's "Copy as
                         code" action — re-resolved to concrete Experiment
                         ids now, at call time (never stored as a live query).

    Every resolved Experiment must belong to the same Company, and you must
    be a member of it — checked before any data is pulled. Authenticates the
    same way `python -m braven login` sets up (~/.braven/config.json); no
    project needed, Company is derived from the referenced Experiments.
    """
    if (experiment_ids is None) == (filter_token is None):
        raise ValueError("collect(): pass exactly one of experiment_ids or filter_token")
    if not name:
        raise ValueError("collect(): name is required")

    cfg = _load_config()
    api_url = cfg["api_url"].rstrip("/")
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}

    body: dict = {"name": name}
    if experiment_ids is not None:
        body["experiment_ids"] = list(experiment_ids)
    else:
        decoded = _decode_filter_token(filter_token)
        body["project_id"] = decoded.get("project_id")
        body["filter"] = decoded.get("filter")

    resp = requests.post(f"{api_url}/analyses", json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    return Analysis(resp.json(), api_url, headers)


# ---------------------------------------------------------------------------
# CLI — python -m braven login
# ---------------------------------------------------------------------------

def _cli_login() -> None:
    print("Braven login\n" + "─" * 40)

    existing: dict = {}
    if _CONFIG_PATH.exists():
        try:
            existing = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    default_url = existing.get("api_url", _DEFAULT_API_URL)
    raw_url = input(f"API URL [{default_url}]: ").strip().rstrip("/")
    api_url = raw_url if raw_url else default_url

    api_key = getpass.getpass("SDK API key (braven_...): ").strip()
    if not api_key:
        api_key = existing.get("api_key", "")

    if not api_key:
        print("API key is required.")
        return

    # Verify connectivity only — no auth required for /health
    try:
        requests.get(f"{api_url}/health", timeout=10).raise_for_status()
    except Exception as e:
        print(f"\nFailed to connect: {e}")
        return

    login(api_url, api_key)
    print(f"\nLogged in. Credentials saved to {_CONFIG_PATH}")
    print("\nIn your scripts, specify the company and project you want to log to:")
    print('  braven.init(name="My Experiment", company="My Company", project="My Project")')


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "login":
        _cli_login()
    else:
        print("Usage: python -m braven login")
        sys.exit(1)
