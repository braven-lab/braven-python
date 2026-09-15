"""
braven.py — Python SDK for the Braven experiment tracker.

Every logging call goes through one `Run`, returned by `braven.init()`. The
only thing that differs between a direct-logging script and a pipeline
script is how the Experiment is identified — everything after that (config,
summary, series, plots, uploads, devices) is the same object, same methods.

DIRECT LOGGING (wandb-style, your own script)
────────────────────────────────────────────────
One-time setup:
    python -m braven login

Then in any script:
    import braven

    run = braven.init(name="My Experiment")
    run.config("learning_rate", 0.001)
    run.summary("accuracy", 0.94)
    run.upload("plot.png")
    run.finish()

QUERYING EXPERIMENTS
─────────────────────
    from braven import Braven

    b = Braven("https://api.bravenlab.com", api_key="braven_...")
    exp = b.get("high temp run")
    df  = exp.file("data.csv").as_dataframe()

PIPELINE SCRIPTS (run by the Braven worker)
─────────────────────────────────────────────
    import braven

    run = braven.init()   # adopts the Experiment the platform already created
    run.config("lr", "0.001")
    run.summary("acc", "0.94")
    run.upload("plot.png")
    run.table("readings", dataframe)   # named, columnar Table (pandas)

    # `run = braven.init()` also makes local iteration work: run the same
    # script standalone (`python script.py`, no worker) before pasting it
    # into the webapp editor, and every call above prints instead of raising.

MULTIPLE DEVICES IN ONE PIPELINE RUN (ADR-0013)
──────────────────────────────────────────────
    # set_device() retargets `run` onto that device's own child Experiment —
    # the KPI name stays the same for every device (no "SNR_dev1"), each
    # device just gets its own child Experiment under the run's (parent) one.
    for sensor_id, snr in results.items():
        run.set_device(sensor_id)
        run.summary("SNR", snr)
    run.set_device(None)                       # back to the parent
    run.summary("max_device_mismatch", spread)  # parent-level
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
import time
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
# overwrite the first trace's key, since Run._metadata dedupes by exact key.
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
# Table (braven-mvp's CONTEXT.md/ADR-0016) — named, columnar Run output
# distinct from series() (numeric x/y traces only): can mix column types,
# logged wholesale via table(name, dataframe). Pandas stays an optional
# dependency (`pip install braven[dataframe]`), same instinct as matplotlib
# below — never imported at module load time.
# ---------------------------------------------------------------------------


def _pandas_dtype_to_table_type(dtype) -> str:
    """Map a pandas dtype to one of Table's column types (int/float/string/
    bool/datetime) — see braven-mvp's backend-ts/tableBundle.ts for the
    matching set on the receiving end."""
    kind = getattr(dtype, "kind", "O")
    if kind in ("i", "u"):
        return "int"
    if kind == "f":
        return "float"
    if kind == "b":
        return "bool"
    if kind == "M":
        return "datetime"
    return "string"


def _table_payload_from_dataframe(dataframe) -> dict:
    """Convert a pandas DataFrame into the {columns, rows} JSON payload
    table() sends. Column names/types are read directly off the DataFrame's
    own columns/dtypes — no separate hand-declared schema, the same
    pivot-off-a-well-known-object pattern upload() uses for a live
    matplotlib Figure."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for table(): pip install braven[dataframe]")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"table() expects a pandas DataFrame, got {type(dataframe).__name__}")

    columns = [{"name": str(c), "type": _pandas_dtype_to_table_type(dataframe[c].dtype)} for c in dataframe.columns]

    def cell(value, col_type: str):
        try:
            is_na = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_na = False
        if is_na:
            return None
        if col_type == "int":
            return int(value)
        if col_type == "float":
            return float(value)
        if col_type == "bool":
            return bool(value)
        if col_type == "datetime":
            return value.isoformat()
        return str(value)

    col_types = [c["type"] for c in columns]
    rows = [[cell(row[i], col_types[i]) for i in range(len(columns))] for row in dataframe.itertuples(index=False, name=None)]
    return {"columns": columns, "rows": rows}


def _build_table_entry(name: str, dataframe) -> tuple[str, str, str]:
    """Build the (key, value, category) triple table() sends — the whole
    Table rides one generic log-entry, same wire mechanism config()/
    summary()/series() already use (no new ingestion endpoint)."""
    return (name, json.dumps(_table_payload_from_dataframe(dataframe)), "table")


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
    the original in R2 (see _make_image_preview). Every upload path — Run.upload()
    and Analysis.upload() — routes through this one function so a preview is
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


_LOCAL_OUTPUT_DIR = "braven_local_output"

# Debounce interval for a device-scoped (set_device()-active) streaming
# flush — see Run._queue_device_stream_entry's docstring for why device
# entries are batched on a cadence instead of streamed one-by-one the way a
# parent-level entry is. Overridable per run via
# braven.init(stream_interval=...) or run.settings(stream_interval=...).
DEVICE_STREAM_INTERVAL_S = 2.0


# ---------------------------------------------------------------------------
# Run — the one logging handle, for both direct-logging and pipeline scripts
# ---------------------------------------------------------------------------


class Run:
    """A live handle to one Experiment. Returned by braven.init() — either
    freshly created (direct logging: `braven.init(name=..., project=...)`)
    or adopted from a worker-run pipeline's already-existing Experiment
    (`braven.init()`, bare). Every method below works identically regardless
    of which one — the only difference between the two is how the Experiment
    was identified in the first place.

        run = braven.init(name="Firefly temp sweep")   # or braven.init() in a pipeline
        run.config("ambient_C", 21.0)                    # parent-level
        run.summary("max_mismatch", 3.1)                 # parent-level

        run.set_device("SENSOR-4471")                    # retarget onto a child Experiment
        run.summary("SNR", 14.2)                         # this device's own child
        run.upload(fig, "spectrum.png")                  # also the child's
        run.set_device(None)                             # back to the parent

        run.finish()

    Do not construct directly — always via braven.init().
    """

    def __init__(
        self,
        *,
        mode: str,
        api_url: str | None,
        headers: dict,
        experiment_id: str | None,
        name: str | None = None,
        pipeline_id: str | None = None,
        flush: bool = True,
        stream_interval: float | None = None,
    ) -> None:
        self._mode = mode  # "created" (direct-logging) | "adopted" (pipeline)
        self._base_url = api_url.rstrip("/") if api_url else None
        self._headers = headers
        self.id = experiment_id
        self.name = name
        self._pipeline_id = pipeline_id
        self._auto_flush = flush
        self._stream_interval = DEVICE_STREAM_INTERVAL_S if stream_interval is None else stream_interval

        # (device_key, key) → {key, value, category, device_key, device_type}.
        # device_key is None for parent-level values; set while set_device()
        # is active, so two devices' "SNR" are distinct entries rather than
        # one overwriting the other. The backend resolves device_key to that
        # device's own child Experiment (ADR-0013).
        self._metadata: dict[tuple[str | None, str], dict] = {}
        self._device_target: str | None = None
        self._device_target_type: str | None = None
        self._device_stream_queue: list[dict] = []
        self._device_stream_last_flush_at: float = 0.0

    # ------------------------------------------------------------------
    # Construction (called by braven.init() — not part of the public API)
    # ------------------------------------------------------------------

    @classmethod
    def _create(cls, name, notes, project, company, flush, stream_interval) -> "Run":
        cfg = _load_config()
        project_id = _resolve_project_id(cfg, company=company, project=project)
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "X-Project-Id": project_id}
        payload = {"name": name, "notes": notes}
        payload.update(_capture_script_info())
        resp = requests.post(f"{cfg['api_url'].rstrip('/')}/experiments", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return cls(
            mode="created",
            api_url=cfg["api_url"],
            headers=headers,
            experiment_id=data["id"],
            name=data.get("name"),
            flush=flush,
            stream_interval=stream_interval,
        )

    @classmethod
    def _adopted(cls, experiment_id, api_url, watcher_secret, pipeline_id, flush, stream_interval) -> "Run":
        headers = {"Authorization": f"Bearer {watcher_secret}"} if watcher_secret else {}
        return cls(
            mode="adopted",
            api_url=api_url,
            headers=headers,
            experiment_id=experiment_id,
            pipeline_id=pipeline_id,
            flush=flush,
            stream_interval=stream_interval,
        )

    @property
    def _local_dry_mode(self) -> bool:
        # Only possible for an adopted (pipeline) run: a direct-logging Run
        # always has a real id/base_url by the time __init__ returns (POST
        # already succeeded), or __init__ itself raised. See the module docs
        # ("Local dry mode") below for why this exists. NEITHER id nor
        # base_url resolving is a genuine local script (dry mode); exactly
        # ONE resolving is a real misconfiguration, not a local script, and
        # raises loudly instead of silently degrading into dry mode.
        if self._mode != "adopted":
            return False
        has_id, has_url = bool(self.id), bool(self._base_url)
        if has_id and has_url:
            return False
        if not has_id and not has_url:
            return True
        missing = "experiment_id" if not has_id else "api_url"
        raise RuntimeError(
            f"braven {missing} not set. The worker should have called braven.init() with it, "
            f"or set BRAVEN_{missing.upper()} in the environment."
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def settings(self, flush: bool | None = None, stream_interval: float | None = None) -> None:
        """Change run-level settings after construction. Mainly for pipeline
        scripts: `braven.init()` there takes no arguments (it adopts the
        Experiment the platform already created), so this is how a pipeline
        script opts out of streaming or widens the device-scoped debounce
        interval. A direct-logging script can just pass these to init()
        instead. Only non-None arguments are changed."""
        if flush is not None:
            self._auto_flush = flush
        if stream_interval is not None:
            self._stream_interval = stream_interval

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def config(self, key: str, value) -> None:
        """Log an input / configuration parameter. Targets the parent
        Experiment, or the device set by set_device() if one is active."""
        self._log(key, value, "config")

    log_config = config  # deprecated alias

    def summary(self, key: str, value) -> None:
        """Log an output metric. See config()."""
        self._log(key, value, "summary")

    log_summary = summary  # deprecated alias

    def series(self, key: str, values: list) -> None:
        """Log a numeric data series (stored as JSON). See config()."""
        self._log(key, json.dumps(values), "series")

    log_series = series  # deprecated alias

    def table(self, name: str, dataframe) -> None:
        """Log a named, columnar Table (targets the parent Experiment, or the
        device set by set_device() if one is active) — a pandas DataFrame,
        column names/types read directly off it. Distinct from series()
        (numeric x/y traces only): a Table can mix column types. Replaces
        any prior Table of the same `name` on this Experiment wholesale.
        Requires pandas: `pip install braven[dataframe]`."""
        key, value, category = _build_table_entry(name, dataframe)
        self._log(key, value, category, display=f"<DataFrame {dataframe.shape[0]}x{dataframe.shape[1]}>")

    log_table = table  # alias — no plain log_ predecessor to deprecate here,
    # kept for naming parity with config/summary/series's log_x aliases

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

    def set_device(self, key: str | None, type: str | None = None) -> None:
        """Point this run at a device's own child Experiment (ADR-0013):
        every config()/summary()/series()/plot_series()/upload() call after
        this targets that child instead of the parent, until
        set_device(None) returns to parent-level. Resolves-or-creates the
        device (and its child Experiment) on the backend on first sight;
        `type` names the device type to create it under — applied only when
        the device is new, ignored (never re-typing) for one that already
        exists.

            run.set_device("SENSOR-4471")
            run.summary("SNR", 14.2)   # this device's own child Experiment
            run.summary("SNR", 9.8)    # still SENSOR-4471
            run.set_device(None)       # back to the parent
        """
        self._device_target = str(key) if key is not None else None
        self._device_target_type = type if key is not None else None

    def upload(self, path_or_fig, name: str | None = None, interactive: bool = True) -> None:
        """Upload a file and attach it to this Experiment (the parent, or
        the device set by set_device() if one is active).

        `path_or_fig` may be a file path, or a live matplotlib Figure — in
        the latter case the PNG is saved the same way fig.savefig() would,
        and (when `interactive` is True, the default) the SDK also attempts
        to extract an interactive companion series from the figure's
        line/scatter data (best-effort, never blocks the upload).

        Pass `interactive=False` to skip that extraction — it costs an extra
        metadata flush to the backend. Worth setting for a figure
        re-uploaded rapidly in a loop (e.g. a live-updating plot), where
        only the final frame's interactive data is likely to matter.
        """
        fig = path_or_fig if _is_matplotlib_figure(path_or_fig) else None
        if fig is not None:
            upload_name = name or "figure.png"
        else:
            file_path = Path(path_or_fig)
            if not file_path.exists():
                raise FileNotFoundError(f"upload: file not found: {file_path}")
            upload_name = name or file_path.name

        device_key, device_type = self._device_target, self._device_target_type

        if self._local_dry_mode:
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

        params: dict = {"experiment_id": self.id}
        if self._pipeline_id:
            params["pipeline_id"] = self._pipeline_id
        if device_key:
            params["device_key"] = device_key
            if device_type:
                params["device_type"] = device_type
        with open(file_path, "rb") as fh:
            resp = requests.post(
                f"{self._base_url}/files",
                files=_upload_file_parts(upload_name, fh, file_path),
                params=params,
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

    def log_artifact(self, path: Union[str, Path], name: str | None = None) -> None:
        """Upload a file attached to this Experiment. An alias for upload()
        kept for scripts written before upload() accepted matplotlib
        Figures too — identical behavior for a plain file path."""
        self.upload(path, name)

    def flush(self) -> None:
        """Send any pending logged values now. No-op if nothing is pending.
        Called automatically by finish(). Returns the device created-report
        (new devices/types, type mismatches — see ADR-0013) for a pipeline
        run, or None."""
        return self._flush()

    def finish(self) -> None:
        """Flush and close out this run (idempotent)."""
        self._flush()

    def __repr__(self) -> str:
        if self._mode == "created":
            return f"Run(name={self.name!r}, id={self.id!r})"
        return f"Run(pipeline, id={self.id!r})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record(
        self, key: str, value, category: str, device_key: str | None, device_type: str | None, display=None
    ) -> None:
        """Buffer one entry into self._metadata (and print the local-dry-mode
        line if applicable). Pure bookkeeping — never sends anything.
        `display`, when given, replaces `value` in the printed line only —
        table() uses this so local-dry-mode printing shows a short shape
        summary instead of dumping the whole serialized Table JSON."""
        self._metadata[(device_key, key)] = {
            "key": key, "value": str(value), "category": category,
            "device_key": device_key, "device_type": device_type,
        }
        if self._local_dry_mode:
            shown = value if display is None else display
            suffix = f", device={device_key!r}" if device_key else ""
            print(f"[braven:local] would {category}({key!r}, {shown!r}{suffix})", flush=True)

    def _log(self, key: str, value, category: str, display=None) -> None:
        device_key, device_type = self._device_target, self._device_target_type
        self._record(key, value, category, device_key, device_type, display=display)
        if self._mode == "created":
            if self._auto_flush:
                self._flush()
            return
        # table stays on the buffered flush() path, same as series — a
        # per-call streaming write here would mean a full R2 write on every
        # table() call (ADR-0016 in braven-mvp); appendPipelineMetadataEntries
        # drops table-categorized entries server-side for the same reason, so
        # skipping the round trip here is purely avoiding wasted network.
        if self._local_dry_mode or not self._auto_flush or category in ("series", "table"):
            return
        if device_key is None:
            # Parent-level: stream this one entry immediately — best-effort
            # live preview. Hits a narrower endpoint (PATCH .../pipeline-
            # metadata/{id}, an incremental upsert scoped to just this key)
            # than flush()'s PUT, which re-derives and rewrites the WHOLE
            # accumulated flush every call — fine once at the end, not once
            # per log statement (O(n²) over a run otherwise).
            self._stream_one_entry(key, str(value), category)
        else:
            self._queue_device_stream_entry(key, str(value), category, device_key, device_type)

    def _log_many(self, entries: list[tuple[str, str, str]]) -> None:
        """Like _log(), but for several (key, value, category) triples
        flushed together — a plot_series() call or a matplotlib figure's
        extracted traces. Always category="series", which never streams
        per-call (see _log) regardless of the run's flush setting — buffered
        here for the next flush()/finish() (or the parent's next flush()
        call, for a created-mode run with flush=True)."""
        device_key, device_type = self._device_target, self._device_target_type
        for key, value, category in entries:
            self._record(key, value, category, device_key, device_type)
        if self._mode == "created" and self._auto_flush:
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

    def _flush(self) -> dict | None:
        if self._mode == "created":
            self._flush_created()
            return None
        return self._flush_adopted()

    def _flush_created(self) -> None:
        if not self._metadata:
            return
        resp = requests.put(
            f"{self._base_url}/experiments/{self.id}/metadata",
            json=list(self._metadata.values()),
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()

    def _flush_adopted(self) -> dict | None:
        """Atomically replace all of this run's metadata (called by the
        worker at run end, via braven.init(...).flush()). Returns the
        backend's device created-report ({createdDeviceKeys, createdTypeNames,
        typeMismatches}) so the worker can fold it into the run result, and
        prints a human line to stdout when the run created/flagged devices.
        Returns None when there's nothing to flush."""
        if self._local_dry_mode:
            print("[braven:local] flush() — nothing to flush (local dry mode)", flush=True)
            return None
        entries = list(self._metadata.values())
        if not entries:
            return None
        if not self._pipeline_id:
            raise RuntimeError("flush(): pipeline_id is required")
        resp = requests.put(
            f"{self._base_url}/experiments/{self.id}/pipeline-metadata/{self._pipeline_id}",
            json=entries,
            headers=self._headers,
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

    def _stream_one_entry(self, key: str, value: str, category: str) -> None:
        """Best-effort immediate write for one parent-level, non-series entry
        (streaming mode) — see _flush_adopted's docstring for the buffered,
        source-of-truth path this is a live-preview shortcut for. Never
        raises: a transient failure here just means this value shows up once
        the run finishes instead of live."""
        if not self._pipeline_id:
            return
        try:
            resp = requests.patch(
                f"{self._base_url}/experiments/{self.id}/pipeline-metadata/{self._pipeline_id}",
                json=[{"key": key, "value": value, "category": category}],
                headers=self._headers,
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"[braven] streaming flush skipped for {key!r}: {e}", flush=True)

    def _queue_device_stream_entry(
        self, key: str, value: str, category: str, device_key: str, device_type: str | None
    ) -> None:
        """Buffer one device-scoped entry for the debounced flush and send the
        batch once self._stream_interval has elapsed since the last one —
        bounds call volume, not latency: a wafer run calling config()/
        summary() per die would otherwise mean hundreds-thousands of
        streaming round trips regardless of how fast any one of them is."""
        self._device_stream_queue = [
            e for e in self._device_stream_queue if not (e["key"] == key and e.get("device_key") == device_key)
        ]
        self._device_stream_queue.append(
            {"key": key, "value": value, "category": category, "device_key": device_key, "device_type": device_type}
        )
        now = time.time()
        if now - self._device_stream_last_flush_at >= self._stream_interval:
            self._flush_device_stream_queue()
            self._device_stream_last_flush_at = now

    def _flush_device_stream_queue(self) -> None:
        """Best-effort immediate write of every currently-buffered device-scoped
        entry, in one PATCH — the batch may span several devices, each
        resolved to its own child Experiment server-side. Clears the buffer
        on success; a transient failure leaves entries queued for the next
        debounce tick (or the final flush(), the source of truth regardless)
        rather than dropping them. Never raises."""
        if not self._device_stream_queue or not self._pipeline_id:
            return
        batch = self._device_stream_queue
        try:
            resp = requests.patch(
                f"{self._base_url}/experiments/{self.id}/pipeline-metadata/{self._pipeline_id}",
                json=batch,
                headers=self._headers,
                timeout=15,
            )
            resp.raise_for_status()
            self._device_stream_queue = []
        except Exception as e:
            print(f"[braven] device-scoped streaming flush skipped ({len(batch)} entries): {e}", flush=True)


# ---------------------------------------------------------------------------
# braven.init() — the one entry point
# ---------------------------------------------------------------------------

# The current pipeline's adopted Run, if any — set once by the worker's own
# braven.init(experiment_id=..., ...) call (executor.py) before the script
# runs, and returned again by the script's own bare `run = braven.init()`
# (the "adopt-first" rule). Also what a bare init() falls back to resolving
# from BRAVEN_* environment variables when nothing has configured this yet —
# see "Local dry mode" below.
_adopted_run: Run | None = None


def init(
    name: str | None = None,
    notes: str | None = None,
    project: str | None = None,
    company: str | None = None,
    flush: bool = True,
    stream_interval: float | None = None,
    *,
    experiment_id: str | None = None,
    api_url: str | None = None,
    watcher_secret: str | None = None,
    user_id: str | None = None,
    pipeline_id: str | None = None,
    stream: bool | None = None,
) -> Run:
    """Create or adopt the Run this script logs to.

    Direct logging (your own script) — pass at least one of name/notes/
    project/company, always creates a new Experiment:
        run = braven.init(name="My Experiment")

    Pipeline scripts (run by the Braven worker) — call bare, no arguments:
        run = braven.init()
    This adopts the Experiment the worker already created (no credentials or
    project needed) — including when you run the same script standalone,
    with no worker around, for local iteration: see "Local dry mode" below.

    `flush` (default True) controls whether config()/summary()/series()/
    plot_series() calls send immediately or only join the buffered flush()/
    finish() — pass False to batch several calls into one write. Pass
    `stream_interval` to override how often a pipeline run's device-scoped
    (set_device()-active) entries are batch-flushed (default
    DEVICE_STREAM_INTERVAL_S seconds). A pipeline script, which calls
    init() with no arguments, sets these after construction instead —
    see Run.settings().

    The keyword-only experiment_id/api_url/watcher_secret/user_id/
    pipeline_id/stream arguments are how the worker itself configures the
    pipeline context (executor.py) — a script never needs to pass these.
    `stream` is a deprecated alias for `flush`.
    """
    global _adopted_run
    if stream is not None:
        flush = stream

    if name is not None or notes is not None or project is not None or company is not None:
        return Run._create(name, notes, project, company, flush, stream_interval)

    if experiment_id is not None:
        # The worker's own once-per-run call (executor.py) — always builds a
        # fresh adopted Run for this run. api_url/watcher_secret still fall
        # back to environment variables even here, same as the bare-call
        # branch below, in case a caller passes experiment_id explicitly but
        # relies on the environment for the rest.
        _adopted_run = Run._adopted(
            experiment_id,
            api_url or os.environ.get("BRAVEN_API_URL"),
            watcher_secret or os.environ.get("BRAVEN_WATCHER_SECRET") or os.environ.get("WATCHER_SECRET"),
            pipeline_id,
            flush,
            stream_interval,
        )
        return _adopted_run

    if _adopted_run is not None:
        # A pipeline script's own bare `run = braven.init()` — adopt what the
        # worker already configured above.
        return _adopted_run

    # Nothing configured yet in this process — either a user manually
    # replicating a pipeline run locally (BRAVEN_* environment variables
    # set), or a pipeline script being iterated on standalone with no worker
    # around at all, in which case nothing resolves and every logging call
    # below prints what it would have done instead of raising (see "Local
    # dry mode"). A *partially* configured environment (one variable set,
    # the other not) is a real misconfiguration, not a local script — the
    # first genuinely network-facing call still raises via the missing field.
    _adopted_run = Run._adopted(
        os.environ.get("BRAVEN_EXPERIMENT_ID"),
        api_url or os.environ.get("BRAVEN_API_URL"),
        watcher_secret or os.environ.get("BRAVEN_WATCHER_SECRET") or os.environ.get("WATCHER_SECRET"),
        pipeline_id,
        flush,
        stream_interval,
    )
    return _adopted_run


def _resolve_project_id(cfg: dict, company: str | None, project: str | None) -> str:
    """Look up the project ID by calling the Braven API with the stored credentials.

    Raises a RuntimeError with a clear message if no match is found.
    """
    if project is None and company is None:
        raise RuntimeError(
            "Specify company and project in braven.init():\n"
            '    braven.init(name="My Experiment", company="My Company", project="My Project")\n'
            "(project alone is enough if it's unique across your companies)"
        )
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    resp = requests.get(f"{cfg['api_url'].rstrip('/')}/projects", headers=headers, timeout=30)
    resp.raise_for_status()
    projects = resp.json()

    candidates = projects
    if company is not None:
        candidates = [p for p in candidates if (p.get("company_name") or "").lower() == company.lower()]
    if project is not None:
        candidates = [p for p in candidates if (p.get("name") or "").lower() == project.lower()]

    if len(candidates) == 1:
        return candidates[0]["id"]
    if len(candidates) == 0:
        available = ", ".join(f"{p.get('company_name')}/{p.get('name')}" for p in projects) or "(none)"
        raise RuntimeError(f"No project matches company={company!r}, project={project!r}. Available: {available}")
    available = ", ".join(f"{p.get('company_name')}/{p.get('name')}" for p in candidates)
    raise RuntimeError(f"Multiple projects match company={company!r}, project={project!r}: {available}. Be more specific.")


# ---------------------------------------------------------------------------
# Local dry mode — pipeline scripts are normally written and iterated on
# locally (`python script.py`) before being copy-pasted into the webapp
# Pipeline editor. `run = braven.init()` (bare, no worker around) resolves
# nothing (no BRAVEN_* environment variables set), so every Run method
# prints what it would have done instead of raising — see Run._local_dry_mode
# and each method's local-dry-mode branch above.
#
# This can never fire inside a real worker run: worker/executor.py always
# calls braven.init(experiment_id=..., api_url=..., ...) with explicit
# keyword arguments before user code executes, and explicit arguments
# already take precedence over every other source above. A *partially*
# configured context (one field resolves, the other doesn't) is left alone —
# that's a real misconfiguration, not a local script, and still raises.
#
# See braven-mvp's docs/adr/0008-local-pipeline-dry-mode.md and
# .scratch/local-pipeline-dry-mode/spec.md for the full design.
# ---------------------------------------------------------------------------


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


# Convenience dict populated by the executor: filename → local temp path
files: dict[str, str] = {}

# Extracted parameters populated by the executor before process() runs.
# Keys are canonical names (from field mappings); raw key used as fallback.
params: dict = {}

# Column rename map populated by the mapping layer before process() runs.
# Maps raw CSV column names to canonical names: {raw: canonical}
column_maps: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Removed names — every logging call now goes through an explicit `run`
# (see the module docstring). Raises with migration guidance instead of an
# opaque AttributeError, since older stored pipeline scripts may still
# reference these bare.
# ---------------------------------------------------------------------------

_REMOVED_AMBIENT_NAMES = {
    "config", "summary", "series", "log_config", "log_summary", "log_series",
    "table", "log_table",
    "plot_series", "upload", "log_artifact", "flush", "finish",
    "get_device", "device", "set_device", "log", "log_plot", "log_metadata",
}


def __getattr__(name: str):
    if name in _REMOVED_AMBIENT_NAMES:
        raise RuntimeError(
            f"braven.{name}() has been removed — every logging call now goes through an "
            f"explicit run: `run = braven.init(); run.{name}(...)`. In a pipeline script, "
            f"`run = braven.init()` (no arguments) adopts the Experiment the platform "
            f"already created."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
