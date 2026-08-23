"""Tests for braven.py's pure, network-free helpers.

No mocking of internals — every test calls the real function directly and
asserts on real, undoctored output (list-thumbnail preview bytes really
decoded with PIL, a real matplotlib Figure really walked). This matches the
one pre-existing precedent for this module, worker/test_braven_preview.py in
the private braven-mvp repo, which these preview-generation tests are ported
from.

Out of scope (per the spec): anything that hits the network — Run, Device,
Braven (query client), Analysis, collect(). login()/_load_config() are pure
filesystem + stdin, so their branching (interactive prompt vs. non-interactive
error, per the wandb-style first-use flow) is covered below with a monkeypatched
_CONFIG_PATH — never your real ~/.braven/config.json.
"""

import io
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed to run these tests
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import braven


def _tmp_image(name: str, size=(2082, 1183), mode="RGB") -> Path:
    path = Path(tempfile.mkdtemp()) / name
    Image.new(mode, size, "white").save(path)
    return path


# ---------------------------------------------------------------------------
# _make_image_preview / _upload_file_parts (ported from
# worker/test_braven_preview.py)
# ---------------------------------------------------------------------------


def test_preview_is_downscaled_webp():
    # Production shape: matplotlib PNG ~2082x1183, several MB, shown at ~206px.
    path = _tmp_image("figure.png")
    data = braven._make_image_preview(path)
    assert data is not None
    im = Image.open(io.BytesIO(data))
    assert im.format == "WEBP"
    assert max(im.size) <= 420
    # aspect preserved (2082:1183 ≈ 1.76)
    assert abs(im.size[0] / im.size[1] - 2082 / 1183) < 0.02
    assert len(data) <= 100 * 1024


def test_small_image_is_not_upscaled():
    path = _tmp_image("tiny.png", size=(64, 48))
    data = braven._make_image_preview(path)
    assert data is not None
    assert Image.open(io.BytesIO(data)).size == (64, 48)


def test_palette_and_rgba_modes_convert():
    for mode in ("P", "RGBA", "L"):
        path = _tmp_image(f"img_{mode}.png", size=(500, 500), mode=mode)
        data = braven._make_image_preview(path)
        assert data is not None, mode
        assert Image.open(io.BytesIO(data)).format == "WEBP", mode


def test_non_image_returns_none():
    path = Path(tempfile.mkdtemp()) / "data.csv"
    path.write_text("a,b\n1,2\n")
    assert braven._make_image_preview(path) is None


def test_missing_file_returns_none_never_raises():
    assert braven._make_image_preview(Path("does/not/exist.png")) is None


def test_upload_parts_attach_preview_for_images_only():
    img = _tmp_image("figure.png")
    with open(img, "rb") as fh:
        parts = braven._upload_file_parts("figure.png", fh, img)
        assert parts["file"] == ("figure.png", fh)
        assert "preview" in parts
        name, data, ctype = parts["preview"]
        assert name == "figure.png.webp"
        assert ctype == "image/webp"
        assert Image.open(io.BytesIO(data)).format == "WEBP"

    csv = Path(tempfile.mkdtemp()) / "data.csv"
    csv.write_text("a,b\n1,2\n")
    with open(csv, "rb") as fh:
        parts = braven._upload_file_parts("data.csv", fh, csv)
        assert parts == {"file": ("data.csv", fh)}


def test_upload_parts_pass_through_content_type():
    # Analysis.upload() resolves a content_type and passes it through — the
    # file part becomes a 3-tuple instead of 2, preview attachment unaffected.
    img = _tmp_image("figure.png")
    with open(img, "rb") as fh:
        parts = braven._upload_file_parts("figure.png", fh, img, content_type="image/png")
        assert parts["file"] == ("figure.png", fh, "image/png")
        assert "preview" in parts


# ---------------------------------------------------------------------------
# Matplotlib figure introspection
# ---------------------------------------------------------------------------


def test_to_float_list_coerces_numeric_iterables():
    assert braven._to_float_list([1, 2, 3]) == [1.0, 2.0, 3.0]
    assert braven._to_float_list(np.array([1.5, 2.5])) == [1.5, 2.5]


def test_to_float_list_returns_none_for_non_numeric():
    assert braven._to_float_list(["a", "b"]) is None


def test_safe_filename_sdk_replaces_unsafe_characters():
    assert braven._safe_filename_sdk("plot #1 (final).png") == "plot _1 _final_.png"


def test_safe_filename_sdk_preserves_path_segments():
    assert braven._safe_filename_sdk("sub/dir/plot?.png") == "sub/dir/plot_.png"


def test_safe_filename_sdk_empty_name_falls_back():
    assert braven._safe_filename_sdk("") == "file"


def test_downsample_trace_below_cap_is_unchanged():
    xs = list(range(10))
    ys = [float(x) for x in xs]
    xd, yd = braven._downsample_trace(xs, ys)
    assert xd == xs
    assert yd == ys


def test_downsample_trace_above_cap_keeps_last_point():
    n = braven._MAX_PLOT_POINTS_PER_TRACE * 3
    xs = list(range(n))
    ys = [float(x) for x in xs]
    xd, yd = braven._downsample_trace(xs, ys)
    assert len(xd) <= braven._MAX_PLOT_POINTS_PER_TRACE + 1
    assert xd[-1] == xs[-1]
    assert yd[-1] == ys[-1]


def test_walk_matplotlib_figure_extracts_line_trace():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [0, 1, 4, 9], label="parabola")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    groups = braven._walk_matplotlib_figure(fig)
    plt.close(fig)

    assert len(groups) == 1
    g = groups[0]
    assert g["x_label"] == "x"
    assert g["y_label"] == "y"
    assert len(g["traces"]) == 1
    tr = g["traces"][0]
    assert tr["label"] == "parabola"
    assert tr["mode"] == "line"
    assert tr["x"] == [0.0, 1.0, 2.0, 3.0]
    assert tr["y"] == [0.0, 1.0, 4.0, 9.0]


def test_walk_matplotlib_figure_marker_only_line_is_scatter():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 2, 3], linestyle="None", marker="o")

    groups = braven._walk_matplotlib_figure(fig)
    plt.close(fig)

    assert groups[0]["traces"][0]["mode"] == "scatter"


def test_walk_matplotlib_figure_scatter_collection():
    fig, ax = plt.subplots()
    ax.scatter([0, 1, 2], [3, 1, 4], label="pts")

    groups = braven._walk_matplotlib_figure(fig)
    plt.close(fig)

    assert len(groups) == 1
    tr = groups[0]["traces"][0]
    assert tr["mode"] == "scatter"
    assert tr["x"] == [0.0, 1.0, 2.0]
    assert tr["y"] == [3.0, 1.0, 4.0]


def test_walk_matplotlib_figure_no_usable_artists_returns_empty():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((4, 4)))

    groups = braven._walk_matplotlib_figure(fig)
    plt.close(fig)

    assert groups == []


def test_walk_matplotlib_figure_downsamples_large_trace():
    n = braven._MAX_PLOT_POINTS_PER_TRACE * 5
    fig, ax = plt.subplots()
    ax.plot(range(n), range(n))

    groups = braven._walk_matplotlib_figure(fig)
    plt.close(fig)

    tr = groups[0]["traces"][0]
    assert len(tr["x"]) <= braven._MAX_PLOT_POINTS_PER_TRACE + 1


# ---------------------------------------------------------------------------
# Plot-series key building
# ---------------------------------------------------------------------------


def test_next_plot_trace_idx_increments_per_name():
    braven._plot_series_counters.clear()
    assert braven._next_plot_trace_idx("snr") == 0
    assert braven._next_plot_trace_idx("snr") == 1
    assert braven._next_plot_trace_idx("other") == 0


def test_build_plot_series_entries_shape():
    braven._plot_series_counters.clear()
    entries = braven._build_plot_series_entries(
        "snr", y=[1, 2, 3], x=[0, 1, 2], x_label="time", y_label="dB", mode="line"
    )
    keys = [k for k, _, _ in entries]
    assert keys == ["plot::snr::0::y", "plot::snr::0::x", "plot::snr::0::meta"]
    assert all(cat == "series" for _, _, cat in entries)

    y_value = json.loads(dict((k, v) for k, v, _ in entries)["plot::snr::0::y"])
    assert y_value == [1.0, 2.0, 3.0]

    meta = json.loads(dict((k, v) for k, v, _ in entries)["plot::snr::0::meta"])
    assert meta == {"mode": "line", "trace_label": "dB", "x_label": "time", "y_label": "dB"}


def test_build_plot_series_entries_without_x():
    braven._plot_series_counters.clear()
    entries = braven._build_plot_series_entries(
        "snr", y=[1, 2], x=None, x_label=None, y_label=None, mode="scatter"
    )
    keys = [k for k, _, _ in entries]
    assert keys == ["plot::snr::0::y", "plot::snr::0::meta"]


def test_build_plot_series_entries_second_call_gets_distinct_index():
    braven._plot_series_counters.clear()
    first = braven._build_plot_series_entries("snr", [1], None, None, "a", "line")
    second = braven._build_plot_series_entries("snr", [2], None, None, "b", "line")
    assert first[0][0] == "plot::snr::0::y"
    assert second[0][0] == "plot::snr::1::y"


def test_build_plot_series_entries_coerces_numpy_values():
    braven._plot_series_counters.clear()
    entries = braven._build_plot_series_entries(
        "snr", y=np.array([1.5, 2.5]), x=None, x_label=None, y_label=None, mode="line"
    )
    y_value = json.loads(entries[0][1])
    assert y_value == [1.5, 2.5]


# ---------------------------------------------------------------------------
# _load_config / _prompt_login — wandb-style first-use login prompt.
# Every test redirects braven._CONFIG_PATH into pytest's tmp_path, so none of
# these ever touch the real ~/.braven/config.json.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_config_path(tmp_path, monkeypatch):
    path = tmp_path / ".braven" / "config.json"
    monkeypatch.setattr(braven, "_CONFIG_PATH", path)
    return path


def test_load_config_reads_an_existing_file(isolated_config_path):
    isolated_config_path.parent.mkdir(parents=True)
    isolated_config_path.write_text(json.dumps({"api_url": "https://example.com", "api_key": "braven_abc"}))

    assert braven._load_config() == {"api_url": "https://example.com", "api_key": "braven_abc"}


def test_load_config_raises_a_clear_error_on_corrupted_json(isolated_config_path):
    isolated_config_path.parent.mkdir(parents=True)
    isolated_config_path.write_text("not json")

    with pytest.raises(RuntimeError, match="Corrupted credentials file"):
        braven._load_config()


def test_load_config_raises_when_missing_and_non_interactive(isolated_config_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(RuntimeError, match="Not logged in"):
        braven._load_config()
    assert not isolated_config_path.exists()


def test_load_config_prompts_and_persists_when_missing_and_interactive(isolated_config_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(braven.getpass, "getpass", lambda _prompt: "braven_pasted_key")

    cfg = braven._load_config()

    assert cfg == {"api_url": braven._DEFAULT_API_URL, "api_key": "braven_pasted_key"}
    assert json.loads(isolated_config_path.read_text()) == cfg
    # A second call reads the now-saved file back rather than prompting again
    # (an isatty()/getpass patched to raise would fail this call if it did).
    monkeypatch.setattr(sys.stdin, "isatty", lambda: (_ for _ in ()).throw(AssertionError("should not re-prompt")))
    assert braven._load_config() == cfg


def test_load_config_raises_when_interactive_prompt_left_blank(isolated_config_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(braven.getpass, "getpass", lambda _prompt: "  ")

    with pytest.raises(RuntimeError, match="No API key entered"):
        braven._load_config()
    assert not isolated_config_path.exists()


# ---------------------------------------------------------------------------
# Local dry mode (docs/adr/0008-local-pipeline-dry-mode.md in braven-mvp) —
# a bare `braven.init()` (adopted mode) with no configured experiment_id/
# api_url prints instead of raising or touching the network. Every test
# resets braven.py's module-level adopted-run singleton (process-wide state,
# not per-Run) and asserts no HTTP call was ever attempted, since "doesn't
# raise" alone wouldn't catch a version that silently degrades into a real
# network call instead of a true no-op.
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_pipeline_context(monkeypatch):
    """Isolates each test from real BRAVEN_* env vars and resets braven.py's
    module-level adopted-run singleton before and after, so tests can't leak
    state into each other via shared module state."""
    for var in ("BRAVEN_EXPERIMENT_ID", "BRAVEN_API_URL", "BRAVEN_WATCHER_SECRET", "WATCHER_SECRET", "BRAVEN_USER_ID"):
        monkeypatch.delenv(var, raising=False)

    def _clear():
        monkeypatch.setattr(braven, "_adopted_run", None)

    _clear()
    yield
    _clear()


def _no_network(monkeypatch):
    """Fails the test immediately if braven.py makes any real HTTP call —
    local dry mode must never reach the network."""
    def _boom(*args, **kwargs):
        raise AssertionError(f"unexpected network call: args={args} kwargs={kwargs}")
    monkeypatch.setattr(braven.requests, "post", _boom)
    monkeypatch.setattr(braven.requests, "put", _boom)
    monkeypatch.setattr(braven.requests, "get", _boom)
    monkeypatch.setattr(braven.requests, "patch", _boom)


def test_config_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)
    run = braven.init()

    run.config("lr", "0.001")  # must not raise

    assert "[braven:local] would config('lr', '0.001')" in capsys.readouterr().out


def test_summary_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)
    run = braven.init()

    run.summary("acc", "0.94")

    assert "would summary('acc', '0.94')" in capsys.readouterr().out


def test_series_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)
    run = braven.init()

    run.series("temperature", [1.0, 2.0, 3.0])

    assert "would series" in capsys.readouterr().out


def test_set_device_logging_with_no_context_prints_and_tags_device(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)
    run = braven.init()

    run.set_device("SENSOR-1")
    run.summary("SNR", 14.2)

    assert "would summary('SNR', 14.2, device='SENSOR-1')" in capsys.readouterr().out


def test_removed_ambient_names_raise_with_migration_guidance(reset_pipeline_context):
    for name in ("device", "get_device", "log_config", "log_summary", "set_device", "upload"):
        with pytest.raises(RuntimeError, match="run\\."):
            getattr(braven, name)


def test_set_device_makes_subsequent_calls_ambient(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)
    run = braven.init()

    run.set_device("SENSOR-1")
    run.config("gain", "10")
    run.summary("SNR", 14.2)
    run.series("temperature", [1.0, 2.0])

    out = capsys.readouterr().out
    assert "would config('gain', '10', device='SENSOR-1')" in out
    assert "would summary('SNR', 14.2, device='SENSOR-1')" in out
    assert "device='SENSOR-1'" in out  # series line


def test_set_device_none_returns_to_parent_level(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)
    run = braven.init()

    run.set_device("SENSOR-1")
    run.set_device(None)
    run.summary("max_device_mismatch", 3.1)

    out = capsys.readouterr().out
    assert "would summary('max_device_mismatch', 3.1)" in out
    assert "device=" not in out


def test_a_new_pipeline_run_starts_with_no_device_target(reset_pipeline_context, monkeypatch):
    # A genuinely new run (experiment_id passed explicitly, as the worker's
    # own once-per-run call does) is a fresh Run — set_device() state from a
    # previous run can never leak into it.
    run1 = braven.init(experiment_id="exp_1", api_url="https://api.example")
    run1.set_device("SENSOR-1")

    run2 = braven.init(experiment_id="exp_2", api_url="https://api.example")

    assert run2 is not run1
    assert run2._device_target is None


def test_device_upload_with_no_context_prints_and_tags_device(reset_pipeline_context, monkeypatch, capsys, tmp_path):
    _no_network(monkeypatch)
    path = tmp_path / "figure.png"
    path.write_bytes(b"fake")
    run = braven.init()
    run.set_device("SENSOR-1")

    run.upload(path)

    assert "would upload('figure.png', device='SENSOR-1')" in capsys.readouterr().out


def test_created_run_upload_includes_device_key_when_set_device_active(monkeypatch, tmp_path):
    # A direct-logging (created-mode) run's upload() supports set_device()
    # too (ADR-0013 no longer restricts device-scoped uploads to pipeline
    # scripts, now that config()/summary()/series() route onto the child the
    # same way) — asserts the params sent, not a real network call.
    run = braven.Run(mode="created", api_url="https://api.example", headers={}, experiment_id="exp_1")
    run.set_device("SENSOR-1", "Photodiode")
    path = tmp_path / "figure.png"
    path.write_bytes(b"fake")
    captured = {}

    def _fake_post(url, files=None, params=None, headers=None, timeout=None):
        captured["params"] = params
        return type("Resp", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr(braven.requests, "post", _fake_post)

    run.upload(path)

    assert captured["params"] == {"experiment_id": "exp_1", "device_key": "SENSOR-1", "device_type": "Photodiode"}


def test_flush_with_no_context_is_a_noop_print(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)
    run = braven.init()

    result = run.flush()

    assert result is None
    assert "flush() — nothing to flush (local dry mode)" in capsys.readouterr().out


def test_log_artifact_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys, tmp_path):
    _no_network(monkeypatch)
    path = tmp_path / "model.pkl"
    path.write_bytes(b"fake")
    run = braven.init()

    run.log_artifact(path)

    assert "would upload('model.pkl')" in capsys.readouterr().out


def test_log_artifact_with_no_context_still_raises_on_missing_file(reset_pipeline_context, monkeypatch, tmp_path):
    _no_network(monkeypatch)
    run = braven.init()

    with pytest.raises(FileNotFoundError):
        run.log_artifact(tmp_path / "missing.pkl")


def test_upload_path_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys, tmp_path):
    _no_network(monkeypatch)
    path = tmp_path / "plot.png"
    path.write_bytes(b"fake")
    run = braven.init()

    run.upload(path)

    assert "would upload('plot.png')" in capsys.readouterr().out


def test_upload_figure_with_no_context_saves_png_locally(reset_pipeline_context, monkeypatch, capsys, tmp_path):
    _no_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run = braven.init()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    run.upload(fig, "value_over_index.png")
    plt.close(fig)

    saved = tmp_path / braven._LOCAL_OUTPUT_DIR / "value_over_index.png"
    assert saved.exists()
    with Image.open(saved) as im:
        im.verify()  # raises if the bytes aren't a real, decodable image
    assert "saved figure to" in capsys.readouterr().out


def test_upload_figure_with_no_context_does_not_overwrite_on_second_call(reset_pipeline_context, monkeypatch, tmp_path):
    _no_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run = braven.init()
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2], [1, 2])
    fig2, ax2 = plt.subplots()
    ax2.plot([3, 4], [3, 4])

    run.upload(fig1, "figure.png")
    run.upload(fig2, "figure.png")
    plt.close(fig1)
    plt.close(fig2)

    out_dir = tmp_path / braven._LOCAL_OUTPUT_DIR
    assert (out_dir / "figure.png").exists()
    assert (out_dir / "figure_1.png").exists()  # second call didn't overwrite the first


def test_partial_context_still_raises_not_local_mode(reset_pipeline_context, monkeypatch):
    # Only experiment_id resolves, api_url doesn't — a real misconfiguration,
    # not a local script; must still raise loudly, not silently fall back.
    run = braven.init(experiment_id="exp_123")

    with pytest.raises(RuntimeError, match="api_url not set"):
        run.config("k", "v")


def test_configured_context_is_unaffected_by_local_mode_change(reset_pipeline_context, monkeypatch):
    # Regression guard: a fully-configured context still queues for a real
    # flush rather than local-mode printing.
    run = braven.init(experiment_id="exp_123", api_url="https://api.example.com")

    run.config("lr", "0.001")

    assert run._metadata == {
        (None, "lr"): {"key": "lr", "value": "0.001", "category": "config", "device_key": None, "device_type": None}
    }


# ---------------------------------------------------------------------------
# Streaming mode (2026-08-22) — config()/summary() write immediately, by
# default, alongside the buffered set flush() still needs at run end.
# requests.patch is monkeypatched (not hit for real) — same "no real
# network" boundary the rest of this file keeps, just with a recording fake
# instead of _no_network's failing one, since asserting the call did/didn't
# happen IS the behavior under test here.
# ---------------------------------------------------------------------------

def _pipeline_run(monkeypatch, *, pipeline_id="pipe_1"):
    return braven.init(experiment_id="exp_123", api_url="https://api.example.com", pipeline_id=pipeline_id)


def test_streaming_default_on_patches_experiment_level_entries_immediately(reset_pipeline_context, monkeypatch):
    run = _pipeline_run(monkeypatch)
    calls = []

    def _fake_patch(url, json=None, headers=None, timeout=None):
        calls.append((url, json))
        return type("Resp", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr(braven.requests, "patch", _fake_patch)

    run.config("lr", "0.001")

    assert calls == [
        ("https://api.example.com/experiments/exp_123/pipeline-metadata/pipe_1",
         [{"key": "lr", "value": "0.001", "category": "config"}])
    ]
    # Still queued too — flush() at run end stays the source of truth.
    assert run._metadata == {
        (None, "lr"): {"key": "lr", "value": "0.001", "category": "config", "device_key": None, "device_type": None}
    }


def test_streaming_disabled_never_hits_the_network(reset_pipeline_context, monkeypatch):
    run = _pipeline_run(monkeypatch)
    run.settings(flush=False)
    _no_network(monkeypatch)

    run.config("lr", "0.001")  # must not raise

    assert run._metadata == {
        (None, "lr"): {"key": "lr", "value": "0.001", "category": "config", "device_key": None, "device_type": None}
    }


def test_streaming_skips_device_tagged_entries(reset_pipeline_context, monkeypatch):
    run = _pipeline_run(monkeypatch)
    _no_network(monkeypatch)  # a device-scoped call must not hit the streaming endpoint

    run.set_device("SENSOR-1")
    run.summary("SNR", 14.2)

    assert run._metadata == {
        ("SENSOR-1", "SNR"): {"key": "SNR", "value": "14.2", "category": "summary", "device_key": "SENSOR-1", "device_type": None}
    }


def test_streaming_skips_series_entries(reset_pipeline_context, monkeypatch):
    run = _pipeline_run(monkeypatch)
    _no_network(monkeypatch)  # series stays on the buffered path — no per-call R2 write

    run.series("trace", [1, 2, 3])

    assert len(run._metadata) == 1
    assert next(iter(run._metadata.values()))["category"] == "series"


def test_streaming_failure_is_swallowed_not_raised(reset_pipeline_context, monkeypatch, capsys):
    run = _pipeline_run(monkeypatch)

    def _boom(*args, **kwargs):
        raise ConnectionError("network is down")

    monkeypatch.setattr(braven.requests, "patch", _boom)

    run.config("lr", "0.001")  # must not raise despite the streaming call failing

    assert "streaming flush skipped" in capsys.readouterr().out
    assert run._metadata == {
        (None, "lr"): {"key": "lr", "value": "0.001", "category": "config", "device_key": None, "device_type": None}
    }


def test_streaming_without_pipeline_id_is_a_noop_not_an_error(reset_pipeline_context, monkeypatch):
    # pipeline_id unset (e.g. a caller that never passed it) — streaming has
    # nothing to PATCH against, so it must skip silently, not hit the network.
    run = braven.init(experiment_id="exp_123", api_url="https://api.example.com")
    _no_network(monkeypatch)

    run.config("lr", "0.001")

    assert run._metadata == {
        (None, "lr"): {"key": "lr", "value": "0.001", "category": "config", "device_key": None, "device_type": None}
    }


def test_settings_overrides_stream_interval(reset_pipeline_context, monkeypatch):
    run = _pipeline_run(monkeypatch)
    assert run._stream_interval == braven.DEVICE_STREAM_INTERVAL_S

    run.settings(stream_interval=9.0)

    assert run._stream_interval == 9.0
