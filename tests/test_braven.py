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
# pipeline-vocabulary calls with no configured context print instead of
# raising or touching the network. Every test resets the module-level
# pipeline-context globals (they're process-wide state, not per-Run) and
# asserts no HTTP call was ever attempted, since "doesn't raise" alone
# wouldn't catch a version that silently degrades into a real network call
# instead of a true no-op.
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_pipeline_context(monkeypatch):
    """Isolates each test from real BRAVEN_* env vars and resets braven.py's
    module-level pipeline/direct-run context before and after, so tests
    can't leak state into each other via shared module globals."""
    for var in ("BRAVEN_EXPERIMENT_ID", "BRAVEN_API_URL", "BRAVEN_WATCHER_SECRET", "WATCHER_SECRET", "BRAVEN_USER_ID"):
        monkeypatch.delenv(var, raising=False)

    def _clear():
        monkeypatch.setattr(braven, "braven_experiment_id", None)
        monkeypatch.setattr(braven, "braven_api_url", None)
        monkeypatch.setattr(braven, "braven_watcher_secret", None)
        monkeypatch.setattr(braven, "braven_user_id", None)
        monkeypatch.setattr(braven, "braven_pipeline_id", None)
        monkeypatch.setattr(braven, "_metadata_queue", [])
        monkeypatch.setattr(braven, "params", {})
        monkeypatch.setattr(braven, "column_maps", {})
        monkeypatch.setattr(braven, "_pipeline_run", None)
        monkeypatch.setattr(braven, "_current_run", None)

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


def test_log_config_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)

    braven.log_config("lr", "0.001")  # must not raise

    out = capsys.readouterr().out
    assert "[braven:local] would log_config('lr', '0.001')" in out
    assert braven._metadata_queue == []  # nothing queued in local mode


def test_log_summary_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)

    braven.log_summary("acc", "0.94")

    assert "would log_summary('acc', '0.94')" in capsys.readouterr().out


def test_log_series_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)

    braven.log_series("temperature", [1.0, 2.0, 3.0])

    assert "would log_series" in capsys.readouterr().out


def test_device_logging_with_no_context_prints_and_tags_device(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)

    braven.device("SENSOR-1").log_summary("SNR", 14.2)

    assert "would log_summary('SNR', '14.2', device='SENSOR-1')" in capsys.readouterr().out


def test_flush_metadata_with_no_context_is_a_noop_print(reset_pipeline_context, monkeypatch, capsys):
    _no_network(monkeypatch)

    result = braven.flush_metadata(pipeline_id="pipe_1")

    assert result is None
    assert "flush_metadata() — nothing to flush (local dry mode)" in capsys.readouterr().out


def test_log_artifact_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys, tmp_path):
    _no_network(monkeypatch)
    path = tmp_path / "model.pkl"
    path.write_bytes(b"fake")

    braven.log_artifact(path)

    assert "would log_artifact('model.pkl')" in capsys.readouterr().out


def test_log_artifact_with_no_context_still_raises_on_missing_file(reset_pipeline_context, monkeypatch, tmp_path):
    _no_network(monkeypatch)

    with pytest.raises(FileNotFoundError):
        braven.log_artifact(tmp_path / "missing.pkl")


def test_upload_path_with_no_context_prints_instead_of_raising(reset_pipeline_context, monkeypatch, capsys, tmp_path):
    _no_network(monkeypatch)
    path = tmp_path / "plot.png"
    path.write_bytes(b"fake")

    braven.upload(path)

    assert "would upload('plot.png')" in capsys.readouterr().out


def test_upload_figure_with_no_context_saves_png_locally(reset_pipeline_context, monkeypatch, capsys, tmp_path):
    _no_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    braven.upload(fig, "value_over_index.png")
    plt.close(fig)

    saved = tmp_path / braven._LOCAL_OUTPUT_DIR / "value_over_index.png"
    assert saved.exists()
    with Image.open(saved) as im:
        im.verify()  # raises if the bytes aren't a real, decodable image
    assert "saved figure to" in capsys.readouterr().out


def test_upload_figure_with_no_context_does_not_overwrite_on_second_call(reset_pipeline_context, monkeypatch, tmp_path):
    _no_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2], [1, 2])
    fig2, ax2 = plt.subplots()
    ax2.plot([3, 4], [3, 4])

    braven.upload(fig1, "figure.png")
    braven.upload(fig2, "figure.png")
    plt.close(fig1)
    plt.close(fig2)

    out_dir = tmp_path / braven._LOCAL_OUTPUT_DIR
    assert (out_dir / "figure.png").exists()
    assert (out_dir / "figure_1.png").exists()  # second call didn't overwrite the first


def test_partial_context_still_raises_not_local_mode(reset_pipeline_context, monkeypatch):
    # Only experiment_id resolves, api_url doesn't — a real misconfiguration,
    # not a local script; must still raise loudly, not silently fall back.
    monkeypatch.setattr(braven, "braven_experiment_id", "exp_123")

    with pytest.raises(RuntimeError, match="api_url not set"):
        braven.log_config("k", "v")


def test_configured_context_is_unaffected_by_local_mode_change(reset_pipeline_context, monkeypatch):
    # Regression guard: a fully-configured context still queues for a real
    # flush rather than local-mode printing.
    monkeypatch.setattr(braven, "braven_experiment_id", "exp_123")
    monkeypatch.setattr(braven, "braven_api_url", "https://api.example.com")

    braven.log_config("lr", "0.001")

    assert braven._metadata_queue == [
        {"key": "lr", "value": "0.001", "category": "config", "device_key": None, "device_type": None}
    ]
