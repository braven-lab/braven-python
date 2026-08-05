# braven

Python SDK for [Braven](https://bravenlab.com), the experiment tracker for
hardware and photonics R&D teams.

```bash
pip install braven
```

## Install and login

```bash
pip install braven
python -m braven login
```

`login` asks for your backend URL and an SDK API key (`braven_...`, from
your Braven account's Settings → Watcher Keys) and saves them to
`~/.braven/config.json`. Every script on the machine picks the credentials
up automatically — no need to pass them around.

## Direct logging (wandb-style)

The common case: a script you run yourself, logging config, metrics, and
plots straight to Braven as it runs.

```python
import braven

run = braven.init(name="My Experiment", company="My Company", project="My Project")

braven.config("radius_um", 55.0)
braven.config("gap_nm", 200.0)

braven.summary("q_factor", 18_400)
braven.summary("extinction_ratio_dB", 22.1)

braven.upload("spectrum.png")
braven.finish()
```

`company`/`project` are only required if your API key can see more than one
project — pass them to disambiguate.

## Matplotlib figures

Pass a live `matplotlib.figure.Figure` to `upload()` instead of a file path
and the SDK saves the PNG for you (same as `fig.savefig()` would) *and*
extracts an interactive companion plot from the figure's line/scatter data,
so the experiment page shows a zoomable, hoverable chart alongside the
static image — no extra API to learn:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(wavelengths_nm, through_db, label="Through port")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Transmission (dB)")

braven.upload(fig, name="spectrum.png")
```

This works the same way whether the figure came from a script you run
yourself or a script run through Braven's Pipelines tab — one code path, one
behavior, no separate API for "give me a nice list thumbnail."

Prefer to log a plain array without a matplotlib figure at all?
`plot_series()` does that directly:

```python
braven.plot_series("snr_vs_temp", y=snr_values, x=temperatures, x_label="Temp (C)", y_label="SNR (dB)")
```

## Devices — multiple sensors/units in one experiment

Tag a value with a stable per-device key and the KPI name stays the same
across devices (no `SNR_dev1`, `SNR_dev2`) — the device becomes a separate
coordinate instead of a suffix:

```python
for sensor_id, snr in results.items():
    braven.device(sensor_id).log_summary("SNR", snr)

braven.log_summary("max_device_mismatch", spread)  # experiment-level, all devices
```

A device is auto-created on first sight and its history accumulates across
experiments — pass an optional type on first use
(`braven.device("SENSOR-4471", "Photodiode")`) to name what kind of device
it is; it's ignored once the device already exists.

## Pipeline scripts (run by the Braven worker)

If your script runs through Braven's Pipelines tab (server-side, executed by
the Braven worker against uploaded files) rather than on your own machine,
the SDK dispatches to the same functions automatically — no `braven.init()`
call needed, and no credentials to manage (the worker supplies the
experiment context):

```python
import braven

def process(braven=None):
    braven.log_config("lr", "0.001")
    braven.log_summary("acc", "0.94")
    braven.log_artifact("plot.png")

    # Devices and matplotlib figures work exactly as in direct logging:
    braven.device("SENSOR-1").log_summary("SNR", 14.2)
    braven.upload(fig, name="spectrum.png")
```

`braven.files` (uploaded file → local temp path) and `braven.params`
(extracted parameters) are populated by the worker before `process()` runs.

## Querying experiments

Read-only access to experiments already logged in Braven — no API key
required beyond what you already set up with `python -m braven login`, or
construct a client directly:

```python
from braven import Braven

b = Braven("https://your-backend.example.com", api_key="braven_...")
exp = b.get("high temp run")          # partial, case-insensitive name match
df = exp.file("data.csv").as_dataframe()  # requires: pip install "braven[dataframe]"

print(exp.metadata)   # {key: value} config/summary metadata
for f in exp.files:
    print(f.filename)
```

`b.get()` also accepts a list of names and returns a list of `Experiment`
objects. `b.experiments()` returns lightweight summaries of everything
visible to the API key.

## Cross-experiment analysis

Pull an explicit, frozen set of experiments out of Braven for local work
(training a model, building a comparison plot — anything outside Braven),
then log results back onto the same object so the record of what was
analyzed can never drift from what was actually read:

```python
analysis = braven.collect(
    experiment_ids=["exp_1", "exp_2", "exp_3"],
    name="Cross-lot yield comparison",
)

for exp in analysis.experiments():
    df = exp.file("data.csv").as_dataframe()
    # ... do local analysis ...

analysis.log_summary("mean_yield", 0.94)
analysis.upload(fig, name="comparison.png")
```

`filter_token` (copied from Braven's webapp "Copy as code" action) works as
an alternative to an explicit `experiment_ids` list — it's re-resolved to
concrete experiment IDs at call time, never stored as a live query.

## Optional dependencies

`requests` is the only hard dependency. Everything else is optional and
imported lazily:

| Feature | Install |
|---|---|
| `.as_dataframe()` on a downloaded file | `pip install "braven[dataframe]"` |
| Uploading a live matplotlib `Figure` (and its list thumbnail) | `pip install "braven[plots]"` |

## Contributing / issues

Source: [github.com/bravenlab/braven-python](https://github.com/bravenlab/braven-python).
Found a bug or have a feature request? File it on
[GitHub Issues](https://github.com/bravenlab/braven-python/issues) — that's
the right place for anything specific to the SDK itself.

## License

MIT — see [LICENSE](LICENSE).
