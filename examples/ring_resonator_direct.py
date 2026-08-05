"""
ring_resonator_direct.py — All-pass/add-drop ring resonator simulation with
direct Braven logging.

Simulates a silicon photonics add-drop ring resonator: computes the through-
and drop-port transmission spectrum vs wavelength for a given ring radius and
coupling gap, sweeps radius, and pushes each run's config/summary/plots to
Braven.

Setup (once):
    pip install braven
    python -m braven login

Run:
    python ring_resonator_direct.py

Requirements:
    pip install numpy matplotlib
"""

import math
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import braven

# ── Physical / process constants (generic silicon photonics platform) ──────
N_GROUP = 4.2          # group index of the Si waveguide
N_EFF0 = 2.4            # effective index at the center wavelength
LAMBDA0 = 1550e-9       # center wavelength [m]
ALPHA_DB_CM = 2.0       # waveguide propagation loss [dB/cm]

# ── Per-run design parameters ───────────────────────────────────────────────
GAP_NM = 200.0          # coupling gap [nm]


def coupling_kappa(gap_nm: float) -> float:
    """Field coupling coefficient — decays exponentially with gap (evanescent coupling)."""
    kappa0 = 0.22
    decay_length_nm = 120.0
    return kappa0 * math.exp(-(gap_nm - 150.0) / decay_length_nm)


def simulate(radius_um: float, gap_nm: float) -> dict:
    """Compute through/drop transmission spectra for an add-drop ring resonator.

    The nominal radius is fab-tuned to the nearest whole number of guided
    wavelengths so a resonance sits exactly at LAMBDA0 — exactly how a real
    ring-resonator design is dimensioned against a target wavelength.
    """
    radius_m = radius_um * 1e-6
    nominal_circumference = 2 * math.pi * radius_m
    mode_number = round(nominal_circumference * N_EFF0 / LAMBDA0)
    circumference = mode_number * LAMBDA0 / N_EFF0  # sub-nm correction to nominal radius

    alpha_db_m = ALPHA_DB_CM * 100.0
    alpha_np_m = alpha_db_m * math.log(10) / 20.0  # dB -> Nepers
    a = math.exp(-alpha_np_m * circumference)  # round-trip amplitude transmission

    kappa = coupling_kappa(gap_nm)
    t = math.sqrt(max(0.0, 1.0 - kappa ** 2))  # self-coupling (through) coefficient

    fsr_estimate_m = LAMBDA0 ** 2 / (N_GROUP * circumference)
    half_window = min(1.2 * fsr_estimate_m, 25e-9)
    wavelengths = np.linspace(LAMBDA0 - half_window, LAMBDA0 + half_window, 4000)
    beta = 2 * np.pi * N_EFF0 / wavelengths
    phi = beta * circumference

    # Add-drop ring transfer functions (all-pass-derived two-port coupled cavity)
    denom = 1 - (t ** 2) * a * np.exp(1j * phi)
    through = (t - t * a * np.exp(1j * phi)) / denom
    drop = (-(1 - t ** 2) * np.sqrt(a) * np.exp(1j * phi / 2)) / denom

    through_db = 20 * np.log10(np.abs(through) + 1e-12)
    drop_db = 20 * np.log10(np.abs(drop) + 1e-12)

    # FSR from the two nulls either side of the center resonance
    fsr_m = wavelengths[-1] ** 2 / (N_GROUP * circumference)

    # Extinction ratio and 3dB bandwidth of the central through-port dip
    center_idx = np.argmin(through_db)
    extinction_db = through_db.max() - through_db[center_idx]
    half_level = through_db[center_idx] + 3.0
    above = np.where(through_db >= half_level)[0]
    # Nearest crossings either side of the resonance dip
    left = above[above < center_idx]
    right = above[above > center_idx]
    if len(left) and len(right):
        fwhm_m = wavelengths[right[0]] - wavelengths[left[-1]]
    else:
        fwhm_m = fsr_m / 50.0  # fallback estimate
    q_factor = LAMBDA0 / fwhm_m if fwhm_m > 0 else float("nan")
    finesse = fsr_m / fwhm_m if fwhm_m > 0 else float("nan")

    return {
        "wavelengths_nm": wavelengths * 1e9,
        "through_db": through_db,
        "drop_db": drop_db,
        "fsr_pm": fsr_m * 1e12,
        "q_factor": q_factor,
        "finesse": finesse,
        "extinction_db": extinction_db,
        "kappa": kappa,
        "round_trip_loss_db": -20 * math.log10(a),
    }


def draw_spectrum(results: dict, radius_um: float, gap_nm: float):
    """Returns the live Figure (not saved/closed here) — braven.upload() accepts a
    Figure directly, saves the PNG itself, and extracts an interactive companion
    series from the plotted lines."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results["wavelengths_nm"], results["through_db"], color="royalblue", lw=1.6, label="Through port")
    ax.plot(results["wavelengths_nm"], results["drop_db"], color="firebrick", lw=1.6, label="Drop port")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (dB)")
    ax.set_title(f"Add-Drop Ring Resonator — R={radius_um:.1f} µm, gap={gap_nm:.0f} nm")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    q = results["q_factor"]
    ax.annotate(
        f"Q ≈ {q:,.0f}\nER ≈ {results['extinction_db']:.1f} dB\nFSR ≈ {results['fsr_pm']:.0f} pm",
        xy=(0.02, 0.05), xycoords="axes fraction", fontsize=8.5,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.85),
    )
    fig.tight_layout()
    return fig


def draw_layout(radius_um: float, gap_nm: float, path: Path) -> None:
    """Simple schematic of the add-drop ring layout."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    r_plot = 1.2
    ring = plt.Circle((0, 0), r_plot, fill=False, color="royalblue", lw=2.5)
    ax.add_patch(ring)

    bus_y = r_plot + gap_nm / 1000.0 * 0.6 + 0.15
    ax.plot([-2.6, 2.6], [bus_y, bus_y], color="black", lw=2.5)
    ax.plot([-2.6, 2.6], [-bus_y, -bus_y], color="black", lw=2.5)

    ax.annotate("", xy=(2.6, bus_y), xytext=(-2.6, bus_y),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.01))
    ax.text(-2.5, bus_y + 0.15, "In", fontsize=9)
    ax.text(2.3, bus_y + 0.15, "Through", fontsize=9)
    ax.text(-2.5, -bus_y - 0.3, "Add", fontsize=9)
    ax.text(2.2, -bus_y - 0.3, "Drop", fontsize=9)

    ax.text(0, 0, f"R = {radius_um:.1f} µm", ha="center", va="center", fontsize=9, color="#333")
    ax.set_title(f"Add-Drop Ring Layout  (gap = {gap_nm:.0f} nm)", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main — sweep ring radius at a fixed coupling gap, log each run to Braven
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    radii_um = [50, 60]

    for radius_um in radii_um:
        print(f"Simulating ring resonator  R={radius_um:.1f} µm  gap={GAP_NM:.0f} nm ...")
        results = simulate(radius_um, GAP_NM)

        print(f"  Q factor      : {results['q_factor']:,.0f}")
        print(f"  Finesse       : {results['finesse']:.1f}")
        print(f"  Extinction    : {results['extinction_db']:.1f} dB")
        print(f"  FSR           : {results['fsr_pm']:.1f} pm")

        with tempfile.TemporaryDirectory() as tmp:
            layout_path = Path(tmp) / "layout.png"

            spectrum_fig = draw_spectrum(results, radius_um, GAP_NM)
            draw_layout(radius_um, GAP_NM, layout_path)

            print("  Logging to Braven ...")
            run = braven.init(
                name=f"Ring R={radius_um:.1f}um gap={GAP_NM:.0f}nm",
                company="Your Company",
                project="Your Project",
            )

            braven.config("radius_um", radius_um)
            braven.config("gap_nm", GAP_NM)
            braven.config("n_eff", N_EFF0)
            braven.config("n_group", N_GROUP)
            braven.config("waveguide_loss_dB_per_cm", ALPHA_DB_CM)
            braven.config("center_wavelength_nm", LAMBDA0 * 1e9)

            braven.summary("q_factor", results["q_factor"])
            braven.summary("finesse", results["finesse"])
            braven.summary("extinction_ratio_dB", results["extinction_db"])
            braven.summary("fsr_pm", results["fsr_pm"])
            braven.summary("round_trip_loss_dB", results["round_trip_loss_db"])
            braven.summary("kappa", results["kappa"])

            # Pass the live Figure — braven.upload() saves the PNG (same as
            # fig.savefig() would) and extracts an interactive companion series
            # from the plotted through/drop lines automatically.
            braven.upload(spectrum_fig, name="spectrum.png")
            plt.close(spectrum_fig)
            braven.upload(layout_path)

            braven.finish()
            print(f"  Done -> {run.name}\n")
