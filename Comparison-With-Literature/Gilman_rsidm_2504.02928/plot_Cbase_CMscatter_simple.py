"""
Simplified C-based c-M scatter comparison.

Main changes relative to the previous version
---------------------------------------------
1) We plot only ONE sigma_1D convention at a time:
       SIGMA_1D_FACTOR = 1.05
   or
       SIGMA_1D_FACTOR = 1.10

2) Therefore each mass panel contains:
   - 3 rSIDM curves
   - 3 cSIDM curves

3) The legend is simplified:
   - one small legend for line styles (rSIDM / cSIDM),
   - the numerical values of c and sigma_eff for
     (-1 sigma_c, fiducial, +1 sigma_c) are written in
     the annotation box on the right panel.

Expected directory structure
----------------------------
Data/
├── rSIDM/
│   └── C-Base-CMScatter-0p16dex/
│       ├── c-minus1sigma/
│       ├── c-fiducial/
│       └── c-plus1sigma/
└── cSIDM/
    └── C-Base-CMScatter-0p16dex/
        ├── sigma1D-1p05/
        │   ├── c-minus1sigma/
        │   ├── c-fiducial/
        │   └── c-plus1sigma/
        └── sigma1D-1p10/
            ├── c-minus1sigma/
            ├── c-fiducial/
            └── c-plus1sigma/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import GravothermalData as gravoF
from TruncatedNFWProfile import TruncatedNFWProfile


# =============================================================================
# Global settings
# =============================================================================

BETA = 0.85
CM_SCATTER_DEX = 0.16

# ---------------------------------------------------------------------------
# Choose exactly ONE convention:
#     1.05  -> Luca convention
#     1.10  -> Krzysztof convention
# ---------------------------------------------------------------------------
SIGMA_1D_FACTOR = 1.10

ELEMENTS = 3
RSIDM_CORE_INDEX = 2

PLOT_SAVE = False
SHOW_PLOTS = True
OUTPUT_DIR = Path("Results") / "C-Base-CMScatter-0p16dex"

MASS_INDICES_TO_PLOT = [0, 1, 2, 3, 4]

CHECK_FILENAME_CONCENTRATION = True
CONCENTRATION_RTOL = 5e-3


# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_RSIDM_ROOT = (
    ROOT_DIR / "Data" / "rSIDM" / "C-Base-CMScatter-0p16dex"
)

DATA_CSIDM_ROOT = (
    ROOT_DIR / "Data" / "cSIDM" / "C-Base-CMScatter-0p16dex"
)

SCATTER_SUBDIRS = {
    # "minus1sigma": "c-minus1sigma",
    "minus1sigma": "c-fiducial-minus1sigma",
    "fiducial": "c-fiducial",
    # "plus1sigma": "c-plus1sigma",
    "plus1sigma": "c-fiducial-plus1sigma",

}

SIGMA_FACTOR_SUBDIRS = {
    1.05: "sigma1D-1p05",
    1.10: "sigma1D-1p10",
}


# =============================================================================
# Fiducial C-based c-M relation
# =============================================================================

MASSES_LOG10 = np.array([7.0, 7.5, 8.0, 8.5, 9.0])

C_BASED_CONCENTRATIONS_FIDUCIAL = np.array(
    [21.196, 19.816, 18.436, 17.056, 15.676]
)


@dataclass(frozen=True)
class ConcentrationVariant:
    key: str
    dex_offset: float
    short_label: str


CM_VARIANTS = (
    ConcentrationVariant("minus1sigma", -CM_SCATTER_DEX, r"$-1\sigma_c$"),
    ConcentrationVariant("fiducial", 0.0, "fiducial"),
    ConcentrationVariant("plus1sigma", +CM_SCATTER_DEX, r"$+1\sigma_c$"),
)


@dataclass(frozen=True)
class RunSpec:
    log10_mass: float
    mass_msun: float
    variant: ConcentrationVariant
    concentration: float


@dataclass
class LoadedCurve:
    model: object
    sigma_eff: float | None = None
    tau_kappa: float | None = None
    path: Path | None = None


def concentration_from_scatter(c_fiducial: float, dex_offset: float) -> float:
    return float(c_fiducial * 10.0**dex_offset)


def build_run_specs() -> list[list[RunSpec]]:
    all_specs: list[list[RunSpec]] = []

    for log10_mass, c0 in zip(MASSES_LOG10, C_BASED_CONCENTRATIONS_FIDUCIAL):
        specs_for_mass = []
        for variant in CM_VARIANTS:
            specs_for_mass.append(
                RunSpec(
                    log10_mass=float(log10_mass),
                    mass_msun=float(10.0**log10_mass),
                    variant=variant,
                    concentration=concentration_from_scatter(c0, variant.dex_offset),
                )
            )
        all_specs.append(specs_for_mass)

    return all_specs


RUN_SPECS = build_run_specs()


# =============================================================================
# File discovery
# =============================================================================

_SIGMA_RE = re.compile(
    r"_sigma(?P<sigma>[0-9]+(?:\.[0-9]+)?)_",
    flags=re.IGNORECASE,
)

_CONCENTRATION_RE = re.compile(
    r"_c(?P<c>[0-9]+(?:\.[0-9]+)?)_",
    flags=re.IGNORECASE,
)


def _find_unique_file(directory: Path, pattern: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist:\n  {directory}")

    matches = sorted(directory.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matching\n  {pattern}\ninside\n  {directory}"
        )

    if len(matches) > 1:
        formatted = "\n".join(f"  - {p.name}" for p in matches)
        raise RuntimeError(
            "Expected exactly one matching simulation file, "
            f"but found {len(matches)} in {directory}:\n{formatted}"
        )

    return matches[0]


def _parse_sigma_eff_from_filename(path: Path) -> float:
    match = _SIGMA_RE.search(path.name)
    if match is None:
        raise ValueError(
            "Could not parse sigma_eff from cSIDM filename. "
            'Expected a token like "_sigma9.190_".\n'
            f"File: {path.name}"
        )
    return float(match.group("sigma"))


def _check_concentration_in_filename(path: Path, expected_c: float) -> None:
    if not CHECK_FILENAME_CONCENTRATION:
        return

    match = _CONCENTRATION_RE.search(path.name)
    if match is None:
        return

    file_c = float(match.group("c"))
    if not np.isclose(file_c, expected_c, rtol=CONCENTRATION_RTOL, atol=0.0):
        raise ValueError(
            "Concentration mismatch between c-M prescription and filename:\n"
            f"  expected c = {expected_c:.6f}\n"
            f"  filename c = {file_c:.6f}\n"
            f"  file       = {path}"
        )


def rsidm_directory(spec: RunSpec) -> Path:
    return DATA_RSIDM_ROOT / SCATTER_SUBDIRS[spec.variant.key]


def csidm_directory(spec: RunSpec, sigma_1d_factor: float) -> Path:
    if sigma_1d_factor not in SIGMA_FACTOR_SUBDIRS:
        raise ValueError(
            f"Unsupported sigma_1D factor: {sigma_1d_factor}. "
            f"Available: {tuple(SIGMA_FACTOR_SUBDIRS.keys())}"
        )
    return (
        DATA_CSIDM_ROOT
        / SIGMA_FACTOR_SUBDIRS[sigma_1d_factor]
        / SCATTER_SUBDIRS[spec.variant.key]
    )


def find_rsidm_file(spec: RunSpec) -> Path:
    pattern = f"RSIDM_Gilman_M{spec.log10_mass:.1f}_*.csv"
    path = _find_unique_file(rsidm_directory(spec), pattern)
    _check_concentration_in_filename(path, spec.concentration)
    return path


def find_csidm_file(spec: RunSpec, sigma_1d_factor: float) -> Path:
    pattern = f"CSIDM_Gilman_M{spec.log10_mass:.1f}_*.csv"
    path = _find_unique_file(csidm_directory(spec, sigma_1d_factor), pattern)
    _check_concentration_in_filename(path, spec.concentration)
    return path


# =============================================================================
# Loading and tau_kappa
# =============================================================================

def load_gravothermal_run(path: Path, mass_msun: float, sigma_eff: float):
    model = gravoF.create_gravothermalData_from_file(
        path.name,
        str(path.parent),
        beta=BETA,
    )
    model.put_the_name("NFW")
    model.put_extra_parameters(mass_msun, sigma_eff)
    return model


def calculate_tau_kappa(spec: RunSpec, sigma_eff: float) -> float:
    halo = TruncatedNFWProfile(
        _M_vir=spec.mass_msun,
        _con=spec.concentration,
    )
    return float(halo.tau(beta=BETA, sigma_eff=sigma_eff))


def load_csidm(spec: RunSpec, sigma_1d_factor: float) -> LoadedCurve:
    path = find_csidm_file(spec, sigma_1d_factor)
    sigma_eff = _parse_sigma_eff_from_filename(path)
    tau_kappa = calculate_tau_kappa(spec, sigma_eff)

    model = load_gravothermal_run(
        path=path,
        mass_msun=spec.mass_msun,
        sigma_eff=sigma_eff,
    )

    return LoadedCurve(
        model=model,
        sigma_eff=sigma_eff,
        tau_kappa=tau_kappa,
        path=path,
    )


def load_rsidm(spec: RunSpec, sigma_eff_for_metadata: float) -> LoadedCurve:
    path = find_rsidm_file(spec)

    model = load_gravothermal_run(
        path=path,
        mass_msun=spec.mass_msun,
        sigma_eff=sigma_eff_for_metadata,
    )

    return LoadedCurve(
        model=model,
        sigma_eff=sigma_eff_for_metadata,
        path=path,
    )


# =============================================================================
# Plot style
# =============================================================================

SCATTER_COLORS = {
    "minus1sigma": "tab:blue",
    "fiducial": "black",
    "plus1sigma": "tab:red",
}

RSIDM_LINESTYLE = "-"
CSIDM_LINESTYLE = "--"


def apply_scale_grid(
    ax: plt.Axes,
    xscale: str = "linear",
    yscale: str = "log",
    log_numticks: int = 12,
) -> None:
    allowed_scales = {"log", "linear"}

    if xscale not in allowed_scales:
        raise ValueError(f"Invalid xscale: {xscale}")
    if yscale not in allowed_scales:
        raise ValueError(f"Invalid yscale: {yscale}")

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.grid(which="major", alpha=0.40)
    ax.grid(which="minor", alpha=0.20)

    if xscale == "log":
        ax.xaxis.set_major_locator(
            mticker.LogLocator(base=10, numticks=log_numticks)
        )
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(
                base=10,
                subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                numticks=log_numticks,
            )
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.xaxis.set_major_locator(mticker.AutoLocator())
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    if yscale == "log":
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10, numticks=log_numticks)
        )
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(
                base=10,
                subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                numticks=log_numticks,
            )
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.yaxis.set_major_locator(mticker.AutoLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    ax.tick_params(
        "both",
        direction="in",
        top=True,
        right=True,
        length=10,
        width=1,
        which="major",
    )
    ax.tick_params(
        "both",
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1,
        which="minor",
    )


# =============================================================================
# Core-evolution helpers
# =============================================================================

def get_core_evolution_csidm(curve: LoadedCurve):
    t, rho = curve.model.return_rho_core_evolution(elements=ELEMENTS)
    return np.asarray(t), np.asarray(rho)


def get_core_evolution_rsidm(curve: LoadedCurve):
    t, rho = curve.model.return_rho_core_evolution(
        elements=ELEMENTS,
        index=RSIDM_CORE_INDEX,
    )
    return np.asarray(t), np.asarray(rho)


# =============================================================================
# Plotting
# =============================================================================

def print_concentration_table() -> None:
    factor_low = 10.0**(-CM_SCATTER_DEX)
    factor_high = 10.0**(+CM_SCATTER_DEX)

    print("=" * 92)
    print("C-based c-M scatter")
    print(f"scatter = {CM_SCATTER_DEX:.3f} dex")
    print(f"10^(-scatter) = {factor_low:.6f}")
    print(f"10^(+scatter) = {factor_high:.6f}")
    print("-" * 92)
    print(
        f"{'log10M':>8s} "
        f"{'c(-1sigma)':>14s} "
        f"{'c(fid)':>14s} "
        f"{'c(+1sigma)':>14s}"
    )
    print("-" * 92)

    for specs in RUN_SPECS:
        by_key = {spec.variant.key: spec.concentration for spec in specs}
        print(
            f"{specs[0].log10_mass:8.3f} "
            f"{by_key['minus1sigma']:14.6f} "
            f"{by_key['fiducial']:14.6f} "
            f"{by_key['plus1sigma']:14.6f}"
        )

    print("=" * 92)


def print_expected_directories() -> None:
    print("\nExpected data directories")
    print("-" * 92)

    for variant in CM_VARIANTS:
        print("rSIDM:", DATA_RSIDM_ROOT / SCATTER_SUBDIRS[variant.key])

    sample_specs = RUN_SPECS[0]
    for spec in sample_specs:
        print(
            f"cSIDM ({SIGMA_1D_FACTOR:.2f}):",
            csidm_directory(spec, SIGMA_1D_FACTOR),
        )

    print("-" * 92)


def plot_cm_scatter_for_mass(mass_index: int, save: bool = PLOT_SAVE) -> None:
    if not 0 <= mass_index < len(RUN_SPECS):
        raise IndexError(
            f"mass_index must be between 0 and {len(RUN_SPECS) - 1}."
        )

    specs = RUN_SPECS[mass_index]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.4),
        sharey=True,
    )
    ax_left, ax_right = axes

    annotation_rows = [
        rf"$\sigma_{{1D}}={SIGMA_1D_FACTOR:.2f}\,\sigma_s$",
        rf"$\sigma_{{\log_{{10}}c}}={CM_SCATTER_DEX:.2f}\,\mathrm{{dex}}$",
        "",
    ]

    for spec in specs:
        color = SCATTER_COLORS[spec.variant.key]

        csidm_curve = load_csidm(spec, SIGMA_1D_FACTOR)
        rsidm_curve = load_rsidm(
            spec,
            sigma_eff_for_metadata=csidm_curve.sigma_eff,
        )

        t_csidm, rho_csidm = get_core_evolution_csidm(csidm_curve)
        t_rsidm, rho_rsidm = get_core_evolution_rsidm(rsidm_curve)

        # left: physical time
        ax_left.plot(
            t_rsidm,
            rho_rsidm,
            color=color,
            linestyle=RSIDM_LINESTYLE,
            linewidth=2.4,
        )
        ax_left.plot(
            t_csidm,
            rho_csidm,
            color=color,
            linestyle=CSIDM_LINESTYLE,
            linewidth=2.4,
        )

        # right: normalized by the tau_kappa associated with the same
        # concentration and the same chosen sigma_1D convention.
        ax_right.plot(
            t_rsidm / csidm_curve.tau_kappa,
            rho_rsidm,
            color=color,
            linestyle=RSIDM_LINESTYLE,
            linewidth=2.4,
        )
        ax_right.plot(
            t_csidm / csidm_curve.tau_kappa,
            rho_csidm,
            color=color,
            linestyle=CSIDM_LINESTYLE,
            linewidth=2.4,
        )

        print(
            f"M=1e{spec.log10_mass:.1f} Msun | "
            f"{spec.variant.key:>11s} | "
            f"c={spec.concentration:.6f} | "
            f"sigma1D={SIGMA_1D_FACTOR:.2f} sigma_s | "
            f"sigma_eff={csidm_curve.sigma_eff:.6g} cm^2/g | "
            f"tau_kappa={csidm_curve.tau_kappa:.6g}"
        )

        annotation_rows.append(
            rf"{spec.variant.short_label}: "
            rf"$c={spec.concentration:.3f}$, "
            rf"$\sigma_{{\rm eff}}={csidm_curve.sigma_eff:.3f}$"
        )

    for ax in axes:
        ax.set_ylim(1e0, 3e3)
        ax.tick_params(labelsize=12)
        apply_scale_grid(ax, xscale="linear", yscale="log")

    ax_left.set_xlabel(r"$T\,[\mathrm{Gyr}]$", fontsize=16)
    ax_right.set_xlabel(r"$t=T/\tau_{\kappa}$", fontsize=16)
    ax_left.set_ylabel(r"$\hat{\rho}_c \equiv \rho_c/\rho_s$", fontsize=16)

    log10_mass = specs[0].log10_mass
    fig.suptitle(
        rf"C-based c-M scatter: $M=10^{{{log10_mass:.1f}}}\,M_\odot$",
        fontsize=17,
        y=0.98,
    )

    ax_left.set_title("Physical time", fontsize=14)
    ax_right.set_title(r"Time normalized by $\tau_\kappa$", fontsize=14)

    # -------------------------------------------------------------------------
    # Legend 1: model = line style
    # -------------------------------------------------------------------------
    model_handles = [
        mpl.lines.Line2D(
            [0], [0],
            color="0.2",
            linewidth=2.4,
            linestyle=RSIDM_LINESTYLE,
            label="rSIDM",
        ),
        mpl.lines.Line2D(
            [0], [0],
            color="0.2",
            linewidth=2.4,
            linestyle=CSIDM_LINESTYLE,
            label="cSIDM",
        ),
    ]

    model_legend = ax_left.legend(
        handles=model_handles,
        title="Model",
        fontsize=10.5,
        title_fontsize=10.5,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
    )

    # Keep the first legend when adding the second one
    ax_left.add_artist(model_legend)

    # -------------------------------------------------------------------------
    # Legend 2: c-M realization = color
    # -------------------------------------------------------------------------
    scatter_handles = [
        mpl.lines.Line2D(
            [0], [0],
            color=SCATTER_COLORS["minus1sigma"],
            linewidth=2.5,
            linestyle="-",
            label=r"$-1\sigma_c$",
        ),
        mpl.lines.Line2D(
            [0], [0],
            color=SCATTER_COLORS["fiducial"],
            linewidth=2.5,
            linestyle="-",
            label="fiducial",
        ),
        mpl.lines.Line2D(
            [0], [0],
            color=SCATTER_COLORS["plus1sigma"],
            linewidth=2.5,
            linestyle="-",
            label=r"$+1\sigma_c$",
        ),
    ]

    ax_left.legend(
        handles=scatter_handles,
        title=r"$c$ realization",
        fontsize=10.5,
        title_fontsize=10.5,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.77),
        frameon=True,
        framealpha=0.92,
    )

    annotation = "\n".join(annotation_rows)
    ax_right.text(
        0.98,
        0.03,
        annotation,
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.0,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.87,
            "edgecolor": "0.8",
        },
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = "1p05" if np.isclose(SIGMA_1D_FACTOR, 1.05) else "1p10"
        output_path = OUTPUT_DIR / f"compare_Cbase_CMscatter_sigma{tag}_M{log10_mass:.1f}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_selected_masses() -> None:
    for mass_index in MASS_INDICES_TO_PLOT:
        plot_cm_scatter_for_mass(mass_index=mass_index, save=PLOT_SAVE)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print_concentration_table()
    print_expected_directories()

    print("\nPlot configuration")
    print("-" * 92)
    print(f"beta = {BETA}")
    print(f"sigma_1D factor = {SIGMA_1D_FACTOR:.2f} * sigma_s")
    print("-" * 92)

    plot_selected_masses()


if __name__ == "__main__":
    main()
