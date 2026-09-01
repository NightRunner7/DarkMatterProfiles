"""
Compare cSIDM and rSIDM gravothermal evolution for Gilman benchmark halos.

This version uses only the convention

    sigma_1D = 1.10 * sigma_s

and keeps two data constructions clearly separated:

    C-based:
        Halo quantities and sigma_eff obtained from the computation-based setup.

    T-based:
        Halo quantities taken from the tabulated Gilman benchmark setup.

For every halo mass, cSIDM and rSIDM are normalized using the same tau_kappa
appropriate to the selected basis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import GravothermalData as gravoF
import config as cfg
from TruncatedNFWProfile import TruncatedNFWProfile


# =============================================================================
# Global settings
# =============================================================================

Basis = Literal["C-based", "T-based"]

BETA = 0.85
SIGMA_1D_FACTOR = 1.05
ELEMENTS = 3
RSIDM_CORE_INDEX = 2

PLOT_SAVE = False
SHOW_PLOTS = True
OUTPUT_DIR = Path("Results")

# Which plots should be generated?
MAKE_C_BASED_PLOT = True
MAKE_T_BASED_PLOT = True
MAKE_SINGLE_MASS_PLOTS = False

# New preferred option:
# one figure per mass, with two panels:
#   left  -> physical time T [Gyr]
#   right -> normalized time T / tau_kappa
# and in each panel: C-based vs T-based, for both cSIDM and rSIDM
MAKE_C_VS_T_MASS_PANEL_PLOTS = True
PRINT_HALO_CONSISTENCY = True

# Used only when MAKE_SINGLE_MASS_PLOTS = True
SINGLE_MASS_INDEX = 0

# Used only when MAKE_C_VS_T_MASS_PANEL_PLOTS = True
MASS_INDICES_TO_PLOT = [0, 1, 2, 3, 4]

# Plot normalization:
#   "tau_kappa" -> t / tau_kappa
#   "collapse"  -> t / t_end of the corresponding cSIDM run
#   "physical"  -> raw simulation time
TIME_NORMALIZATION = "tau_kappa"


# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

# select RSIDM FILES
DATA_RSIDM_C_BASED = ROOT_DIR / "Data" / "rSIDM" / "Base-C"
DATA_RSIDM_T_BASED = ROOT_DIR / "Data" / "rSIDM" / "Base-T"

# select CSIDM FILES
if SIGMA_1D_FACTOR == 1.05:
    DATA_CSIDM_C_BASED = ROOT_DIR / "Data" / "cSIDM" / "Luca-C-Base"
    DATA_CSIDM_T_BASED = ROOT_DIR / "Data" / "cSIDM" / "Luca-T-Base"
    # C-BASE: PARAMETERS
    C_BASED_CONCENTRATIONS = np.array([21.196, 19.816, 18.436, 17.056, 15.676])
    C_BASED_SIGMA_EFF = np.array([26.995, 47.414, 30.709, 16.003, 9.7091])
    C_BASED_CSIDM_FILES = [
        "CSIDM_Gilman_M7.0_c21.196_sigma26.995_beta0.85.csv",
        "CSIDM_Gilman_M7.5_c19.816_sigma47.414_beta0.85.csv",
        "CSIDM_Gilman_M8.0_c18.436_sigma30.709_beta0.85.csv",
        "CSIDM_Gilman_M8.5_c17.056_sigma16.003_beta0.85.csv",
        "CSIDM_Gilman_M9.0_c15.676_sigma9.7091_beta0.85.csv",
    ]
    # T-BASE: concentration
    T_BASED_CONCENTRATIONS = np.array([21.21, 19.81, 18.42, 17.05, 15.69])
    T_BASED_SIGMA_EFF = np.array([25.068, 47.321, 30.676, 16.13, 9.6506])
    T_BASED_R200_KPC = np.array([4.55, 6.68, 9.81, 14.4, 21.1])
    T_BASED_RS_KPC = np.array([0.210, 0.340, 0.530, 0.840, 1.350])
    T_BASED_LOG10_RHO_S = np.array([7.57, 7.50, 7.42, 7.33, 7.24])
    T_BASED_CSIDM_FILES = [
        "CSIDM_Gilman_M7.0_c21.21_sigma25.068_beta0.85.csv",
        "CSIDM_Gilman_M7.5_c19.81_sigma47.321_beta0.85.csv",
        "CSIDM_Gilman_M8.0_c18.42_sigma30.676_beta0.85.csv",
        "CSIDM_Gilman_M8.5_c17.05_sigma16.13_beta0.85.csv",
        "CSIDM_Gilman_M9.0_c15.69_sigma9.6506_beta0.85.csv",
    ]


elif SIGMA_1D_FACTOR == 1.10:
    DATA_CSIDM_C_BASED = ROOT_DIR / "Data" / "cSIDM" / "Krzysztof-C-Base"
    DATA_CSIDM_T_BASED = ROOT_DIR / "Data" / "cSIDM" / "Krzysztof-T-Base"
    # C-BASE: concentration
    C_BASED_CONCENTRATIONS = np.array([21.21, 19.81, 18.42, 17.05, 15.69])
    C_BASED_SIGMA_EFF = np.array([30.545, 46.855, 28.378, 14.986, 9.190])
    C_BASED_CSIDM_FILES = [
        "CSIDM_Gilman_M7.0_c21.21_sigma30.545_beta0.85.csv",
        "CSIDM_Gilman_M7.5_c19.81_sigma46.855_beta0.85.csv",
        "CSIDM_Gilman_M8.0_c18.42_sigma28.378_beta0.85.csv",
        "CSIDM_Gilman_M8.5_c17.05_sigma14.986_beta0.85.csv",
        "CSIDM_Gilman_M9.0_c15.69_sigma9.19_beta0.85.csv",
    ]
    # T-BASE: concentration
    T_BASED_CONCENTRATIONS = np.array([21.21, 19.81, 18.42, 17.05, 15.69])
    T_BASED_SIGMA_EFF = np.array([28.482, 46.571, 28.334, 15.097, 9.137])
    T_BASED_R200_KPC = np.array([4.55, 6.68, 9.81, 14.4, 21.1])
    T_BASED_RS_KPC = np.array([0.210, 0.340, 0.530, 0.840, 1.350])
    T_BASED_LOG10_RHO_S = np.array([7.57, 7.50, 7.42, 7.33, 7.24])
    T_BASED_CSIDM_FILES = [
        "CSIDM_Gilman_M7.0_c21.21_sigma28.482_beta0.85.csv",
        "CSIDM_Gilman_M7.5_c19.81_sigma46.571_beta0.85.csv",
        "CSIDM_Gilman_M8.0_c18.42_sigma28.334_beta0.85.csv",
        "CSIDM_Gilman_M8.5_c17.05_sigma15.097_beta0.85.csv",
        "CSIDM_Gilman_M9.0_c15.69_sigma9.137_beta0.85.csv",
    ]



# =============================================================================
# Benchmark definitions
# =============================================================================

@dataclass(frozen=True)
class HaloRun:
    """Input information for one halo mass and one data basis."""

    log10_mass: float
    concentration: float
    sigma_eff: float
    csidm_file: str
    rsidm_file: str

    # Used only for T-based halos.
    r200_kpc: float | None = None
    rs_kpc: float | None = None
    log10_rho_s: float | None = None

    @property
    def mass_msun(self) -> float:
        return 10.0**self.log10_mass



# -----------------------------------------------------------------------------
# Model independent parameters
# -----------------------------------------------------------------------------
MASSES_LOG10 = np.array([7.0, 7.5, 8.0, 8.5, 9.0])

C_BASED_RSIDM_FILES = [
    "RSIDM_Gilman_M7.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv",
    "RSIDM_Gilman_M7.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv",
    "RSIDM_Gilman_M8.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv",
    "RSIDM_Gilman_M8.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv",
    "RSIDM_Gilman_M9.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv",
]

T_BASED_RSIDM_FILES = [
    "RSIDM_Gilman_M7.0.csv",
    "RSIDM_Gilman_M7.5.csv",
    "RSIDM_Gilman_M8.0.csv",
    "RSIDM_Gilman_M8.5.csv",
    "RSIDM_Gilman_M9.0.csv",
]

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

def build_run_table() -> dict[Basis, list[HaloRun]]:
    """Collect all run metadata in one place."""

    c_based = [
        HaloRun(
            log10_mass=float(mass),
            concentration=float(concentration),
            sigma_eff=float(sigma_eff),
            csidm_file=csidm_file,
            rsidm_file=rsidm_file,
        )
        for mass, concentration, sigma_eff, csidm_file, rsidm_file in zip(
            MASSES_LOG10,
            C_BASED_CONCENTRATIONS,
            C_BASED_SIGMA_EFF,
            C_BASED_CSIDM_FILES,
            C_BASED_RSIDM_FILES
        )
    ]

    t_based = [
        HaloRun(
            log10_mass=float(mass),
            concentration=float(concentration),
            sigma_eff=float(sigma_eff),
            csidm_file=csidm_file,
            rsidm_file=rsidm_file,
            r200_kpc=float(r200),
            rs_kpc=float(rs),
            log10_rho_s=float(log_rho_s),
        )
        for (
            mass,
            concentration,
            sigma_eff,
            csidm_file,
            rsidm_file,
            r200,
            rs,
            log_rho_s,
        ) in zip(
            MASSES_LOG10,
            T_BASED_CONCENTRATIONS,
            T_BASED_SIGMA_EFF,
            T_BASED_CSIDM_FILES,
            T_BASED_RSIDM_FILES,
            T_BASED_R200_KPC,
            T_BASED_RS_KPC,
            T_BASED_LOG10_RHO_S
        )
    ]

    return {
        "C-based": c_based,
        "T-based": t_based,
    }


RUNS = build_run_table()


# =============================================================================
# Loading and time normalization
# =============================================================================

@dataclass
class LoadedPair:
    """Loaded cSIDM/rSIDM data and their common normalization."""

    metadata: HaloRun
    csidm: object
    rsidm: object
    tau_kappa: float


def validate_input_file(directory: Path, filename: str) -> None:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")


def load_gravothermal_run(
    filename: str,
    directory: Path,
    mass_msun: float,
    sigma_eff: float,
):
    """Load one simulation and attach the metadata used elsewhere in the code."""

    validate_input_file(directory, filename)

    model = gravoF.create_gravothermalData_from_file(
        filename,
        str(directory),
        beta=BETA,
    )
    model.put_the_name("NFW")
    model.put_extra_parameters(mass_msun, sigma_eff)

    return model


def rescale_time_manual(
    sigma_eff: float,
    rs_kpc: float,
    log10_rho_s: float,
    beta: float = BETA,
) -> float:
    """
    Calculate tau_kappa from explicitly tabulated r_s and rho_s.

    Parameters
    ----------
    sigma_eff
        Effective constant cross section in cm^2/g.
    rs_kpc
        NFW scale radius in kpc.
    log10_rho_s
        log10(rho_s / (M_sun kpc^-3)).
    beta
        Gravothermal conductivity calibration parameter.
    """

    rho_s = 10.0**log10_rho_s

    sigma_m_si = sigma_eff * 1e-4 * 1e3
    sigma_m_star = sigma_m_si * cfg.kpc_SI**-2 * cfg.M_solar_SI

    g_star = cfg.const_G_starUnits

    tau_kappa = (
        (150.0 / beta)
        * (1.0 / (rs_kpc * rho_s))
        * (1.0 / sigma_m_star)
        * (4.0 * np.pi * g_star * rho_s) ** (-0.5)
    )

    return float(tau_kappa)


def calculate_tau_kappa(run: HaloRun, basis: Basis) -> float:
    """Use one consistent tau_kappa for cSIDM and rSIDM of the same halo."""

    if basis == "C-based":
        halo = TruncatedNFWProfile(
            _M_vir=run.mass_msun,
            _con=run.concentration,
        )

    elif basis == "T-based":
        if (
            run.r200_kpc is None
            or run.rs_kpc is None
            or run.log10_rho_s is None
        ):
            raise ValueError(
                "T-based run requires r200_kpc, rs_kpc and log10_rho_s."
            )

        # Fully explicit benchmark mode:
        # preserve all tabulated halo quantities exactly as supplied.
        # The profile class reports any NFW inconsistency but does not abort.
        halo = TruncatedNFWProfile(
            _M_vir=run.mass_msun,
            _con=run.concentration,
            _r200=run.r200_kpc,
            _r_s=run.rs_kpc,
            _log_rho_s=run.log10_rho_s,
            strict_consistency=False,
        )

    else:
        raise ValueError(f"Unknown basis: {basis}")

    return float(
        halo.tau(
            beta=BETA,
            sigma_eff=run.sigma_eff,
        )
    )


def print_halo_consistency(run: HaloRun, basis: Basis) -> None:
    """Print profile-implied mass and concentration for one benchmark halo."""

    if basis == "C-based":
        halo = TruncatedNFWProfile(
            _M_vir=run.mass_msun,
            _con=run.concentration,
        )
    elif basis == "T-based":
        if (
            run.r200_kpc is None
            or run.rs_kpc is None
            or run.log10_rho_s is None
        ):
            raise ValueError(
                "T-based run requires r200_kpc, rs_kpc and log10_rho_s."
            )

        halo = TruncatedNFWProfile(
            _M_vir=run.mass_msun,
            _con=run.concentration,
            _r200=run.r200_kpc,
            _r_s=run.rs_kpc,
            _log_rho_s=run.log10_rho_s,
            strict_consistency=False,
        )
    else:
        raise ValueError(f"Unknown basis: {basis}")

    print(
        f"[{basis}] log10M={run.log10_mass:.1f} | "
        f"M_input={halo.M_vir_input:.6e} | "
        f"M_profile={halo.M200_inner:.6e} | "
        f"mass_mismatch={halo.mass_mismatch:+.3e} | "
        f"c_input={halo.con:.6f} | "
        f"c_implied={halo.con_implied:.6f} | "
        f"con_mismatch={halo.con_mismatch:+.3e}"
    )


def load_pair(run: HaloRun, basis: Basis) -> LoadedPair:
    """Load matching cSIDM and rSIDM runs for one halo."""

    if basis == "C-based":
        csidm_directory = DATA_CSIDM_C_BASED
        rsidm_directory = DATA_RSIDM_C_BASED
    elif basis == "T-based":
        csidm_directory = DATA_CSIDM_T_BASED
        rsidm_directory = DATA_RSIDM_T_BASED
    else:
        raise ValueError(f"Unknown basis: {basis}")

    csidm = load_gravothermal_run(
        filename=run.csidm_file,
        directory=csidm_directory,
        mass_msun=run.mass_msun,
        sigma_eff=run.sigma_eff,
    )

    rsidm = load_gravothermal_run(
        filename=run.rsidm_file,
        directory=rsidm_directory,
        mass_msun=run.mass_msun,
        sigma_eff=run.sigma_eff,
    )

    return LoadedPair(
        metadata=run,
        csidm=csidm,
        rsidm=rsidm,
        tau_kappa=calculate_tau_kappa(run, basis),
    )


def normalize_times(
    t_csidm: np.ndarray,
    t_rsidm: np.ndarray,
    tau_kappa: float,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply the selected common time convention."""

    if normalization == "tau_kappa":
        return (
            t_csidm / tau_kappa,
            t_rsidm / tau_kappa,
            r"$t=T/\tau_{\kappa}$",
        )

    if normalization == "collapse":
        collapse_time = float(t_csidm[-1])
        return (
            t_csidm / collapse_time,
            t_rsidm / collapse_time,
            r"$t=T/T_{\mathrm{collapse}}^{\mathrm{cSIDM}}$",
        )

    if normalization == "physical":
        return (
            t_csidm,
            t_rsidm,
            r"$T\,[\mathrm{Gyr}]$",
        )

    raise ValueError(
        "TIME_NORMALIZATION must be 'tau_kappa', 'collapse', or 'physical'."
    )


# =============================================================================
# Plot style
# =============================================================================

def apply_scale_grid(
    ax: plt.Axes,
    xscale: str = "linear",
    yscale: str = "log",
    log_numticks: int = 12,
) -> None:
    """Apply common axis, grid, and tick settings."""

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
# Plotting
# =============================================================================

def plot_all_masses(
    basis: Basis,
    normalization: str = TIME_NORMALIZATION,
) -> None:
    """Plot cSIDM and rSIDM for all five halo masses."""

    pairs = [load_pair(run, basis) for run in RUNS[basis]]

    cmap = mpl.colormaps["plasma"]
    norm = mpl.colors.Normalize(
        vmin=min(MASSES_LOG10),
        vmax=max(MASSES_LOG10),
    )
    colors = [cmap(norm(mass)) for mass in MASSES_LOG10]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    xlabel = ""

    for pair, color in zip(pairs, colors):
        t_csidm, rho_csidm = pair.csidm.return_rho_core_evolution(
            elements=ELEMENTS
        )
        t_rsidm, rho_rsidm = pair.rsidm.return_rho_core_evolution(
            elements=ELEMENTS,
            index=RSIDM_CORE_INDEX,
        )

        t_csidm_plot, t_rsidm_plot, xlabel = normalize_times(
            np.asarray(t_csidm),
            np.asarray(t_rsidm),
            pair.tau_kappa,
            normalization,
        )

        mass_label = (
            rf"$10^{{{pair.metadata.log10_mass:.1f}}}\,M_\odot$"
        )

        ax.plot(
            t_csidm_plot,
            rho_csidm,
            linestyle="--",
            linewidth=2.3,
            color=color,
            alpha=0.85,
            label=mass_label,
        )

        ax.plot(
            t_rsidm_plot,
            rho_rsidm,
            linestyle="-",
            linewidth=2.3,
            color=color,
            alpha=0.95,
        )

        print(
            f"[{basis}] log10(M/Msun)={pair.metadata.log10_mass:.1f} | "
            f"sigma_eff={pair.metadata.sigma_eff:.3f} cm^2/g | "
            f"tau_kappa={pair.tau_kappa:.6g}"
        )

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(
        r"$\hat{\rho}_c \equiv \rho_c/\rho_s$",
        fontsize=18,
    )
    ax.set_title(
        rf"Core-density evolution: cSIDM vs rSIDM ({basis})",
        fontsize=18,
    )
    # ax.set_ylim(1e0, 5e1)
    ax.set_ylim(1e0, 5e2)
    # ax.set_ylim(1e0, 3e3)
    ax.tick_params(labelsize=13)

    apply_scale_grid(
        ax,
        xscale="linear",
        yscale="log",
    )

    mass_legend = ax.legend(
        title="Halo mass",
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.78),
        frameon=True,
        framealpha=0.90,
    )
    ax.add_artist(mass_legend)

    model_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.3,
            linestyle="--",
            label="cSIDM",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.3,
            linestyle="-",
            label="rSIDM",
        ),
    ]

    ax.legend(
        handles=model_handles,
        title="Model",
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
        frameon=True,
        framealpha=0.90,
    )

    ax.text(
        0.98,
        0.03,
        rf"$\sigma_{{1D}}={SIGMA_1D_FACTOR:.2f}\,\sigma_s$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.8",
        },
    )

    fig.tight_layout()

    if PLOT_SAVE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / (
            f"core_density_csidm_rsidm_"
            f"{basis.lower().replace('-', '_')}_"
            f"{normalization}.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_single_mass(
    basis: Basis,
    mass_index: int,
    normalization: str = TIME_NORMALIZATION,
) -> None:
    """Plot one directly matched cSIDM/rSIDM pair."""

    runs = RUNS[basis]
    if not 0 <= mass_index < len(runs):
        raise IndexError(
            f"mass_index must be between 0 and {len(runs) - 1}."
        )

    pair = load_pair(runs[mass_index], basis)

    t_csidm, rho_csidm = pair.csidm.return_rho_core_evolution(
        elements=ELEMENTS
    )
    t_rsidm, rho_rsidm = pair.rsidm.return_rho_core_evolution(
        elements=ELEMENTS,
        index=RSIDM_CORE_INDEX,
    )

    t_csidm_plot, t_rsidm_plot, xlabel = normalize_times(
        np.asarray(t_csidm),
        np.asarray(t_rsidm),
        pair.tau_kappa,
        normalization,
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    ax.plot(
        t_csidm_plot,
        rho_csidm,
        linestyle="--",
        linewidth=2.4,
        label="cSIDM",
    )
    ax.plot(
        t_rsidm_plot,
        rho_rsidm,
        linestyle="-",
        linewidth=2.4,
        label="rSIDM",
    )

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(
        r"$\hat{\rho}_c \equiv \rho_c/\rho_s$",
        fontsize=18,
    )
    ax.set_title(
        rf"{basis}: $M=10^{{{pair.metadata.log10_mass:.1f}}}\,M_\odot$",
        fontsize=18,
    )
    ax.set_ylim(1e0, 3e3)
    ax.tick_params(labelsize=13)

    apply_scale_grid(
        ax,
        xscale="linear",
        yscale="log",
    )

    ax.legend(
        fontsize=12,
        frameon=True,
        framealpha=0.90,
    )

    annotation = "\n".join(
        [
            rf"$\sigma_{{1D}}={SIGMA_1D_FACTOR:.2f}\,\sigma_s$",
            rf"$\sigma_{{\rm eff}}={pair.metadata.sigma_eff:.3f}\,"
            rf"\mathrm{{cm^2/g}}$",
            rf"$\tau_\kappa={pair.tau_kappa:.4g}$",
        ]
    )

    ax.text(
        0.97,
        0.04,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.8",
        },
    )

    fig.tight_layout()

    if PLOT_SAVE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / (
            f"single_mass_{basis.lower().replace('-', '_')}_"
            f"M{pair.metadata.log10_mass:.1f}_"
            f"{normalization}.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)




def plot_c_vs_t_for_mass(
    mass_index: int,
    save: bool = PLOT_SAVE,
) -> None:
    """
    For one halo mass, make a two-panel figure:

    left panel  : physical time T [Gyr]
    right panel : normalized time T / tau_kappa

    In each panel we compare:
        - C-based cSIDM
        - C-based rSIDM
        - T-based cSIDM
        - T-based rSIDM
    """

    if not 0 <= mass_index < len(MASSES_LOG10):
        raise IndexError(
            f"mass_index must be between 0 and {len(MASSES_LOG10) - 1}."
        )

    pair_c = load_pair(RUNS["C-based"][mass_index], "C-based")
    pair_t = load_pair(RUNS["T-based"][mass_index], "T-based")

    # --- Load core-density evolution ---
    t_csidm_c, rho_csidm_c = pair_c.csidm.return_rho_core_evolution(
        elements=ELEMENTS
    )
    t_rsidm_c, rho_rsidm_c = pair_c.rsidm.return_rho_core_evolution(
        elements=ELEMENTS,
        index=RSIDM_CORE_INDEX,
    )

    t_csidm_t, rho_csidm_t = pair_t.csidm.return_rho_core_evolution(
        elements=ELEMENTS
    )
    t_rsidm_t, rho_rsidm_t = pair_t.rsidm.return_rho_core_evolution(
        elements=ELEMENTS,
        index=RSIDM_CORE_INDEX,
    )

    # --- Physical time ---
    t_csidm_c_phys = np.asarray(t_csidm_c)
    t_rsidm_c_phys = np.asarray(t_rsidm_c)

    t_csidm_t_phys = np.asarray(t_csidm_t)
    t_rsidm_t_phys = np.asarray(t_rsidm_t)

    # --- tau_kappa time ---
    t_csidm_c_tau, t_rsidm_c_tau, _ = normalize_times(
        np.asarray(t_csidm_c),
        np.asarray(t_rsidm_c),
        pair_c.tau_kappa,
        "tau_kappa",
    )

    t_csidm_t_tau, t_rsidm_t_tau, _ = normalize_times(
        np.asarray(t_csidm_t),
        np.asarray(t_rsidm_t),
        pair_t.tau_kappa,
        "tau_kappa",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.4),
        sharey=True,
    )

    ax_left, ax_right = axes

    color_c = "tab:red"
    color_t = "tab:blue"

    # -------------------------------------------------------------------------
    # Left panel: physical time
    # -------------------------------------------------------------------------
    ax_left.plot(
        t_csidm_c_phys,
        rho_csidm_c,
        linestyle="--",
        linewidth=2.3,
        color=color_c,
        label="cSIDM (C-based)",
    )
    ax_left.plot(
        t_rsidm_c_phys,
        rho_rsidm_c,
        linestyle="-",
        linewidth=2.3,
        color=color_c,
        label="rSIDM (C-based)",
    )
    ax_left.plot(
        t_csidm_t_phys,
        rho_csidm_t,
        linestyle="--",
        linewidth=2.3,
        color=color_t,
        label="cSIDM (T-based)",
    )
    ax_left.plot(
        t_rsidm_t_phys,
        rho_rsidm_t,
        linestyle="-",
        linewidth=2.3,
        color=color_t,
        label="rSIDM (T-based)",
    )

    # -------------------------------------------------------------------------
    # Right panel: T / tau_kappa
    # -------------------------------------------------------------------------
    ax_right.plot(
        t_csidm_c_tau,
        rho_csidm_c,
        linestyle="--",
        linewidth=2.3,
        color=color_c,
        label="cSIDM (C-based)",
    )
    ax_right.plot(
        t_rsidm_c_tau,
        rho_rsidm_c,
        linestyle="-",
        linewidth=2.3,
        color=color_c,
        label="rSIDM (C-based)",
    )
    ax_right.plot(
        t_csidm_t_tau,
        rho_csidm_t,
        linestyle="--",
        linewidth=2.3,
        color=color_t,
        label="cSIDM (T-based)",
    )
    ax_right.plot(
        t_rsidm_t_tau,
        rho_rsidm_t,
        linestyle="-",
        linewidth=2.3,
        color=color_t,
        label="rSIDM (T-based)",
    )

    # --- Axis styling ---
    for ax in axes:
        ax.set_ylim(1e0, 3e3)
        ax.tick_params(labelsize=12)
        apply_scale_grid(ax, xscale="linear", yscale="log")

    ax_left.set_xlabel(r"$T\,[\mathrm{Gyr}]$", fontsize=16)
    ax_right.set_xlabel(r"$t = T/\tau_{\kappa}$", fontsize=16)

    ax_left.set_ylabel(
        r"$\hat{\rho}_c \equiv \rho_c/\rho_s$",
        fontsize=16,
    )

    mass_label = rf"$M = 10^{{{pair_c.metadata.log10_mass:.1f}}}\,M_\odot$"
    fig.suptitle(
        rf"Comparison of C-based and T-based runs: {mass_label}",
        fontsize=17,
        y=0.98,
    )

    ax_left.set_title("Physical time", fontsize=14)
    ax_right.set_title(r"Time normalized by $\tau_{\kappa}$", fontsize=14)

    ax_left.legend(
        fontsize=10.5,
        frameon=True,
        framealpha=0.92,
        loc="upper left",
    )

    annotation = "\n".join(
        [
            rf"C-based: $c={pair_c.metadata.concentration:.3f}$",
            rf"C-based: $\sigma_{{\rm eff}}={pair_c.metadata.sigma_eff:.3f}$ cm$^2$/g",
            rf"C-based: $\tau_\kappa={pair_c.tau_kappa:.4g}$",
            rf"T-based: $c={pair_t.metadata.concentration:.3f}$",
            rf"T-based: $\sigma_{{\rm eff}}={pair_t.metadata.sigma_eff:.3f}$ cm$^2$/g",
            rf"T-based: $\tau_\kappa={pair_t.tau_kappa:.4g}$",
            rf"$\sigma_{{1D}}={SIGMA_1D_FACTOR:.2f}\,\sigma_s$",
        ]
    )

    ax_right.text(
        0.98,
        0.03,
        annotation,
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.3,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.8",
        },
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / (
            f"compare_C_vs_T_M{pair_c.metadata.log10_mass:.1f}.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_c_vs_t_for_selected_masses() -> None:
    """Loop over selected masses and produce one two-panel figure per mass."""
    for mass_index in MASS_INDICES_TO_PLOT:
        plot_c_vs_t_for_mass(mass_index=mass_index, save=PLOT_SAVE)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("cSIDM-rSIDM comparison")
    print(f"beta                 = {BETA}")
    print(f"sigma_1D convention  = {SIGMA_1D_FACTOR:.2f} * sigma_s")
    print(f"time normalization    = {TIME_NORMALIZATION}")
    print("=" * 72)

    if PRINT_HALO_CONSISTENCY:
        print("\nHalo consistency diagnostics")
        print("-" * 72)
        for basis in ("C-based", "T-based"):
            for run in RUNS[basis]:
                print_halo_consistency(run, basis)
        print("-" * 72)

    if MAKE_C_BASED_PLOT:
        plot_all_masses(
            basis="C-based",
            normalization=TIME_NORMALIZATION,
        )

    if MAKE_T_BASED_PLOT:
        plot_all_masses(
            basis="T-based",
            normalization=TIME_NORMALIZATION,
        )

    if MAKE_SINGLE_MASS_PLOTS:
        plot_single_mass(
            basis="C-based",
            mass_index=SINGLE_MASS_INDEX,
            normalization=TIME_NORMALIZATION,
        )
        plot_single_mass(
            basis="T-based",
            mass_index=SINGLE_MASS_INDEX,
            normalization=TIME_NORMALIZATION,
        )

    if MAKE_C_VS_T_MASS_PANEL_PLOTS:
        plot_c_vs_t_for_selected_masses()


if __name__ == "__main__":
    main()