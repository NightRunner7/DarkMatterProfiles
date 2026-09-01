"""
Standalone core-Knudsen-number evolution plots for the Gilman benchmark.

For each selected halo mass, this script produces a two-panel figure:

    left panel  : physical time T [Gyr]
    right panel : normalized time T / tau_kappa

Each panel compares cSIDM/rSIDM and C-based/T-based runs.

The global arrays C_BASED_RSIDM_SIGMA_EFF and T_BASED_RSIDM_SIGMA_EFF
are used only for the rSIDM tau_kappa normalization. The local rSIDM
Knudsen number is evaluated using K5(nu) built from the tabulated Gilman
cross section sigma(v_rel)/m.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import interp1d

import GravothermalData as gravoF
import config as cfg
from TruncatedNFWProfile import TruncatedNFWProfile


Basis = Literal["C-based", "T-based"]

BETA = 0.85
SIGMA_1D_FACTOR = 1.1

ELEMENTS_RHO = 1
ELEMENTS_VEL = 1
USE_CORE_AVERAGE = False
STOP_AT_COLLAPSE = False

PLOT_SAVE = False
SHOW_PLOTS = True
OUTPUT_DIR = Path("Results")
MASS_INDICES_TO_PLOT = [0, 1, 2, 3, 4]

X_SCALE = "log"
Y_SCALE = "log"

K5_P = 5
K5_NU_MIN_KMS = 0.05
K5_NU_MAX_KMS = 3000.0
K5_NU_POINTS = 500
K5_X_MIN = 1e-4
K5_X_MAX = 30.0
K5_X_POINTS = 5000

ROOT_DIR = Path(__file__).resolve().parent
GILMAN_CROSS_SECTION_FILE = ROOT_DIR / "Crossv_May7.csv"

DATA_RSIDM_C_BASED = ROOT_DIR / "Data" / "rSIDM" / "Base-C"
DATA_RSIDM_T_BASED = ROOT_DIR / "Data" / "rSIDM" / "Base-T"

# Corrected effective constants for rSIDM, used only in tau_kappa.
# Order: log10(M/Msun) = 7.0, 7.5, 8.0, 8.5, 9.0
C_BASED_RSIDM_SIGMA_EFF = np.array([30.545, 46.855, 28.378, 14.986, 9.190])
T_BASED_RSIDM_SIGMA_EFF = np.array([28.482, 46.571, 28.334, 15.097, 9.137])

if SIGMA_1D_FACTOR == 1.05:
    DATA_CSIDM_C_BASED = ROOT_DIR / "Data" / "cSIDM" / "Luca-C-Base"
    DATA_CSIDM_T_BASED = ROOT_DIR / "Data" / "cSIDM" / "Luca-T-Base"

    C_BASED_CONCENTRATIONS = np.array([21.196, 19.816, 18.436, 17.056, 15.676])
    C_BASED_CSIDM_SIGMA_EFF = np.array([26.995, 47.414, 30.709, 16.003, 9.7091])
    C_BASED_CSIDM_FILES = [
        "CSIDM_Gilman_M7.0_c21.196_sigma26.995_beta0.85.csv",
        "CSIDM_Gilman_M7.5_c19.816_sigma47.414_beta0.85.csv",
        "CSIDM_Gilman_M8.0_c18.436_sigma30.709_beta0.85.csv",
        "CSIDM_Gilman_M8.5_c17.056_sigma16.003_beta0.85.csv",
        "CSIDM_Gilman_M9.0_c15.676_sigma9.7091_beta0.85.csv",
    ]

    T_BASED_CONCENTRATIONS = np.array([21.21, 19.81, 18.42, 17.05, 15.69])
    T_BASED_CSIDM_SIGMA_EFF = np.array([27.010, 47.414, 30.723, 16.006, 9.7062])
    T_BASED_CSIDM_FILES = [
        "CSIDM_Gilman_M7.0_c21.196_sigma26.995_beta0.85.csv",
        "CSIDM_Gilman_M7.5_c19.816_sigma47.414_beta0.85.csv",
        "CSIDM_Gilman_M8.0_c18.436_sigma30.709_beta0.85.csv",
        "CSIDM_Gilman_M8.5_c17.056_sigma16.003_beta0.85.csv",
        "CSIDM_Gilman_M9.0_c15.676_sigma9.7091_beta0.85.csv",
    ]

elif SIGMA_1D_FACTOR == 1.10:
    DATA_CSIDM_C_BASED = ROOT_DIR / "Data" / "cSIDM" / "Krzysztof-C-Base"
    DATA_CSIDM_T_BASED = ROOT_DIR / "Data" / "cSIDM" / "Krzysztof-T-Base"

    C_BASED_CONCENTRATIONS = np.array([21.21, 19.81, 18.42, 17.05, 15.69])
    C_BASED_CSIDM_SIGMA_EFF = np.array([30.545, 46.855, 28.378, 14.986, 9.190])
    C_BASED_CSIDM_FILES = [
        "CSIDM_Gilman_M7.0_c21.21_sigma30.545_beta0.85.csv",
        "CSIDM_Gilman_M7.5_c19.81_sigma46.855_beta0.85.csv",
        "CSIDM_Gilman_M8.0_c18.42_sigma28.378_beta0.85.csv",
        "CSIDM_Gilman_M8.5_c17.05_sigma14.986_beta0.85.csv",
        "CSIDM_Gilman_M9.0_c15.69_sigma9.19_beta0.85.csv",
    ]

    T_BASED_CONCENTRATIONS = np.array([21.21, 19.81, 18.42, 17.05, 15.69])
    T_BASED_CSIDM_SIGMA_EFF = np.array([28.482, 46.571, 28.334, 15.097, 9.137])
    T_BASED_CSIDM_FILES = [
        "CSIDM_Gilman_M7.0_c21.21_sigma28.482_beta0.85.csv",
        "CSIDM_Gilman_M7.5_c19.81_sigma46.571_beta0.85.csv",
        "CSIDM_Gilman_M8.0_c18.42_sigma28.334_beta0.85.csv",
        "CSIDM_Gilman_M8.5_c17.05_sigma15.097_beta0.85.csv",
        "CSIDM_Gilman_M9.0_c15.69_sigma9.137_beta0.85.csv",
    ]
else:
    raise ValueError("SIGMA_1D_FACTOR must be 1.05 or 1.10.")

MASSES_LOG10 = np.array([7.0, 7.5, 8.0, 8.5, 9.0])
T_BASED_R200_KPC = np.array([4.55, 6.68, 9.81, 14.4, 21.1])
T_BASED_RS_KPC = np.array([0.220, 0.340, 0.530, 0.840, 1.350])
T_BASED_LOG10_RHO_S = np.array([7.57, 7.50, 7.42, 7.33, 7.24])

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


@dataclass(frozen=True)
class HaloRun:
    log10_mass: float
    concentration: float
    csidm_sigma_eff: float
    rsidm_sigma_eff: float
    csidm_file: str
    rsidm_file: str
    r200_kpc: float | None = None
    rs_kpc: float | None = None
    log10_rho_s: float | None = None

    @property
    def mass_msun(self) -> float:
        return 10.0**self.log10_mass


@dataclass
class LoadedPair:
    metadata: HaloRun
    csidm: object
    rsidm: object
    csidm_tau_kappa: float
    rsidm_tau_kappa: float


def validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def validate_configuration() -> None:
    expected = len(MASSES_LOG10)
    values = {
        "C_BASED_CONCENTRATIONS": C_BASED_CONCENTRATIONS,
        "T_BASED_CONCENTRATIONS": T_BASED_CONCENTRATIONS,
        "C_BASED_CSIDM_SIGMA_EFF": C_BASED_CSIDM_SIGMA_EFF,
        "T_BASED_CSIDM_SIGMA_EFF": T_BASED_CSIDM_SIGMA_EFF,
        "C_BASED_RSIDM_SIGMA_EFF": C_BASED_RSIDM_SIGMA_EFF,
        "T_BASED_RSIDM_SIGMA_EFF": T_BASED_RSIDM_SIGMA_EFF,
        "C_BASED_CSIDM_FILES": C_BASED_CSIDM_FILES,
        "T_BASED_CSIDM_FILES": T_BASED_CSIDM_FILES,
        "C_BASED_RSIDM_FILES": C_BASED_RSIDM_FILES,
        "T_BASED_RSIDM_FILES": T_BASED_RSIDM_FILES,
    }
    for name, array in values.items():
        if len(array) != expected:
            raise ValueError(f"{name} has length {len(array)}, expected {expected}.")

    if np.any(C_BASED_RSIDM_SIGMA_EFF <= 0.0):
        raise ValueError("Insert positive values in C_BASED_RSIDM_SIGMA_EFF.")
    if np.any(T_BASED_RSIDM_SIGMA_EFF <= 0.0):
        raise ValueError("Insert positive values in T_BASED_RSIDM_SIGMA_EFF.")


class TabulatedCrossSection:
    """Log-log interpolation of sigma(v_rel)/m from a two-column CSV file."""

    def __init__(self, filename: Path):
        validate_file(filename)
        data = np.loadtxt(filename, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{filename} must contain at least two columns.")

        velocity = np.asarray(data[:, 0], dtype=float)
        sigma = np.asarray(data[:, 1], dtype=float)
        mask = (
            np.isfinite(velocity)
            & np.isfinite(sigma)
            & (velocity > 0.0)
            & (sigma > 0.0)
        )
        velocity = velocity[mask]
        sigma = sigma[mask]
        order = np.argsort(velocity)
        self.velocity_kms = velocity[order]
        self.sigma_cm2_g = sigma[order]

        self._interpolator = interp1d(
            np.log(self.velocity_kms),
            np.log(self.sigma_cm2_g),
            kind="linear",
            bounds_error=False,
            fill_value=(
                np.log(self.sigma_cm2_g[0]),
                np.log(self.sigma_cm2_g[-1]),
            ),
        )

    def __call__(self, velocity_kms):
        velocity = np.asarray(velocity_kms, dtype=float)
        velocity = np.maximum(velocity, 1e-300)
        result = np.exp(self._interpolator(np.log(velocity)))
        return float(result) if result.ndim == 0 else result


def sigma_eff_k5_single(nu_kms: float, sigma_of_v) -> float:
    if nu_kms <= 0.0:
        raise ValueError("nu_kms must be positive.")

    x = np.logspace(np.log10(K5_X_MIN), np.log10(K5_X_MAX), K5_X_POINTS)
    v_rel_kms = nu_kms * x
    sigma_values = sigma_of_v(v_rel_kms)
    weight = x ** (K5_P + 2) * np.exp(-x**2 / 4.0)
    numerator = np.trapz(sigma_values * weight, x)
    denominator = np.trapz(weight, x)
    return float(numerator / denominator)


def build_gilman_k5_interpolator():
    sigma_of_v = TabulatedCrossSection(GILMAN_CROSS_SECTION_FILE)
    nu_grid = np.logspace(
        np.log10(K5_NU_MIN_KMS),
        np.log10(K5_NU_MAX_KMS),
        K5_NU_POINTS,
    )
    k5_grid = np.array(
        [sigma_eff_k5_single(nu, sigma_of_v) for nu in nu_grid],
        dtype=float,
    )
    interpolator = interp1d(
        np.log(nu_grid),
        np.log(k5_grid),
        kind="linear",
        bounds_error=False,
        fill_value=(np.log(k5_grid[0]), np.log(k5_grid[-1])),
    )

    def k5_from_nu_kms(nu_kms):
        nu = np.asarray(nu_kms, dtype=float)
        nu = np.maximum(nu, 1e-300)
        result = np.exp(interpolator(np.log(nu)))
        return float(result) if result.ndim == 0 else result

    return k5_from_nu_kms


GILMAN_K5_FROM_NU_KMS = build_gilman_k5_interpolator()


def build_run_table() -> dict[Basis, list[HaloRun]]:
    c_based = []
    for mass, concentration, csidm_sigma, rsidm_sigma, csidm_file, rsidm_file in zip(
        MASSES_LOG10,
        C_BASED_CONCENTRATIONS,
        C_BASED_CSIDM_SIGMA_EFF,
        C_BASED_RSIDM_SIGMA_EFF,
        C_BASED_CSIDM_FILES,
        C_BASED_RSIDM_FILES,
    ):
        c_based.append(
            HaloRun(
                log10_mass=float(mass),
                concentration=float(concentration),
                csidm_sigma_eff=float(csidm_sigma),
                rsidm_sigma_eff=float(rsidm_sigma),
                csidm_file=csidm_file,
                rsidm_file=rsidm_file,
            )
        )

    t_based = []
    for mass, concentration, csidm_sigma, rsidm_sigma, csidm_file, rsidm_file, r200, rs, log_rho_s in zip(
        MASSES_LOG10,
        T_BASED_CONCENTRATIONS,
        T_BASED_CSIDM_SIGMA_EFF,
        T_BASED_RSIDM_SIGMA_EFF,
        T_BASED_CSIDM_FILES,
        T_BASED_RSIDM_FILES,
        T_BASED_R200_KPC,
        T_BASED_RS_KPC,
        T_BASED_LOG10_RHO_S,
    ):
        t_based.append(
            HaloRun(
                log10_mass=float(mass),
                concentration=float(concentration),
                csidm_sigma_eff=float(csidm_sigma),
                rsidm_sigma_eff=float(rsidm_sigma),
                csidm_file=csidm_file,
                rsidm_file=rsidm_file,
                r200_kpc=float(r200),
                rs_kpc=float(rs),
                log10_rho_s=float(log_rho_s),
            )
        )

    return {"C-based": c_based, "T-based": t_based}


def load_gravothermal_run(filename: str, directory: Path, mass_msun: float, sigma_eff: float):
    validate_file(directory / filename)
    model = gravoF.create_gravothermalData_from_file(filename, str(directory), beta=BETA)
    model.put_the_name("NFW")
    model.put_extra_parameters(mass_msun, sigma_eff)
    return model


def tau_kappa_t_based(sigma_eff: float, rs_kpc: float, log10_rho_s: float) -> float:
    rho_s = 10.0**log10_rho_s
    sigma_m_si = sigma_eff * 1e-4 * 1e3
    sigma_m_star = sigma_m_si * cfg.kpc_SI**-2 * cfg.M_solar_SI
    g_star = cfg.const_G_starUnits
    tau = (
        (150.0 / BETA)
        * (1.0 / (rs_kpc * rho_s))
        * (1.0 / sigma_m_star)
        * (4.0 * np.pi * g_star * rho_s) ** (-0.5)
    )
    return float(tau)


def calculate_tau_kappa(run: HaloRun, basis: Basis, sigma_eff: float) -> float:
    if basis == "C-based":
        halo = TruncatedNFWProfile(_M_vir=run.mass_msun, _con=run.concentration)
        return float(halo.tau(beta=BETA, sigma_eff=sigma_eff))

    if basis == "T-based":
        if run.rs_kpc is None or run.log10_rho_s is None:
            raise ValueError("T-based run requires rs_kpc and log10_rho_s.")
        return tau_kappa_t_based(sigma_eff, run.rs_kpc, run.log10_rho_s)

    raise ValueError(f"Unknown basis: {basis}")


def load_pair(run: HaloRun, basis: Basis) -> LoadedPair:
    if basis == "C-based":
        csidm_directory = DATA_CSIDM_C_BASED
        rsidm_directory = DATA_RSIDM_C_BASED
    elif basis == "T-based":
        csidm_directory = DATA_CSIDM_T_BASED
        rsidm_directory = DATA_RSIDM_T_BASED
    else:
        raise ValueError(f"Unknown basis: {basis}")

    csidm = load_gravothermal_run(
        run.csidm_file,
        csidm_directory,
        run.mass_msun,
        run.csidm_sigma_eff,
    )
    rsidm = load_gravothermal_run(
        run.rsidm_file,
        rsidm_directory,
        run.mass_msun,
        run.rsidm_sigma_eff,
    )

    return LoadedPair(
        metadata=run,
        csidm=csidm,
        rsidm=rsidm,
        csidm_tau_kappa=calculate_tau_kappa(run, basis, run.csidm_sigma_eff),
        rsidm_tau_kappa=calculate_tau_kappa(run, basis, run.rsidm_sigma_eff),
    )


def get_knudsen_evolution(pair: LoadedPair):
    kn_csidm = pair.csidm.return_knudsen_core_evolution(
        sigma_m=pair.metadata.csidm_sigma_eff,
        elements_rho=ELEMENTS_RHO,
        elements_vel=ELEMENTS_VEL,
        use_core_average=USE_CORE_AVERAGE,
        stop_at_collapse=STOP_AT_COLLAPSE,
    )

    kn_rsidm = pair.rsidm.return_knudsen_core_evolution(
        sigma_m=GILMAN_K5_FROM_NU_KMS,
        elements_rho=ELEMENTS_RHO,
        elements_vel=ELEMENTS_VEL,
        sigma_input_vel_unit="km/s",
        use_core_average=USE_CORE_AVERAGE,
        stop_at_collapse=STOP_AT_COLLAPSE,
    )

    return {"cSIDM": kn_csidm, "rSIDM": kn_rsidm}


def apply_scale_grid(ax, xscale: str, yscale: str) -> None:
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(which="major", alpha=0.4)
    ax.grid(which="minor", alpha=0.2)

    if xscale == "log":
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if yscale == "log":
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.tick_params("both", direction="in", top=True, right=True, length=10, width=1, which="major")
    ax.tick_params("both", direction="in", top=True, right=True, length=5, width=1, which="minor")


def plot_knudsen_c_vs_t_for_mass(mass_index: int) -> None:
    if not 0 <= mass_index < len(MASSES_LOG10):
        raise IndexError(f"mass_index must be between 0 and {len(MASSES_LOG10) - 1}.")

    runs = build_run_table()
    pair_c = load_pair(runs["C-based"][mass_index], "C-based")
    pair_t = load_pair(runs["T-based"][mass_index], "T-based")

    kn_c = get_knudsen_evolution(pair_c)
    kn_t = get_knudsen_evolution(pair_t)

    t_csidm_c = np.asarray(kn_c["cSIDM"]["time"], dtype=float)
    y_csidm_c = np.asarray(kn_c["cSIDM"]["Kn_core"], dtype=float)
    t_rsidm_c = np.asarray(kn_c["rSIDM"]["time"], dtype=float)
    y_rsidm_c = np.asarray(kn_c["rSIDM"]["Kn_core"], dtype=float)

    t_csidm_t = np.asarray(kn_t["cSIDM"]["time"], dtype=float)
    y_csidm_t = np.asarray(kn_t["cSIDM"]["Kn_core"], dtype=float)
    t_rsidm_t = np.asarray(kn_t["rSIDM"]["time"], dtype=float)
    y_rsidm_t = np.asarray(kn_t["rSIDM"]["Kn_core"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.4), sharey=True)
    ax_left, ax_right = axes

    color_c = "tab:red"
    color_t = "tab:blue"

    curves = [
        (t_csidm_c, y_csidm_c, "--", color_c, "cSIDM (C-based)"),
        (t_rsidm_c, y_rsidm_c, "-", color_c, "rSIDM (C-based)"),
        (t_csidm_t, y_csidm_t, "--", color_t, "cSIDM (T-based)"),
        (t_rsidm_t, y_rsidm_t, "-", color_t, "rSIDM (T-based)"),
    ]
    for time, kn, linestyle, color, label in curves:
        ax_left.plot(time, kn, linestyle=linestyle, linewidth=2.3, color=color, label=label)

    normalized_curves = [
        (t_csidm_c / pair_c.csidm_tau_kappa, y_csidm_c, "--", color_c, "cSIDM (C-based)"),
        (t_rsidm_c / pair_c.rsidm_tau_kappa, y_rsidm_c, "-", color_c, "rSIDM (C-based)"),
        (t_csidm_t / pair_t.csidm_tau_kappa, y_csidm_t, "--", color_t, "cSIDM (T-based)"),
        (t_rsidm_t / pair_t.rsidm_tau_kappa, y_rsidm_t, "-", color_t, "rSIDM (T-based)"),
    ]
    for time, kn, linestyle, color, label in normalized_curves:
        ax_right.plot(time, kn, linestyle=linestyle, linewidth=2.3, color=color, label=label)

    for ax in axes:
        ax.axhline(1.0, linestyle=":", linewidth=1.8, color="gray", alpha=0.9, label=r"$\mathrm{Kn}_{\rm core}=1$")
        apply_scale_grid(ax, X_SCALE, Y_SCALE)
        ax.tick_params(labelsize=12)

    ax_left.set_xlabel(r"$T\,[\mathrm{Gyr}]$", fontsize=16)
    ax_right.set_xlabel(r"$t=T/\tau_\kappa$", fontsize=16)
    ax_left.set_ylabel(r"$\mathrm{Kn}_{\rm core}$", fontsize=16)
    ax_left.set_title("Physical time", fontsize=14)
    ax_right.set_title(r"Time normalized by $\tau_\kappa$", fontsize=14)

    log_mass = pair_c.metadata.log10_mass
    fig.suptitle(
        rf"Core Knudsen-number evolution: $M=10^{{{log_mass:.1f}}}\,M_\odot$",
        fontsize=17,
        y=0.98,
    )

    handles, labels = ax_left.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax_left.legend(unique.values(), unique.keys(), fontsize=10.5, frameon=True, framealpha=0.92, loc="best")

    annotation = "\n".join([
        rf"C-based: $c={pair_c.metadata.concentration:.3f}$",
        rf"C-based: $\sigma_{{\rm eff}}^r={pair_c.metadata.rsidm_sigma_eff:.4g}$",
        rf"T-based: $c={pair_t.metadata.concentration:.3f}$",
        rf"T-based: $\sigma_{{\rm eff}}^r={pair_t.metadata.rsidm_sigma_eff:.4g}$",
        rf"$\sigma_{{1D}}={SIGMA_1D_FACTOR:.2f}\,\sigma_s$",
    ])
    ax_right.text(
        0.98,
        0.97,
        annotation,
        transform=ax_right.transAxes,
        ha="right",
        va="top",
        fontsize=10.3,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if PLOT_SAVE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"knudsen_C_vs_T_M{log_mass:.1f}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def main() -> None:
    validate_configuration()

    print("=" * 72)
    print("Gilman benchmark: core Knudsen-number evolution")
    print(f"beta                = {BETA}")
    print(f"sigma_1D convention = {SIGMA_1D_FACTOR:.2f} * sigma_s")
    print(f"cross-section table  = {GILMAN_CROSS_SECTION_FILE}")
    print("=" * 72)

    for mass_index in MASS_INDICES_TO_PLOT:
        plot_knudsen_c_vs_t_for_mass(mass_index)


if __name__ == "__main__":
    main()
