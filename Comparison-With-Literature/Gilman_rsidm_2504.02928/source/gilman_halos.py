"""
gilman_halos.py

Halo utilities for reproducing the Gilman/Tran benchmark.

This module keeps halo construction and velocity-scale choices separate from
cross-section profiles and averaging.

It expects your existing TruncatedNFWProfile.py to be importable from the same
working directory or from PYTHONPATH.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import numpy as np
try:
    from TruncatedNFWProfile import TruncatedNFWProfile
except ImportError:  # allows docs/imports without your full project path
    TruncatedNFWProfile = None

G_KPC_KMS2_MSUN = 4.300917270e-6

@dataclass(frozen=True)
class GilmanHaloTable:
    """Tabulated halo parameters used in the Gilman/Tran benchmark."""

    log10M: np.ndarray
    M200: np.ndarray
    c: np.ndarray
    r200_kpc: np.ndarray
    log10rho_s: np.ndarray

    @property
    def rs_kpc(self) -> np.ndarray:
        return self.r200_kpc / self.c

    @property
    def rho_s(self) -> np.ndarray:
        return 10.0 ** self.log10rho_s


GILMAN_TABLE = GilmanHaloTable(
    log10M=np.array([7.0, 7.5, 8.0, 8.5, 9.0]),
    M200=np.array([1e7, 10**7.5, 1e8, 10**8.5, 1e9]),
    c=np.array([21.21, 19.81, 18.42, 17.05, 15.69]),
    r200_kpc=np.array([4.55, 6.68, 9.81, 14.4, 21.1]),
    log10rho_s=np.array([7.57, 7.50, 7.42, 7.33, 7.24]),
)


def nfw_mass_factor(c):
    """f(c)=ln(1+c)-c/(1+c)."""
    c = np.asarray(c, dtype=float)
    return np.log(1.0 + c) - c / (1.0 + c)


def compute_rho_s_from_mcr200(M200, c, r200_kpc):
    """Compute NFW rho_s from M200, concentration and r200."""
    rs = r200_kpc / c
    return M200 / (4.0 * np.pi * rs**3 * nfw_mass_factor(c))


def extrapolate_gilman_table(log10M_new: Iterable[float], table: GilmanHaloTable = GILMAN_TABLE) -> GilmanHaloTable:
    """
    Extrapolate Gilman c(M), r200(M), rho_s(M).

    Procedure:
    - linear fit c vs log10(M),
    - r200 proportional to M^(1/3),
    - rho_s from NFW consistency.
    """
    log10M_new = np.asarray(log10M_new, dtype=float)
    M_new = 10.0 ** log10M_new

    a_c, b_c = np.polyfit(table.log10M, table.c, deg=1)
    c_new = a_c * log10M_new + b_c

    M_ref = table.M200[0]
    r200_ref = table.r200_kpc[0]
    r200_new = r200_ref * (M_new / M_ref) ** (1.0 / 3.0)

    rho_s_new = compute_rho_s_from_mcr200(M_new, c_new, r200_new)
    log10rho_s_new = np.log10(rho_s_new)

    return GilmanHaloTable(
        log10M=log10M_new,
        M200=M_new,
        c=c_new,
        r200_kpc=r200_new,
        log10rho_s=log10rho_s_new,
    )


def make_gilman_halos(
    table: GilmanHaloTable = GILMAN_TABLE,
    explicit: bool = True,
    r_d: Optional[float] = None,
):
    """
    Construct TruncatedNFWProfile objects from a GilmanHaloTable.

    explicit=True uses table r200 and log10rho_s directly. This is recommended
    for reproducing Gilman benchmark values.

    explicit=False uses only M200 and c, letting TruncatedNFWProfile compute
    r200 and rho_s internally from the project's cosmology/config.
    """
    if TruncatedNFWProfile is None:
        raise ImportError("Could not import TruncatedNFWProfile. Put it on PYTHONPATH.")

    halos = []
    for M, c, r200, logrho in zip(table.M200, table.c, table.r200_kpc, table.log10rho_s):
        if explicit:
            halos.append(
                TruncatedNFWProfile(
                    _M_vir=float(M),
                    _con=float(c),
                    _r200=float(r200),
                    _log_rho_s=float(logrho),
                    _r_d=r_d,
                )
            )
        else:
            halos.append(
                TruncatedNFWProfile(
                    _M_vir=float(M),
                    _con=float(c),
                    _r_d=r_d,
                )
            )
    return halos


def get_rs_kpc(halo) -> float:
    """Robustly infer r_s from common halo attribute names."""
    for attr in ["r_s", "rs", "R_s", "Rs", "_r_s", "_rs"]:
        if hasattr(halo, attr):
            return float(getattr(halo, attr))

    for r_attr in ["r200", "r_vir", "Rvir", "R_vir", "_Rvir", "_R_vir"]:
        for c_attr in ["con", "_con", "c", "_c"]:
            if hasattr(halo, r_attr) and hasattr(halo, c_attr):
                return float(getattr(halo, r_attr)) / float(getattr(halo, c_attr))

    raise AttributeError("Cannot infer r_s from halo.")


def sigma1D_gilman_analytic(halo) -> float:
    """
    Gilman-style characteristic 1D dispersion:
        sigma_1D = 1.10 sqrt(G rho_s r_s^2)
    Output: km/s.
    """
    rs = get_rs_kpc(halo)
    rho_s = float(getattr(halo, "rho_s"))
    return float(1.10 * np.sqrt(G_KPC_KMS2_MSUN * rho_s * rs**2))


def velocity_scale_for_halo(halo, mode: str = "gilman_analytic", r_kpc: Optional[float] = None) -> float:
    """
    Return the velocity scale nu [km/s] used in the Maxwellian average.

    Modes:
    - gilman_analytic: 1.10 sqrt(G rho_s r_s^2).
    - halo_sigma_rs  : halo.sigma(r_s) converted to km/s if config is available.
    - halo_sigma_r   : halo.sigma(r_kpc) converted to km/s.
    - halo_sigma_accurate_rs: halo.sigma_accurate(r_s), converted if config is available.
    """
    if mode == "gilman_analytic":
        return sigma1D_gilman_analytic(halo)

    try:
        import config as cfg
    except ImportError as exc:
        raise ImportError("velocity mode requires project config.py for kpc/Gyr -> km/s conversion") from exc

    if mode == "halo_sigma_rs":
        r = get_rs_kpc(halo)
        return float(halo.sigma(r) * cfg.kpcGyr_to_kms)

    if mode == "halo_sigma_r":
        if r_kpc is None:
            raise ValueError("r_kpc must be provided for mode='halo_sigma_r'")
        return float(halo.sigma(float(r_kpc)) * cfg.kpcGyr_to_kms)

    if mode == "halo_sigma_accurate_rs":
        r = get_rs_kpc(halo)
        return float(halo.sigma_accurate(r) * cfg.kpcGyr_to_kms)

    raise ValueError(f"Unknown velocity-scale mode: {mode}")
