"""
gilman_benchmark.py

High-level helpers for the Gilman/Tran benchmark.

This module combines:
- halo construction from gilman_halos.py,
- raw cross-section profiles from gilman_cross_sections.py,
- Maxwellian averaging from gilman_averaging.py.

The design is intentionally modular: you can swap a tabulated profile for a
first-principles Yukawa profile without changing the averaging or halo code.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from gilman_averaging import MaxwellianAverager
from gilman_halos import GILMAN_TABLE, make_gilman_halos, velocity_scale_for_halo, get_rs_kpc


@dataclass(frozen=True)
class GilmanBenchmarkRow:
    log10M: float
    M200: float
    c: float
    r_s_kpc: float
    log10rho_s: float
    nu_kms: float
    v_peak_kms: float
    sigma_eff_cm2_g: float
    sigma_point_cm2_g: float
    tau_gyr: Optional[float] = None


def compute_gilman_rows(
    halos,
    sigma_profile: Callable,
    averager: Optional[MaxwellianAverager] = None,
    velocity_mode: str = "gilman_analytic",
    beta_for_tau: Optional[float] = 0.85,
) -> List[GilmanBenchmarkRow]:
    """
    Compute sigma_eff for a list of halos.

    Parameters
    ----------
    halos
        List of halo objects, typically TruncatedNFWProfile.
    sigma_profile
        Callable sigma(v_rel_kms)->cm^2/g.
    averager
        MaxwellianAverager. Default is p=5, suitable for kappa.
    velocity_mode
        See gilman_halos.velocity_scale_for_halo.
    beta_for_tau
        If not None and halo has tau(), compute tau(beta, sigma_eff).
    """
    if averager is None:
        averager = MaxwellianAverager(p=5, vmin=1e-2, vmax=1e4, n_grid=12000)

    rows: List[GilmanBenchmarkRow] = []

    for halo in halos:
        nu = velocity_scale_for_halo(halo, mode=velocity_mode)
        sigma_eff = averager.average(sigma_profile, nu)
        sigma_point = averager.point_proxy(sigma_profile, nu)

        tau = None
        if beta_for_tau is not None and hasattr(halo, "tau"):
            tau = float(halo.tau(beta=beta_for_tau, sigma_eff=sigma_eff))

        M_input = float(getattr(halo, "M_vir_input", getattr(halo, "M_vir", np.nan)))
        c = float(getattr(halo, "con", np.nan))
        rs = get_rs_kpc(halo)
        rho_s = float(getattr(halo, "rho_s", np.nan))

        rows.append(
            GilmanBenchmarkRow(
                log10M=float(np.log10(M_input)),
                M200=M_input,
                c=c,
                r_s_kpc=rs,
                log10rho_s=float(np.log10(rho_s)),
                nu_kms=float(nu),
                v_peak_kms=averager.v_peak(nu),
                sigma_eff_cm2_g=float(sigma_eff),
                sigma_point_cm2_g=float(sigma_point),
                tau_gyr=tau,
            )
        )

    return rows


def rows_to_array(rows: List[GilmanBenchmarkRow], key: str) -> np.ndarray:
    return np.asarray([getattr(r, key) for r in rows], dtype=float)


def print_gilman_rows(rows: List[GilmanBenchmarkRow], title: str = "Gilman benchmark rows") -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'log10M':>8} {'c':>8} {'r_s[kpc]':>10} {'logrho_s':>10} "
        f"{'nu[km/s]':>10} {'v_peak':>10} {'K5':>12} {'point':>12} {'tau[Gyr]':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        tau_str = "nan" if r.tau_gyr is None else f"{r.tau_gyr:.5g}"
        print(
            f"{r.log10M:8.3f} {r.c:8.3f} {r.r_s_kpc:10.4g} {r.log10rho_s:10.3f} "
            f"{r.nu_kms:10.4f} {r.v_peak_kms:10.4f} {r.sigma_eff_cm2_g:12.5g} "
            f"{r.sigma_point_cm2_g:12.5g} {tau_str:>12}"
        )


def plot_raw_profiles(profiles: Dict[str, Callable], v_grid=None, title="Raw sigma(v) profiles"):
    if v_grid is None:
        v_grid = np.logspace(0, np.log10(1000), 400)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, profile in profiles.items():
        ax.loglog(v_grid, profile(v_grid), lw=2.0, label=name)

    ax.set_xlabel(r"$v_{\rm rel}$ [km/s]")
    ax.set_ylabel(r"$\sigma_V/m_\chi$ [cm$^2$/g]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_sigma_eff_vs_mass(rows_by_name: Dict[str, List[GilmanBenchmarkRow]], title="Effective K5 vs halo mass"):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, rows in rows_by_name.items():
        M = rows_to_array(rows, "M200")
        K5 = rows_to_array(rows, "sigma_eff_cm2_g")
        ax.loglog(M, K5, marker="o", lw=2.0, label=name)

    ax.set_xlabel(r"$M_{200}$ [$M_\odot$]")
    ax.set_ylabel(r"$\sigma_\kappa/m$ [cm$^2$/g]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def make_default_gilman_halos(explicit: bool = True):
    """Convenience wrapper for the five benchmark halos."""
    return make_gilman_halos(GILMAN_TABLE, explicit=explicit)
