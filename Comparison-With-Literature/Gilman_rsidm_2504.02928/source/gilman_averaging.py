"""
gilman_averaging.py

Maxwellian averaging utilities for velocity-dependent cross-sections.

This module is deliberately generic: it can average any callable profile

    sigma_profile(v_rel_kms) -> sigma/m [cm^2/g]

regardless of whether that profile comes from a table, a partial-wave solver,
or a future rSIDM model class.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.integrate import simpson


@dataclass(frozen=True)
class MaxwellianAverageResult:
    nu_kms: float
    p: float
    sigma_eff_cm2_g: float
    v_peak_kms: float


class MaxwellianAverager:
    """
    Compute K_p averages:

        sigma_eff^(p)(nu) = < sigma(v_rel) v_rel^p > / < v_rel^p >

    using the relative Maxwellian speed distribution. Constants cancel, so the
    effective integration weight is

        w(v;nu,p) = v^(p+2) exp[-v^2/(4 nu^2)].

    p=5 corresponds to the heat-conduction/kappa average used in the Gilman
    benchmark. p=1 is useful for collision-rate-like checks.
    """

    def __init__(
        self,
        p: float = 5.0,
        vmin: float = 1e-3,
        vmax: float = 1e4,
        n_grid: int = 10000,
        integration: str = "simpson",
    ):
        self.p = float(p)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.n_grid = int(n_grid)
        self.integration = integration

    def velocity_grid(self) -> np.ndarray:
        return np.logspace(np.log10(self.vmin), np.log10(self.vmax), self.n_grid)

    def weight(self, v_rel_kms, nu_kms: float) -> np.ndarray:
        v = np.asarray(v_rel_kms, dtype=float)
        return v ** (self.p + 2.0) * np.exp(-v**2 / (4.0 * nu_kms**2))

    def v_peak(self, nu_kms: float) -> float:
        """Peak of w(v) = v^(p+2) exp[-v^2/(4nu^2)]."""
        return float(np.sqrt(2.0 * (self.p + 2.0)) * nu_kms)

    def average(self, sigma_profile: Callable, nu_kms: float) -> float:
        if nu_kms <= 0:
            return np.nan
        v = self.velocity_grid()
        sig = np.asarray(sigma_profile(v), dtype=float)
        w = self.weight(v, nu_kms)

        if self.integration == "simpson":
            num = simpson(sig * w, x=v)
            den = simpson(w, x=v)
        elif self.integration == "trapz":
            num = np.trapz(sig * w, x=v)
            den = np.trapz(w, x=v)
        else:
            raise ValueError("integration must be 'simpson' or 'trapz'")

        return float(num / den)

    def average_with_metadata(self, sigma_profile: Callable, nu_kms: float) -> MaxwellianAverageResult:
        return MaxwellianAverageResult(
            nu_kms=float(nu_kms),
            p=float(self.p),
            sigma_eff_cm2_g=self.average(sigma_profile, nu_kms),
            v_peak_kms=self.v_peak(nu_kms),
        )

    def point_proxy(self, sigma_profile: Callable, nu_kms: float) -> float:
        """
        Single-velocity proxy sigma(v_peak). Useful for intuition only.
        """
        return float(sigma_profile(self.v_peak(nu_kms)))
