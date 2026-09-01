"""
gilman_cross_sections.py

Clean, profile-level tools for the Gilman/Tran RSIDM benchmark.

This module only defines raw velocity-dependent cross-section profiles:

    sigma(v_rel) / m_chi   [cm^2/g]

It intentionally does not know anything about halo parameters, kappa averaging,
or gravothermal times. Those are handled in separate modules.

Main classes
------------
TabulatedCrossSection
    Reads a tabulated/digitized two-column profile v_rel[km/s], sigma/m[cm^2/g].

YukawaPartialWaveCrossSection
    Computes sigma_V/m from the attractive Yukawa potential using the variable
    phase equation and the viscosity cross section.

Recommended Gilman single-peak working variants
-----------------------------------------------
- "single_paper"             : values printed in the original paper.
- "single_email_rounded"     : corrected rounded values from Daniel's email.
- "single_profile_fit"       : useful reconstruction that matches collaborator tables.
- "multi_paper"              : values printed in the original paper.

All numerical defaults are chosen after the low-v convergence diagnostics:
    x_min = 1e-4,
    range_factor = 120,
    rtol = 1e-8,
    atol = 1e-10,
    method = DOP853.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
import csv
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, spherical_yn, factorial2

# -----------------------------------------------------------------------------
# Units
# -----------------------------------------------------------------------------

C_KM_S = 299792.458
GEV2_TO_CM2 = 0.389379e-27
GEV_TO_G = 1.78266192e-24


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class YukawaNumerics:
    """Numerical settings for the variable-phase integration."""

    x_min: float = 1e-4
    range_factor: float = 120.0
    rtol: float = 1e-8
    atol: float = 1e-10
    method: str = "DOP853"
    use_small_x_initial_condition: bool = True
    warn_on_solver_warning: bool = False


@dataclass(frozen=True)
class YukawaModel:
    """Parameters of the attractive Yukawa scattering model."""

    name: str
    mchi_GeV: float
    mphi_MeV: float
    alpha: float
    ell_sum_max: int


GILMAN_YUKAWA_MODELS: Dict[str, YukawaModel] = {
    "single_paper": YukawaModel(
        name="single_paper",
        mchi_GeV=31.8,
        mphi_MeV=5.7,
        alpha=0.00158,
        ell_sum_max=1,
    ),
    "single_email_rounded": YukawaModel(
        name="single_email_rounded",
        mchi_GeV=37.0,
        mphi_MeV=6.5,
        alpha=0.00158,
        ell_sum_max=1,
    ),
    "single_profile_fit": YukawaModel(
        name="single_profile_fit",
        mchi_GeV=37.0,
        mphi_MeV=6.56,
        alpha=0.00158,
        ell_sum_max=1,
    ),
    "multi_paper": YukawaModel(
        name="multi_paper",
        mchi_GeV=67.7,
        mphi_MeV=1.9,
        alpha=0.00155,
        ell_sum_max=6,
    ),
}


# -----------------------------------------------------------------------------
# Generic interface
# -----------------------------------------------------------------------------

class CrossSectionProfile:
    """
    Minimal interface for any raw cross-section profile.

    Subclasses must implement:
        profile(v_rel_kms) -> sigma/m [cm^2/g]

    v_rel_kms may be a scalar or a numpy array.
    """

    name: str

    def __call__(self, v_rel_kms):
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Tabulated profiles
# -----------------------------------------------------------------------------

class TabulatedCrossSection(CrossSectionProfile):
    """
    Log-log interpolated profile from two-column data.

    Expected columns:
        column 0: v_rel [km/s]
        column 1: sigma/m [cm^2/g]

    The class can be constructed directly from arrays or through from_csv().
    """

    def __init__(
        self,
        v_kms: Iterable[float],
        sigma_cm2_g: Iterable[float],
        name: str = "tabulated",
        log_interp: bool = True,
        extrapolation: str = "edge",
    ):
        self.name = name
        self.log_interp = log_interp
        self.extrapolation = extrapolation

        v = np.asarray(v_kms, dtype=float)
        s = np.asarray(sigma_cm2_g, dtype=float)

        mask = np.isfinite(v) & np.isfinite(s) & (v > 0.0) & (s > 0.0)
        if not np.any(mask):
            raise ValueError("No valid positive points in tabulated cross section.")

        v = v[mask]
        s = s[mask]
        order = np.argsort(v)
        v = v[order]
        s = s[order]

        # Remove duplicate velocities, keeping the first sorted occurrence.
        unique_v, unique_idx = np.unique(v, return_index=True)
        self.v_kms = unique_v
        self.sigma_cm2_g = s[unique_idx]

        self.v_min = float(self.v_kms[0])
        self.v_max = float(self.v_kms[-1])

        if log_interp:
            x = np.log(self.v_kms)
            y = np.log(self.sigma_cm2_g)
        else:
            x = self.v_kms
            y = self.sigma_cm2_g

        if extrapolation == "edge":
            fill_value = (float(y[0]), float(y[-1]))
        elif extrapolation == "extrapolate":
            fill_value = "extrapolate"
        else:
            raise ValueError("extrapolation must be 'edge' or 'extrapolate'")

        self._interp = interp1d(
            x,
            y,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

    @classmethod
    def from_csv(
        cls,
        filename: Union[str, Path],
        name: Optional[str] = None,
        delimiter: str = ",",
        log_interp: bool = True,
        extrapolation: str = "edge",
    ) -> "TabulatedCrossSection":
        """Read a robust two-column CSV/TXT profile."""
        filename = Path(filename)
        data = read_two_column_numeric_file(filename, delimiter=delimiter)
        return cls(
            data[:, 0],
            data[:, 1],
            name=name or filename.stem,
            log_interp=log_interp,
            extrapolation=extrapolation,
        )

    def __call__(self, v_rel_kms):
        v = np.asarray(v_rel_kms, dtype=float)
        v_safe = np.maximum(v, 1e-300)
        if self.log_interp:
            out = np.exp(self._interp(np.log(v_safe)))
        else:
            out = self._interp(v_safe)
        return out


def read_two_column_numeric_file(filename: Union[str, Path], delimiter: str = ",") -> np.ndarray:
    """
    Robust reader for digitized two-column files.

    Handles empty lines, headers, comments, trailing commas, and semicolon files
    with European decimal commas.
    """
    rows = []
    with open(filename, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            for comment in ["#", "//"]:
                if comment in line:
                    line = line.split(comment, 1)[0].strip()
            if not line:
                continue

            if ";" in line:
                parts = [p.strip().replace(",", ".") for p in line.split(";")]
            else:
                parts = [p.strip() for p in line.split(delimiter)]

            parts = [p for p in parts if p != ""]
            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue

            if np.isfinite(x) and np.isfinite(y):
                rows.append((x, y))

    if len(rows) == 0:
        raise ValueError(f"Could not read any numeric two-column rows from {filename}")
    return np.asarray(rows, dtype=float)

# -----------------------------------------------------------------------------
# Yukawa partial-wave profile
# -----------------------------------------------------------------------------

class YukawaPartialWaveCrossSection(CrossSectionProfile):
    """
    First-principles viscosity cross section for attractive Yukawa scattering.

    Uses the variable phase equation in x = k r and computes

        sigma_V = 4*pi/k^2 * sum_l [(l+1)(l+2)/(2l+3)]
                  sin^2(delta_{l+2} - delta_l)

    Returns sigma_V / m_chi in cm^2/g.
    """

    def __init__(self, model: YukawaModel, numerics: Optional[YukawaNumerics] = None):
        self.model = model
        self.numerics = numerics or YukawaNumerics()
        self.name = model.name

    @classmethod
    def from_gilman_model(
        cls,
        model_name: str,
        numerics: Optional[YukawaNumerics] = None,
        **overrides,
    ) -> "YukawaPartialWaveCrossSection":
        """
        Construct from a named Gilman model, optionally overriding parameters.

        Example:
            YukawaPartialWaveCrossSection.from_gilman_model(
                "single_email_rounded", mphi_MeV=6.56
            )
        """
        if model_name not in GILMAN_YUKAWA_MODELS:
            raise KeyError(f"Unknown model {model_name!r}. Available: {list(GILMAN_YUKAWA_MODELS)}")
        model = GILMAN_YUKAWA_MODELS[model_name]
        if overrides:
            model = replace(model, **overrides)
        return cls(model=model, numerics=numerics)

    @staticmethod
    def phase_rhs(x, delta, ell, alpha, mchi_GeV, mphi_GeV, v_dimless):
        d = delta[0]
        jl = spherical_jn(ell, x)
        yl = spherical_yn(ell, x)
        beta = 2.0 * mphi_GeV / (mchi_GeV * v_dimless)
        bracket = np.cos(d) * jl - np.sin(d) * yl
        return [(2.0 * alpha / v_dimless) * x * np.exp(-beta * x) * bracket**2]

    @staticmethod
    def small_x_delta0(ell: int, x_min: float, alpha: float, v_dimless: float) -> float:
        """
        Small-x initial phase estimate:

            delta_l(x_min) ≈ alpha/v * x_min^(2l+2)
                             / ((l+1) [(2l+1)!!]^2)
        """
        denom = (ell + 1.0) * factorial2(2 * ell + 1, exact=False) ** 2
        return float((alpha / v_dimless) * x_min ** (2 * ell + 2) / denom)

    def compute_delta_l(self, ell: int, v_km_s: float) -> float:
        m = self.model
        num = self.numerics

        v_dimless = v_km_s / C_KM_S
        mphi_GeV = m.mphi_MeV * 1e-3

        x_max = num.range_factor * m.mchi_GeV * v_dimless / (2.0 * mphi_GeV)
        x_max = max(float(x_max), 50.0)

        if num.use_small_x_initial_condition:
            delta0 = self.small_x_delta0(ell, num.x_min, m.alpha, v_dimless)
        else:
            delta0 = 0.0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sol = solve_ivp(
                self.phase_rhs,
                t_span=(num.x_min, x_max),
                y0=[delta0],
                args=(ell, m.alpha, m.mchi_GeV, mphi_GeV, v_dimless),
                method=num.method,
                rtol=num.rtol,
                atol=num.atol,
            )

        if num.warn_on_solver_warning and len(caught) > 0:
            print(f"Warnings for ell={ell}, v={v_km_s}: {[str(w.message) for w in caught[:3]]}")

        if not sol.success:
            raise RuntimeError(f"solve_ivp failed for ell={ell}, v={v_km_s}: {sol.message}")

        return float(sol.y[0, -1])

    def sigma_one(self, v_km_s: float) -> float:
        m = self.model
        v_dimless = v_km_s / C_KM_S
        k_GeV = 0.5 * m.mchi_GeV * v_dimless

        deltas = np.array([
            self.compute_delta_l(ell, v_km_s)
            for ell in range(m.ell_sum_max + 3)
        ])

        partial_sum = 0.0
        for ell in range(m.ell_sum_max + 1):
            coeff = (ell + 1.0) * (ell + 2.0) / (2.0 * ell + 3.0)
            partial_sum += coeff * np.sin(deltas[ell + 2] - deltas[ell]) ** 2

        sigma_GeV_minus2 = 4.0 * np.pi / k_GeV**2 * partial_sum
        return float(sigma_GeV_minus2 * GEV2_TO_CM2 / (m.mchi_GeV * GEV_TO_G))

    def __call__(self, v_rel_kms):
        v = np.asarray(v_rel_kms, dtype=float)
        if v.ndim == 0:
            return self.sigma_one(float(v))
        return np.array([self.sigma_one(float(vi)) for vi in v])


def evaluate_profile(profile: CrossSectionProfile, v_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate any profile on a velocity grid."""
    v = np.asarray(v_grid, dtype=float)
    return v, np.asarray(profile(v), dtype=float)
