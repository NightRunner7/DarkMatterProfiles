"""
gilman_benchmark_profiles.py

Clean profile-only benchmark helper for comparing:
  1) first-principles Yukawa/partial-wave reconstruction of Gilman sigma_V/m(v),
  2) one or more tabulated/digitized sigma(v) profiles from CSV files.

No sigma_kappa averaging is included here by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, spherical_yn
from scipy.special import factorial2

def delta_initial_small_x(ell, x_min, alpha, v_dimless):
    """
    Small-x initial phase shift estimate:
    delta_l(x_min) ≈ alpha/v * x_min^(2l+2) /
                    ((l+1) * [(2l+1)!!]^2)
    """
    denom = (ell + 1.0) * factorial2(2 * ell + 1, exact=False)**2
    return (alpha / v_dimless) * x_min**(2 * ell + 2) / denom


C_KM_S = 299792.458
GEV2_TO_CM2 = 0.389379e-27
GEV_TO_G = 1.78266192e-24


@dataclass(frozen=True)
class YukawaModel:
    name: str
    mchi_GeV: float
    mphi_MeV: float
    alpha: float
    ell_sum_max: int


class TabulatedProfile:
    """
    Log-log interpolated sigma(v)/m profile from a two-column CSV file:
        v_rel [km/s], sigma/m [cm^2/g]
    """

    def __init__(
        self,
        filename: str | Path,
        name: Optional[str] = None,
        delimiter: str = ",",
        log_interp: bool = True,
        extrapolation: str = "edge",
    ):
        self.filename = Path(filename)
        self.name = name or self.filename.stem
        self.log_interp = log_interp
        self.extrapolation = extrapolation

        data = np.loadtxt(self.filename, delimiter=delimiter)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{self.filename} must contain at least two columns: v, sigma")

        v = np.asarray(data[:, 0], dtype=float)
        s = np.asarray(data[:, 1], dtype=float)

        mask = np.isfinite(v) & np.isfinite(s) & (v > 0.0) & (s > 0.0)
        if not np.any(mask):
            raise ValueError(f"{self.filename} contains no valid positive points")

        v = v[mask]
        s = s[mask]

        order = np.argsort(v)
        self.v_kms = v[order]
        self.sigma_over_m = s[order]

        # Remove duplicate velocities, keeping the last value after sorting.
        unique_v, unique_idx = np.unique(self.v_kms, return_index=True)
        if unique_v.size != self.v_kms.size:
            self.v_kms = unique_v
            self.sigma_over_m = self.sigma_over_m[unique_idx]

        self.v_min = float(self.v_kms[0])
        self.v_max = float(self.v_kms[-1])

        if log_interp:
            x = np.log(self.v_kms)
            y = np.log(self.sigma_over_m)
        else:
            x = self.v_kms
            y = self.sigma_over_m

        if extrapolation == "edge":
            fill_value = (y[0], y[-1])
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

    def __call__(self, v_kms):
        v = np.asarray(v_kms, dtype=float)
        v_safe = np.maximum(v, 1e-300)

        if self.log_interp:
            out = np.exp(self._interp(np.log(v_safe)))
        else:
            out = self._interp(v_safe)

        return out


class GilmanBenchmarkProfiles:
    """
    Profile-only Gilman benchmark reconstruction.

    This class intentionally does not compute sigma_kappa/m. It only compares
    raw sigma_V/m(v_rel) profiles.
    """

    MODELS: Dict[str, YukawaModel] = {
        "single": YukawaModel(
            name="single",
            mchi_GeV=31.8,
            mphi_MeV=5.7,
            alpha=0.00158,
            ell_sum_max=8,
        ),
        "single_corrected": YukawaModel(
            name="single_corrected",
            mchi_GeV=37.0,
            mphi_MeV=6.56,
            alpha=0.00158,
            ell_sum_max=12,
        ),
        "multi": YukawaModel(
            name="multi",
            mchi_GeV=67.7,
            mphi_MeV=1.9,
            alpha=0.00155,
            ell_sum_max=6,
        ),
    }

    def __init__(
        self,
        v_grid: Optional[np.ndarray] = None,
        range_factor: float = 120.0,
        x_min: float = 1e-5,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ):
        self.v_grid = (
            np.asarray(v_grid, dtype=float)
            if v_grid is not None
            else np.logspace(np.log10(1.0), np.log10(300.0), 250)
        )
        self.range_factor = range_factor
        self.x_min = x_min
        self.rtol = rtol
        self.atol = atol

        self.generated: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.tabulated: Dict[str, TabulatedProfile] = {}

    @staticmethod
    def phase_rhs(x, delta, ell, alpha, mchi_GeV, mphi_GeV, v_dimless):
        d = delta[0]
        jl = spherical_jn(ell, x)
        yl = spherical_yn(ell, x)
        beta = 2.0 * mphi_GeV / (mchi_GeV * v_dimless)
        bracket = np.cos(d) * jl - np.sin(d) * yl
        return [(2.0 * alpha / v_dimless) * x * np.exp(-beta * x) * bracket**2]

    def compute_delta_l(self, ell: int, v_km_s: float, model: YukawaModel) -> float:
        v_dimless = v_km_s / C_KM_S
        mphi_GeV = model.mphi_MeV * 1e-3

        x_max = self.range_factor * model.mchi_GeV * v_dimless / (2.0 * mphi_GeV)
        x_max = max(x_max, 50.0)

        delta0 = delta_initial_small_x(
            ell=ell,
            x_min= self.x_min,
            alpha=model.alpha,
            v_dimless=v_dimless,
        )

        sol = solve_ivp(
            self.phase_rhs,
            t_span=(self.x_min, x_max),
            y0=[0.0],
            # y0=[delta0],
            args=(ell, model.alpha, model.mchi_GeV, mphi_GeV, v_dimless),
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return float(sol.y[0, -1])

    def sigma_V_over_m(self, v_km_s: float, model: YukawaModel) -> float:
        v_dimless = v_km_s / C_KM_S
        deltas = np.array([
            self.compute_delta_l(ell, v_km_s, model)
            for ell in range(model.ell_sum_max + 3)
        ])

        k_GeV = 0.5 * model.mchi_GeV * v_dimless

        partial_sum = 0.0
        for ell in range(model.ell_sum_max + 1):
            coeff = (ell + 1) * (ell + 2) / (2 * ell + 3)
            partial_sum += coeff * np.sin(deltas[ell + 2] - deltas[ell])**2

        sigma_GeV_minus2 = 4.0 * np.pi / k_GeV**2 * partial_sum
        return float(sigma_GeV_minus2 * GEV2_TO_CM2 / (model.mchi_GeV * GEV_TO_G))

    def compute_model(self, model_name: str = "single", v_grid: Optional[np.ndarray] = None):
        model = self.MODELS[model_name]
        v = self.v_grid if v_grid is None else np.asarray(v_grid, dtype=float)
        sigma = np.array([self.sigma_V_over_m(float(vi), model) for vi in v])
        self.generated[model_name] = (v, sigma)
        return v, sigma

    def add_generated_profile(self, name: str, v_kms, sigma_over_m):
        v = np.asarray(v_kms, dtype=float)
        s = np.asarray(sigma_over_m, dtype=float)
        mask = np.isfinite(v) & np.isfinite(s) & (v > 0.0) & (s > 0.0)
        order = np.argsort(v[mask])
        self.generated[name] = (v[mask][order], s[mask][order])

    def load_tabulated(
        self,
        filename: str | Path,
        name: Optional[str] = None,
        delimiter: str = ",",
        log_interp: bool = True,
        extrapolation: str = "edge",
    ):
        profile = TabulatedProfile(
            filename=filename,
            name=name,
            delimiter=delimiter,
            log_interp=log_interp,
            extrapolation=extrapolation,
        )
        self.tabulated[profile.name] = profile
        return profile

    @staticmethod
    def _interp_loglog(v_ref, sigma_ref):
        v_ref = np.asarray(v_ref, dtype=float)
        sigma_ref = np.asarray(sigma_ref, dtype=float)
        return interp1d(
            np.log(v_ref),
            np.log(sigma_ref),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

    def compare_to_tabulated(
        self,
        generated_name: str,
        tabulated_name: str,
        v_eval: Optional[np.ndarray] = None,
    ):
        if generated_name not in self.generated:
            raise KeyError(f"No generated profile named {generated_name!r}")
        if tabulated_name not in self.tabulated:
            raise KeyError(f"No tabulated profile named {tabulated_name!r}")

        v_gen, sig_gen = self.generated[generated_name]
        tab = self.tabulated[tabulated_name]

        if v_eval is None:
            vmin = max(np.min(v_gen), tab.v_min)
            vmax = min(np.max(v_gen), tab.v_max)
            v_eval = np.logspace(np.log10(vmin), np.log10(vmax), 300)
        else:
            v_eval = np.asarray(v_eval, dtype=float)

        gen_interp = self._interp_loglog(v_gen, sig_gen)
        sig_g = np.exp(gen_interp(np.log(v_eval)))
        sig_t = tab(v_eval)
        ratio = sig_g / sig_t

        return {
            "v_kms": v_eval,
            "generated": sig_g,
            "tabulated": sig_t,
            "ratio_generated_over_tabulated": ratio,
            "median_ratio": float(np.nanmedian(ratio)),
            "max_abs_log10_diff": float(np.nanmax(np.abs(np.log10(ratio)))),
        }

    def plot_profiles(
        self,
        include_generated: Optional[Iterable[str]] = None,
        include_tabulated: Optional[Iterable[str]] = None,
        title: Optional[str] = None,
        savepath: Optional[str | Path] = None,
    ):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))

        gen_names = list(include_generated) if include_generated is not None else list(self.generated)
        tab_names = list(include_tabulated) if include_tabulated is not None else list(self.tabulated)

        for name in gen_names:
            v, sig = self.generated[name]
            ax.loglog(v, sig, lw=2.2, label=f"first principles: {name}")

        for name in tab_names:
            p = self.tabulated[name]
            ax.loglog(p.v_kms, p.sigma_over_m, marker="o", ms=3.0, lw=1.2, label=f"table: {name}")

        ax.set_xlabel(r"$v_{\rm rel}\ [{\rm km/s}]$")
        ax.set_ylabel(r"$\sigma_V/m_\chi\ [{\rm cm^2/g}]$")
        if title is not None:
            ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=250, bbox_inches="tight")

        return fig, ax

    def plot_ratio(
        self,
        generated_name: str,
        tabulated_name: str,
        savepath: Optional[str | Path] = None,
    ):
        comp = self.compare_to_tabulated(generated_name, tabulated_name)
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.semilogx(comp["v_kms"], comp["ratio_generated_over_tabulated"], lw=2.0)
        ax.axhline(1.0, ls="--", lw=1.0)
        ax.set_xlabel(r"$v_{\rm rel}\ [{\rm km/s}]$")
        ax.set_ylabel("first principles / table")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, dpi=250, bbox_inches="tight")
        return fig, ax, comp


if __name__ == "__main__":
    print("I DO NOTHING")
    # Example usage. Adjust CSV paths to your local files.
    bench = GilmanBenchmarkProfiles()

    # Compute profile from first principles. For quick debugging, use fewer v points:
    # bench.compute_model("single", np.logspace(0, np.log10(300), 80))
    bench.compute_model("single")

    # Optional tabulated profiles:
    bench.load_tabulated("data/Gilma_plot_points.csv", name="Krzysztof digitization")
    bench.load_tabulated("data/Crossv_May7.csv", name="Camilo table")

    bench.plot_profiles(title="Gilman benchmark profiles, no averaging")
    plt.show()
