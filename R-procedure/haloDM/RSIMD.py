#!/usr/bin/env python3
"""
RSIDM effective cross section: Mathematica-matched p=5 version

This reproduces the Mathematica quantity:

  sigma_eff(nu) = sigma_m0 + (256*pi)/m^3 * [ ∫_0^∞ dv  K(v) * f_rel(v;nu) ] / <v^5>

with:
  K(v) = ( v^(4L+5) * Gamma^2 ) / ( (v^2 - vR^2)^2 + 16 * v^(2+4L) * Gamma^2 )

and f_rel is the relative-speed Maxwellian:
  f_rel(v) = (4*pi*v^2 / (4*pi*nu^2)^(3/2)) * exp( -v^2 / (4*nu^2) )

All v, nu, vR are dimensionless (v/c). Inputs for nu and vR are in km/s.

Integration is split exactly as in your Mathematica notebook using deltaFrac(L,Gamma,vR).
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq  # fsolve
from scipy.interpolate import PchipInterpolator
# ------------ IMPORT FROM FILES ------------ #
import config as cfg
import units as uni
from NFWProfile import sigma, NFWProfile, rho

# ----------------------------
# Units: GeV^-3 -> (cm^2 / g)
# Mathematica used division by 4578.17, which is 1/GEV_M3_TO_CM2_PER_G.
# ----------------------------
GEV_M2_TO_CM2 = 0.389379e-27
GEV_TO_G = 1.78266192e-24
GEV_M3_TO_CM2_PER_G = GEV_M2_TO_CM2 / GEV_TO_G  # ~2.185e-4


# ----------------------------
# Relative Maxwellian PDF for relative speed
# v, nu dimensionless (v/c)
# ----------------------------
def maxwell_rel_pdf(v: float, nu: float) -> float:
    return (4.0 * math.pi * v * v *
            math.exp(-v * v / (4.0 * nu * nu))) / ((4.0 * math.pi * nu * nu) ** 1.5)


# ----------------------------
# Resonant kernel factor (Mathematica integrand piece)
# ----------------------------
def resonant_factor(v: float, L: int, Gamma: float, vR: float) -> float:
    num = (v ** (4 * L + 5)) * (Gamma ** 2)
    den = (v * v - vR * vR) ** 2 + 16.0 * (v ** (2 + 4 * L)) * (Gamma ** 2)
    return num / den

def resonant_factor_k1(v: float, L: int, Gamma: float, vR: float) -> float:
    num = (v ** (4 * L + 1)) * (Gamma ** 2)
    den = (v * v - vR * vR) ** 2 + 16.0 * (v ** (2 + 4 * L)) * (Gamma ** 2)
    return num / den

# ----------------------------
# Mathematica resonance window fraction deltaFrac
# (literal port of your expression inside Abs[...])
# ----------------------------
def deltaFrac_mathematica(L: int, Gamma: float, vR: float) -> float:
    # Guard: for extreme values, the radicand could go tiny-negative due to rounding.
    rad = (vR ** (6.0 + 4.0 * L)) * (Gamma ** 2) - 8.0 * L * (1.0 + 2.0 * L) * (vR ** (4.0 + 8.0 * L)) * (Gamma ** 4)
    rad = max(rad, 0.0)

    A = 2.0 * (
            2.0 * (1.0 + 2.0 * L) * (vR ** (2.0 + 4.0 * L)) * (Gamma ** 2)
            + math.sqrt(rad)
    )
    B = (vR ** 4) - 4.0 * (1.0 + 6.0 * L + 8.0 * L * L) * (vR ** (2.0 + 4.0 * L)) * (Gamma ** 2)

    # Avoid division by zero (should not happen for your parameter regime)
    if B == 0.0:
        return 1.0  # fallback: huge window

    return abs(10.0 * A / B)


# ----------------------------
# Public: sigma_eff for p=5 with Mathematica-matched limits
# ----------------------------
def sigma_m_eff_p5(
        nu_kms: float,
        sigma_m0: float,
        m_GeV: float,
        L: int,
        Gamma: float,
        vR_kms: float,
        epsrel: float = 1e-10,
) -> float:
    """
    Effective <sigma v^5>/<v^5>/m in cm^2/g.

    Parameters
    ----------
    nu_kms : float
        1D velocity dispersion in km/s (the nu on your x-axis).
    sigma_m0 : float
        Baseline sigma/m in cm^2/g.
    m_GeV : float
        Particle mass in GeV (as in your Mathematica mtilde).
    L : int
        Partial wave index (usually 0 here).
    Gamma : float
        Resonance width parameter in the same dimensionless convention as your Mathematica input.
    vR_kms : float
        Resonance velocity (km/s).
    epsrel : float
        Relative tolerance for integration.

    Returns
    -------
    float
        sigma_eff(nu_kms) in cm^2/g.
    """
    # Convert to dimensionless v/c
    nu = nu_kms / 3.0e5
    vR = vR_kms / 3.0e5

    # <v^5> analytic for this relative Maxwellian convention (matches your Mathematica)
    v5mean = (384.0 / math.sqrt(math.pi)) * (nu ** 5)

    # Prefactor (256 pi)/m^3 with GeV^-3 -> cm^2/g conversion
    pref = (256.0 * math.pi) * (GEV_M3_TO_CM2_PER_G) / (m_GeV ** 3)

    # Mathematica window
    dfrac = deltaFrac_mathematica(L, Gamma, vR)
    vL = max((1.0 - dfrac) * vR, 0.0)
    vU = (1.0 + dfrac) * vR

    def integrand(v: float) -> float:
        return resonant_factor(v, L, Gamma, vR) * maxwell_rel_pdf(v, nu)

    def I_segment_scaled(v_a, v_b):
        # integrate over x = v/nu
        xa = v_a / nu
        xb = v_b / nu

        def integrand_x(x):
            v = nu * x
            return resonant_factor(v, L, Gamma, vR) * maxwell_rel_pdf(v, nu) * nu  # dv = nu dx

        return quad(integrand_x, xa, xb, epsrel=epsrel, epsabs=1e-30, limit=4000)[0]

    # Mimic Mathematica: [0,vL] + [vL,vU] + [vU,∞]
    # Use large recursion limits because the resonance can be very narrow.
    # I1 = quad(integrand, 0.0, vL, epsrel=epsrel, epsabs=1e-30, limit=4000)[0]
    I1 = I_segment_scaled(0.0, vL)

    I2 = quad(integrand, vL, vU, epsrel=epsrel, epsabs=1e-30, limit=8000, points=[vR])[0]

    # Infinity tail: truncate safely (the Maxwellian kills it fast)
    vmax = max(200.0 * nu, 20.0 * vR)
    # I3 = quad(integrand, vU, vmax, epsrel=epsrel, epsabs=1e-30, limit=4000)[0]
    I3 = quad(integrand, vU, np.inf, epsrel=epsrel, epsabs=1e-30, limit=4000)[0]

    I = I1 + I2 + I3

    # print("I1,I2,I3", I1, I2, I3, "fractions", I1 / (I1 + I2 + I3), I2 / (I1 + I2 + I3), I3 / (I1 + I2 + I3))

    return sigma_m0 + pref * I / v5mean

def sigma_m_eff_p1(
        nu_kms: float,
        sigma_m0: float,
        m_GeV: float,
        L: int,
        Gamma: float,
        vR_kms: float,
        epsrel: float = 1e-12,
) -> float:
    """
    Effective <sigma v>/<v>/m in cm^2/g  (K1 averaging).

    Same conventions as sigma_m_eff_p5 but using v^1 weighting.
    """

    # Convert to dimensionless v/c
    nu = nu_kms / 3.0e5
    vR = vR_kms / 3.0e5

    # <v> analytic for this relative Maxwellian
    v1mean = (4.0 / math.sqrt(math.pi)) * nu

    # Prefactor (256 pi)/m^3 with GeV^-3 -> cm^2/g conversion
    pref = (256.0 * math.pi) * (GEV_M3_TO_CM2_PER_G) / (m_GeV ** 3)

    # Mathematica-style window
    dfrac = deltaFrac_mathematica(L, Gamma, vR)
    vL = max((1.0 - dfrac) * vR, 0.0)
    vU = (1.0 + dfrac) * vR

    def integrand(v: float) -> float:
        return resonant_factor_k1(v, L, Gamma, vR) * maxwell_rel_pdf(v, nu)

    def I_segment_scaled(v_a, v_b):
        # integrate over x = v/nu
        xa = v_a / nu
        xb = v_b / nu

        def integrand_x(x):
            v = nu * x
            return resonant_factor_k1(v, L, Gamma, vR) * maxwell_rel_pdf(v, nu) * nu  # dv = nu dx

        return quad(integrand_x, xa, xb, epsrel=epsrel, epsabs=1e-30, limit=4000)[0]

    # Split integration like Mathematica
    # I1 = quad(integrand, 0.0, vL, epsrel=epsrel, epsabs=1e-30, limit=4000)[0]
    I1 = I_segment_scaled(0.0, vL)

    I2 = quad(integrand, vL, vU, epsrel=epsrel, epsabs=1e-30, limit=8000, points=[vR])[0]

    vmax = max(200.0 * nu, 20.0 * vR)
    # I3 = quad(integrand, vU, vmax, epsrel=epsrel, epsabs=1e-30, limit=4000)[0]
    I3 = quad(integrand, vU, np.inf, epsrel=epsrel, epsabs=1e-30, limit=4000)[0]

    I = I1 + I2 + I3

    # print("I1,I2,I3", I1, I2, I3, "fractions", I1 / (I1 + I2 + I3), I2 / (I1 + I2 + I3), I3 / (I1 + I2 + I3))

    return sigma_m0 + pref * I / v1mean
# ----------------------------
# Optional: normalization check (should return ~1.0)
# ----------------------------
def v5_ratio_check(nu_kms: float) -> float:
    nu = nu_kms / 3e5

    # work in w=v/nu to avoid tiny absolute scales
    def pdf_w(w):
        return (4.0 * math.pi * w * w * math.exp(-w * w / 4.0)) / ((4.0 * math.pi) ** 1.5)

    w5_num = quad(lambda w: (w ** 5) * pdf_w(w), 0.0, np.inf,
                  epsrel=1e-12, epsabs=1e-30, limit=2000)[0]
    w5_ana = 384.0 / math.sqrt(math.pi)
    return w5_num / w5_ana


# ----------------------------
# Plot helper (Mathematica-like domain)
# ----------------------------
def plot_sigma_eff_p5(
        sigma_m0=0.008,
        m_GeV=0.02,
        L=0,
        Gamma=6e-12,
        vR_kms=85.0,
        nu_min=10.0,
        nu_max=100.0,
        N=160,
):
    nus = np.logspace(np.log10(nu_min), np.log10(nu_max), N)
    sig = np.array([sigma_m_eff_p5(nu, sigma_m0, m_GeV, L, Gamma, vR_kms) for nu in nus])

    print(f"sigma_eff min/max on [{nu_min:g},{nu_max:g}] km/s:",
          float(np.min(sig)), float(np.max(sig)))

    plt.figure(figsize=(7.4, 4.8))
    plt.loglog(nus, sig, lw=2.2, color="black")
    plt.xlabel(r"$\nu\ \mathrm{[km/s]}$")
    plt.ylabel(r"$\langle\sigma v^5\rangle/\langle v^5\rangle / m\ \mathrm{[cm^2/g]}$")
    plt.title(r"Effective resonant SIDM cross section (p=5, Mathematica limits)")
    plt.grid(True, which="both", alpha=0.25)

    # auto y-range so you never "hide" the curve
    ymin = max(1e-6, np.min(sig) * 0.8)
    ymax = np.max(sig) * 1.2
    plt.ylim(ymin, ymax)
    plt.xlim(nu_min, nu_max)

    plt.tight_layout()
    plt.show()


# ============================================================
# 2) Fast interpolator wrapper for σ_eff(ν)
#    (You MUST use this for r1 scans; direct integration inside F(r) is too slow.)
# ============================================================

def build_sigma_m_eff_p5_interpolator(
        sigma_m0: float, m_GeV: float, L: int, Gamma: float, vR_kms: float,
        nu_kms_min: float = 0.5, nu_kms_max: float = 5000.0, N: int = 260,
):
    nu_grid = np.logspace(np.log10(nu_kms_min), np.log10(nu_kms_max), N)
    vals = np.array([sigma_m_eff_p5(nu, sigma_m0, m_GeV, L, Gamma, vR_kms) for nu in nu_grid])
    # vals = np.array([sigma_m_eff_p1(nu, sigma_m0, m_GeV, L, Gamma, vR_kms) for nu in nu_grid])

    # log-log PCHIP (monotone, stable)
    interp = PchipInterpolator(np.log10(nu_grid), np.log10(vals), extrapolate=True)

    def sigma_eff_from_nu_kpcGyr(nu_kpcGyr):
        # uses your cfg.kpcGyr_to_kms
        nu_kms = nu_kpcGyr * cfg.kpcGyr_to_kms
        return float(10.0 ** interp(np.log10(nu_kms)))

    # Optional helper if you want direct ν[km/s] access
    def sigma_eff_from_nu_kms(nu_kms):
        nu_kms = np.asarray(nu_kms)
        return 10 ** interp(np.log10(nu_kms))

    return sigma_eff_from_nu_kpcGyr, sigma_eff_from_nu_kms, (nu_grid, vals)


# ============================================================
# 3) RSIDM r1 root function F(r) — THIS is the correct form
# ============================================================

# (cm^2/g) -> (kpc^2/M_sun)
SIGMAMX_CGS_TO_KPC2_PER_MSUN = 2.08889e-10


def F_r1_rsidm(r, halo, tage_gyr, K_n, sigma_m_eff_nu_kpcGyr, other=True):
    """
    F(r) = (4/sqrt(pi)) * rho(r) * nu(r) * <sigma/m>_eff(nu(r)) * t_age - K_n
    Units:
      rho: M_sun/kpc^3
      nu : kpc/Gyr
      (sigma/m): kpc^2/M_sun
      t_age: Gyr
    -> dimensionless
    """
    nu = sigma(halo, r)  # kpc/Gyr
    sig_cgs = sigma_m_eff_nu_kpcGyr(nu)  # cm^2/g
    sig = sig_cgs * SIGMAMX_CGS_TO_KPC2_PER_MSUN  # kpc^2/M_sun

    if other:
        return cfg.FourOverRootPi * rho(halo, r) * nu * sig * tage_gyr - K_n
    else:
        return cfg.FourOverRootPi * rho(halo, r) * nu * sig


# ============================================================
# 4) Find all roots (branches) of F(r)=0 on a log grid
# ============================================================

def find_all_r1_roots_rsidm(
        halo, tage_gyr, sigma_m_eff_nu_kpcGyr,
        K_n=1.0, cff=0.001, rmax=2000.0, ngrid=900,
        pick="outermost", xtol=1e-4, rtol=1e-7,
):
    rmin = cfg.find_Rres(halo, cff=cff)
    if rmin <= 0:
        raise ValueError("cfg.find_Rres returned rmin <= 0; cannot build log grid.")

    rs = np.logspace(np.log10(rmin), np.log10(rmax), int(ngrid))
    Fs = np.array([F_r1_rsidm(r, halo, tage_gyr, K_n, sigma_m_eff_nu_kpcGyr) for r in rs])

    valid = np.isfinite(Fs)
    rs_v, Fs_v = rs[valid], Fs[valid]
    if len(rs_v) < 2:
        return rmin, [], {"rs": rs, "Fs": Fs, "rmin": rmin, "rmax": rmax}

    # sign-change brackets
    idx = np.where(Fs_v[:-1] * Fs_v[1:] < 0.0)[0]

    roots = []
    for i in idx:
        a, b = rs_v[i], rs_v[i + 1]
        try:
            root = brentq(
                lambda rr: F_r1_rsidm(rr, halo, tage_gyr, K_n, sigma_m_eff_nu_kpcGyr),
                a, b, xtol=xtol, rtol=rtol, maxiter=2000
            )
            roots.append(float(root))
        except ValueError:
            pass

    roots = sorted(set(roots))
    if len(roots) == 0:
        return rmin, [], {"rs": rs, "Fs": Fs, "rmin": rmin, "rmax": rmax}

    if pick == "innermost":
        r1 = roots[0]
    elif pick == "outermost":
        r1 = roots[-1]
    elif pick == "myChoice":
        if len(roots) == 3:
            r1 = roots[1]
        elif len(roots) == 2:
            r1 = roots[0]
        else:
            r1 = roots[0]
    else:
        raise ValueError("pick must be 'outermost' or 'innermost'")

    return r1, roots, {"rs": rs, "Fs": Fs, "roots": roots, "rmin": rmin, "rmax": rmax}


# ============================================================
# 5) Diagnostics (the “cross checks”)
# ============================================================

def plot_sigma_eff_curve(nu_grid_kms, sigma_vals):
    plt.figure(figsize=(7.2, 4.6))
    plt.loglog(nu_grid_kms, sigma_vals, lw=2.0, color="black")
    plt.xlabel(r"$\nu$ [km/s]")
    plt.ylabel(r"$\sigma_{\rm eff}(\nu)$ [cm$^2$/g]")
    plt.title("RSIDM σ_eff(ν) used in r₁")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_nu_and_sigma_vs_r(
        halo, sigma_eff_from_nu_kpcGyr,
        cff=0.001, rmax=2000.0, ngrid=700
):
    rmin = cfg.find_Rres(halo, cff=cff)
    rs = np.logspace(np.log10(rmin), np.log10(rmax), int(ngrid))

    nu_kms = np.array([sigma(halo, r) * cfg.kpcGyr_to_kms for r in rs])
    sig_eff = np.array([sigma_eff_from_nu_kpcGyr(sigma(halo, r)) for r in rs])

    plt.figure(figsize=(7.2, 4.6))
    plt.loglog(rs, nu_kms, lw=1.8)
    plt.xlabel(r"$r$ [kpc]")
    plt.ylabel(r"$\nu(r)$ [km/s]")
    plt.title("Halo velocity dispersion profile")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.2, 4.6))
    plt.loglog(rs, sig_eff, lw=1.8)
    plt.xlabel(r"$r$ [kpc]")
    plt.ylabel(r"$\sigma_{\rm eff}(\nu(r))$ [cm$^2$/g]")
    plt.title("Effective cross section sampled along the halo")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_F_r_for_times(
        halo, sigma_eff_from_nu_kpcGyr,
        times_gyr=(0.2, 0.4, 0.6, 1.0, 3.0, 10.0),
        K_n=1.0, cff=0.001, rmax=2000.0, ngrid=900
):
    rmin = cfg.find_Rres(halo, cff=cff)
    rs = np.logspace(np.log10(rmin), np.log10(rmax), int(ngrid))

    plt.figure(figsize=(7.6, 5.0))
    for t in times_gyr:
        Fs = np.array([F_r1_rsidm(r, halo, t, K_n, sigma_eff_from_nu_kpcGyr, other=False) for r in rs])
        plt.loglog(rs, Fs, lw=1.4, label=rf"$t={t:g}$ Gyr")

    plt.axhline(0.0, lw=1.0)
    plt.xlabel(r"$r$ [kpc]")
    plt.ylabel(r"$F(r)$")
    plt.title("RSIDM r₁ root function F(r) at different halo ages")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def r1_vs_time_plot(
        halo, sigma_eff_from_nu_kpcGyr,
        tmin=0.1, tmax=100.0, Nt=60,
        pick="outermost", K_n=1.0,
        cff=0.001, rmax=2000.0, ngrid=900
):
    ts = np.logspace(np.log10(tmin), np.log10(tmax), Nt)
    r1_outer = np.full_like(ts, np.nan, dtype=float)
    r1_inner = np.full_like(ts, np.nan, dtype=float)
    nroots = np.zeros_like(ts, dtype=int)

    for i, t in enumerate(ts):
        r1, roots, _ = find_all_r1_roots_rsidm(
            halo, tage_gyr=t, sigma_m_eff_nu_kpcGyr=sigma_eff_from_nu_kpcGyr,
            K_n=K_n, cff=cff, rmax=rmax, ngrid=ngrid, pick=pick
        )
        nroots[i] = len(roots)
        if len(roots) > 0:
            r1_inner[i] = roots[0]
            r1_outer[i] = roots[-1]

    plt.figure(figsize=(7.2, 4.8))
    plt.loglog(ts, r1_outer, marker="o", ms=3, lw=1.2, label="r1 (outermost)")
    plt.loglog(ts, r1_inner, marker="o", ms=3, lw=1.2, alpha=0.7, label="r1 (innermost)")
    plt.xlabel(r"$t_{\rm age}$ [Gyr]")
    plt.ylabel(r"$r_1$ [kpc]")
    plt.title("RSIDM r₁(t) with branch structure")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.2, 3.8))
    plt.semilogx(ts, nroots, marker="o", ms=3, lw=1.2)
    plt.xlabel(r"$t_{\rm age}$ [Gyr]")
    plt.ylabel("number of roots")
    plt.title("How many r₁ roots exist vs time?")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()

    return ts, r1_outer, r1_inner, nroots


# ============================================================
# 6) Example driver: run all cross checks
# ============================================================

def run_r1_rsidm_crosschecks(
        halo,
        sigma_m0=0.008, m_GeV=0.02, L=0, Gamma=6e-12, vR_kms=85.0,
        cff=0.001, rmax=2000.0
):
    # Build fast interpolator
    sigma_eff_from_nu_kpcGyr, sigma_eff_from_nu_kms, (nu_grid, vals) = build_sigma_m_eff_p5_interpolator(
        sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
        nu_kms_min=0.5, nu_kms_max=5000.0, N=260
    )

    # Cross-check 1: σ_eff curve (the one you matched to Mathematica)
    plot_sigma_eff_curve(nu_grid, vals)

    # Cross-check 2: ν(r) and σ_eff(ν(r))
    plot_nu_and_sigma_vs_r(halo, sigma_eff_from_nu_kpcGyr, cff=cff, rmax=rmax, ngrid=700)

    # Cross-check 3: F(r) at multiple ages (see sign changes and where roots appear)
    plot_F_r_for_times(
        halo, sigma_eff_from_nu_kpcGyr,
        times_gyr=(0.002, 0.02, 0.2, 0.4, 0.6, 0.8, 1.0, 3.0, 10.0),
        K_n=1.0, cff=cff, rmax=rmax, ngrid=900
    )

    # Cross-check 4: r1(t) + branch count
    r1_vs_time_plot(
        halo, sigma_eff_from_nu_kpcGyr,
        tmin=1e-3, tmax=1e4, Nt=65,
        pick="outermost", K_n=1.0,
        cff=cff, rmax=rmax, ngrid=900
    )


# ============================================================
# 7) Example driver: run all cross checks
# ============================================================

def _F_at_radius_from_grid(rs, Fs, r):
    """Quick robust evaluation using nearest grid point (good enough for topology)."""
    j = int(np.argmin(np.abs(rs - r)))
    return float(Fs[j])


def pick_r1_center_connected(rs, Fs, roots, rmin, rmax, convention="F>=0"):
    """
    Pick r1 as the OUTER EDGE of the CENTER-CONNECTED collisional region.

    Convention: collisional if F >= 0 (this matches your current F definition).
    If the center is not collisional, returns None (meaning: no center-connected core).

    Parameters
    ----------
    rs, Fs : arrays (log grid and values)
    roots  : sorted list of roots
    rmin   : inner cutoff radius
    rmax   : max radius scanned
    """
    if len(roots) == 0:
        return None

    # interval boundaries: [rmin, root1, root2, ..., rmax]
    bounds = [rmin] + list(roots) + [rmax]

    # Determine which intervals are collisional by sampling midpoint
    coll = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a <= 0 or b <= 0:
            coll.append(False)
            continue
        mid = math.sqrt(a * b)
        Fmid = _F_at_radius_from_grid(rs, Fs, mid)
        coll.append(Fmid >= 0.0)

    # If the first interval [rmin, root1) is not collisional -> no center-connected core
    if not coll[0]:
        return None

    # Find the first NON-collisional interval after the connected block.
    # Boundary between interval (k-1) and k is at bounds[k] (a root).
    for k in range(1, len(coll)):
        if not coll[k]:
            return bounds[k]  # root position

    # If everything collisional out to rmax, the boundary is effectively rmax
    return rmax


def select_root_with_continuity(candidates, r_prev):
    """
    Choose candidate root closest to previous r1 in log-space.
    """
    if r_prev is None or not np.isfinite(r_prev):
        # default: take the largest candidate (often stable early)
        return candidates[-1]

    log_prev = math.log(r_prev)
    diffs = [abs(math.log(r) - log_prev) for r in candidates]
    return candidates[int(np.argmin(diffs))]


def track_r1_center_connected(
        halo,
        sigma_eff_from_nu_kpcGyr,
        tmin=0.1, tmax=1e4, Nt=70,
        K_n=1.0, cff=0.001,
        rmax=2000.0, ngrid=900,
        fallback="outermost",  # what to do if center is not collisional
):
    """
    Track r1(t) using center-connected definition with optional continuity smoothing.

    Returns
    -------
    ts : array
    r1_cc : array (selected r1)
    nroots : array
    roots_list : list of lists of roots at each time
    """
    ts = np.logspace(np.log10(tmin), np.log10(tmax), Nt)
    r1_cc = np.full_like(ts, np.nan, dtype=float)
    nroots = np.zeros_like(ts, dtype=int)
    roots_list = []

    r_prev = None

    for i, t in enumerate(ts):
        # Use your existing root finder to get all roots + debug grid
        r1_any, roots, dbg = find_all_r1_roots_rsidm(
            halo,
            tage_gyr=t,
            sigma_m_eff_nu_kpcGyr=sigma_eff_from_nu_kpcGyr,
            K_n=K_n,
            cff=cff,
            rmax=rmax,
            ngrid=ngrid,
            pick="outermost"  # doesn't matter; we override selection
        )

        roots_list.append(list(roots))
        nroots[i] = len(roots)

        rs = dbg["rs"]
        Fs = dbg["Fs"]
        rmin = dbg["rmin"]

        # (A) primary: center-connected boundary
        r_cc = pick_r1_center_connected(rs, Fs, roots, rmin, rmax)

        # (B) if no center-connected region, fall back
        if r_cc is None:
            if fallback == "outermost" and len(roots) > 0:
                r_cc = roots[-1]
            elif fallback == "innermost" and len(roots) > 0:
                r_cc = roots[0]
            else:
                r_cc = np.nan

        # (C) continuity: if there are multiple plausible candidates
        # Sometimes topology changes and both r_cc and a nearby root could be valid.
        # We'll allow a small set of candidates and pick closest to previous.
        candidates = []
        if len(roots) > 0:
            # always include computed r_cc if finite
            if np.isfinite(r_cc):
                candidates.append(r_cc)
            # also include adjacent roots (helps smooth switches)
            candidates.extend(roots)

            # unique + sorted
            candidates = sorted(set([float(r) for r in candidates if np.isfinite(r)]))

        if len(candidates) > 0:
            r_sel = select_root_with_continuity(candidates, r_prev)
        else:
            r_sel = np.nan

        r1_cc[i] = r_sel
        if np.isfinite(r_sel):
            r_prev = r_sel

    return ts, r1_cc, nroots, roots_list


# ============================================================
# Plot helpers
# ============================================================

def plot_r1_tracking(ts, r1_old_outer, r1_new, title="r1(t) branch tracking comparison"):
    plt.figure(figsize=(7.4, 4.8))
    plt.loglog(ts, r1_old_outer, marker="o", ms=3, lw=1.2, label="old: outermost pick")
    plt.loglog(ts, r1_new, marker="o", ms=3, lw=1.2, label="new: center-connected + continuity")
    plt.xlabel(r"$t_{\rm age}$ [Gyr]")
    plt.ylabel(r"$r_1$ [kpc]")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_nroots(ts, nroots):
    plt.figure(figsize=(7.4, 3.8))
    plt.semilogx(ts, nroots, marker="o", ms=3, lw=1.2)
    plt.xlabel(r"$t_{\rm age}$ [Gyr]")
    plt.ylabel("number of roots")
    plt.title("How many r1 roots exist vs time?")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_root_map(ts, roots_list, rmin_floor=None, rmax_cap=None):
    """
    Visualize all roots as scatter points in (t, r).
    Great for seeing topology changes.
    """
    plt.figure(figsize=(7.6, 4.8))
    for t, roots in zip(ts, roots_list):
        if len(roots) == 0:
            continue
        rr = np.array(roots)
        if rmin_floor is not None:
            rr = rr[rr >= rmin_floor]
        if rmax_cap is not None:
            rr = rr[rr <= rmax_cap]
        tt = np.full_like(rr, t)
        plt.scatter(tt, rr, s=10)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$t_{\rm age}$ [Gyr]")
    plt.ylabel(r"roots $r$ [kpc]")
    plt.title("Root topology map: all r1 solutions vs time")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()


# ============================================================
# Runner: compare old vs new selection
# ============================================================

def run_branch_tracking_demo(
        halo,
        sigma_eff_from_nu_kpcGyr,
        tmin=0.1, tmax=1e4, Nt=70,
        K_n=1.0, cff=0.001,
        rmax=2000.0, ngrid=900
):
    # old outermost picks (for comparison)
    ts = np.logspace(np.log10(tmin), np.log10(tmax), Nt)
    r1_outer = np.full_like(ts, np.nan, dtype=float)
    nroots_old = np.zeros_like(ts, dtype=int)

    for i, t in enumerate(ts):
        r1, roots, _ = find_all_r1_roots_rsidm(
            halo, tage_gyr=t, sigma_m_eff_nu_kpcGyr=sigma_eff_from_nu_kpcGyr,
            K_n=K_n, cff=cff, rmax=rmax, ngrid=ngrid, pick="outermost"
        )
        nroots_old[i] = len(roots)
        r1_outer[i] = roots[-1] if len(roots) else np.nan

    # new tracking
    ts2, r1_cc, nroots, roots_list = track_r1_center_connected(
        halo, sigma_eff_from_nu_kpcGyr,
        tmin=tmin, tmax=tmax, Nt=Nt,
        K_n=K_n, cff=cff,
        rmax=rmax, ngrid=ngrid,
        fallback="outermost"
    )

    # plots
    plot_r1_tracking(ts, r1_outer, r1_cc, title="r1(t): outermost vs center-connected tracking")
    plot_nroots(ts2, nroots)
    plot_root_map(ts2, roots_list, rmin_floor=None, rmax_cap=None)

    return ts2, r1_outer, r1_cc, nroots, roots_list


def make_sigma_eff_from_nu_kpcGyr(
        sigma_m0=0.008, m_GeV=0.02, L=0, Gamma=6e-12, vR_kms=85.0
):
    """
    Returns a callable sigma_eff_from_nu_kpcGyr(nu_kpcGyr) -> cm^2/g
    using your p=5 Mathematica-matched implementation.
    """

    def sigma_eff_from_nu_kpcGyr(nu_kpcGyr):
        nu_kms = nu_kpcGyr * cfg.kpcGyr_to_kms
        # return sigma_m_eff_p5(
        #     nu_kms=nu_kms,
        #     sigma_m0=sigma_m0,
        #     m_GeV=m_GeV,
        #     L=L,
        #     Gamma=Gamma,
        #     vR_kms=vR_kms
        # )

        return sigma_m_eff_p1(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0,
            m_GeV=m_GeV,
            L=L,
            Gamma=Gamma,
            vR_kms=vR_kms
        )

    return sigma_eff_from_nu_kpcGyr


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------------
    # 1) CROSS CHECK
    # -----------------------------------------------------------------------------------------
    # Main plot: this is what you want
    # plot_sigma_eff_p5()

    # Optional: sanity check that <v^5> normalization matches analytic formula
    # for nu_kms in [10, 20, 30]:
    #     print(nu_kms, v5_ratio_check(nu_kms))

    # -----------------------------------------------------------------------------------------
    # 2) CROSS CHECK
    # -----------------------------------------------------------------------------------------
    halo = NFWProfile(_M_vir=1e11, _con=10.6)  # example
    run_r1_rsidm_crosschecks(halo, cff=0.001, rmax=2000.0)

    # -----------------------------------------------------------------------------------------
    # 3) CROSS CHECK
    # -----------------------------------------------------------------------------------------
    # halo = NFWProfile(_M_vir=1e10, _con=13.3)
    sigma_eff_from_nu_kpcGyr = make_sigma_eff_from_nu_kpcGyr(
        sigma_m0=0.008,
        m_GeV=0.02,
        L=0,
        Gamma=6e-12,
        vR_kms=85.0
    )
    #
    # # Optional: quick plot to verify sigma_eff looks right
    # nus = np.logspace(np.log10(1.0), np.log10(3000.0), 250)  # km/s
    # sigs = np.array([sigma_m_eff_p5(nu, 0.008, 0.02, 0, 6e-12, 85.0) for nu in nus])
    # plt.figure(figsize=(7.4,4.8))
    # plt.loglog(nus, sigs, lw=2.0, color="black")
    # plt.xlabel(r"$\nu$ [km/s]")
    # plt.ylabel(r"$\sigma_{\rm eff}$ [cm$^2$/g]")
    # plt.grid(True, which="both", alpha=0.25)
    # plt.tight_layout()
    # plt.show()
    #
    # ts, r1_outer, r1_cc, nroots, roots_list = run_branch_tracking_demo(
    #     halo,
    #     sigma_eff_from_nu_kpcGyr,
    #     tmin=0.1, tmax=1e4, Nt=65,
    #     cff=0.001, rmax=2000.0, ngrid=900
    # )
