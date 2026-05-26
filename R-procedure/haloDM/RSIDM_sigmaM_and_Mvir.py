import numpy as np
import math
import matplotlib.pyplot as plt

from NFWProfile import NFWProfile, sigma
from RSIMD import resonant_factor, resonant_factor_k1, sigma_m_eff_p5, sigma_m_eff_p1
import config as cfg
import auxiliaryFunctions as aux

# ----------------------------
# Units: GeV^-3 -> (cm^2 / g)
# ----------------------------
GEV_M2_TO_CM2 = 0.389379e-27
GEV_TO_G = 1.78266192e-24
GEV_M3_TO_CM2_PER_G = GEV_M2_TO_CM2 / GEV_TO_G  # ~2.185e-4

def find_mass_of_max_sigma(rows, sigma_key="sig_kappa5_num"):
    """
    Find the halo mass where a chosen effective cross section is maximal.

    Parameters
    ----------
    rows : list of dict
        Output from compute_benchmark_rows(...)
    sigma_key : str
        Which sigma column to maximize, e.g.
        - "sig_kappa5_num"
        - "sig_delta5_num"
        - "sig_kappa5_analytic"
        - "sig_delta5_analytic"

    Returns
    -------
    best_row : dict
        The row corresponding to the maximum value.
    """
    if len(rows) == 0:
        raise ValueError("rows is empty")

    sigma_arr = np.array([row[sigma_key] for row in rows], dtype=float)

    if np.all(np.isnan(sigma_arr)):
        raise ValueError(f"All values in {sigma_key} are NaN")

    imax = np.nanargmax(sigma_arr)
    return rows[imax]

def get_rs_kpc(halo):
    """Try several common attribute names for r_s."""
    for attr in ["r_s", "rs", "R_s", "Rs", "_r_s", "_rs"]:
        if hasattr(halo, attr):
            return float(getattr(halo, attr))

    # Try virial radius / concentration combinations
    Rvir_names = ["Rvir", "R_vir", "_Rvir", "_R_vir"]
    con_names = ["con", "_con", "c", "_c"]

    Rvir_val = None
    con_val = None

    for attr in Rvir_names:
        if hasattr(halo, attr):
            Rvir_val = float(getattr(halo, attr))
            break

    for attr in con_names:
        if hasattr(halo, attr):
            con_val = float(getattr(halo, attr))
            break

    if Rvir_val is not None and con_val is not None:
        return Rvir_val / con_val

    raise AttributeError("Cannot infer r_s from halo. Add halo.r_s (kpc) or adjust get_rs_kpc().")


def sigma1D_NFW_analytic(halo):
    """
    Analytic estimate of the characteristic 1D velocity dispersion for NFW halo.
    Returns km/s.
    """
    rs = get_rs_kpc(halo)
    rho_s = halo.rho_s  # assumed M_sun / kpc^3
    G = 4.30091e-6      # kpc km^2 s^-2 M_sun^-1

    return 1.10 * math.sqrt(G * rho_s * rs**2)


def sigma_m_delta_p5_from_nu(
    nu_kms: float,
    sigma_m0: float, m_GeV: float, L: int, Gamma: float, vR_kms: float,
    vstar="mean",
):
    """
    Single-velocity proxy for p=5:
      sigma_delta,5 = sigma_m0 + pref * resonant_factor(v*) / v*^5
    """
    nu = nu_kms / 3.0e5
    vR = vR_kms / 3.0e5

    if isinstance(vstar, (int, float)):
        v = float(vstar) / 3.0e5
    else:
        if vstar == "mean":
            v = (4.0 / math.sqrt(math.pi)) * nu
        elif vstar == "rms":
            v = math.sqrt(6.0) * nu
        else:
            raise ValueError("vstar must be 'mean', 'rms', or a float km/s")

    if v <= 0:
        return float("nan")

    pref = (256.0 * math.pi) * GEV_M3_TO_CM2_PER_G / (m_GeV ** 3)
    K = resonant_factor(v, L, Gamma, vR)
    return sigma_m0 + pref * K / (v ** 5)


def sigma_m_delta_p1_from_nu(
    nu_kms: float,
    sigma_m0: float, m_GeV: float, L: int, Gamma: float, vR_kms: float,
    vstar="mean",
):
    """
    Single-velocity proxy for p=1:
      sigma_delta,1 = sigma_m0 + pref * resonant_factor_k1(v*) / v*
    """
    nu = nu_kms / 3.0e5
    vR = vR_kms / 3.0e5

    if isinstance(vstar, (int, float)):
        v = float(vstar) / 3.0e5
    else:
        if vstar == "mean":
            v = (4.0 / math.sqrt(math.pi)) * nu
        elif vstar == "rms":
            v = math.sqrt(6.0) * nu
        else:
            raise ValueError("vstar must be 'mean', 'rms', or a float km/s")

    if v <= 0:
        return float("nan")

    pref = (256.0 * math.pi) * GEV_M3_TO_CM2_PER_G / (m_GeV ** 3)
    K = resonant_factor_k1(v, L, Gamma, vR)
    return sigma_m0 + pref * K / v


def compute_benchmark_rows(
    halos,
    sigma_m0=0.008, m_GeV=0.02, L=0, Gamma=6e-12, vR_kms=85.0,
    rstar_mode="rs",
    vstar="mean",
    epsrel=1e-10,
):
    """
    Return structured results for each halo.
    """
    rows = []

    for halo in halos:
        if rstar_mode == "rs":
            r_star = get_rs_kpc(halo)
        elif isinstance(rstar_mode, (int, float)):
            r_star = float(rstar_mode)
        else:
            raise ValueError("rstar_mode must be 'rs' or a float kpc")

        # numerical velocity scale from profile
        nu_kpcGyr = sigma(halo, r_star)
        nu_kms = nu_kpcGyr * cfg.kpcGyr_to_kms

        # analytic velocity estimate
        nu_analytic_kms = sigma1D_NFW_analytic(halo)

        # full averages using numerical nu(r*)
        sig_kappa5_num = sigma_m_eff_p5(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            epsrel=epsrel
        )
        sig_avg1_num = sigma_m_eff_p1(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            epsrel=epsrel
        )

        # proxy approximations using numerical nu(r*)
        sig_delta5_num = sigma_m_delta_p5_from_nu(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            vstar=vstar
        )
        sig_delta1_num = sigma_m_delta_p1_from_nu(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            vstar=vstar
        )

        # full averages using analytic velocity estimate
        sig_kappa5_analytic = sigma_m_eff_p5(
            nu_kms=nu_analytic_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            epsrel=epsrel
        )
        sig_delta5_analytic = sigma_m_delta_p5_from_nu(
            nu_kms=nu_analytic_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            vstar=vstar
        )

        Mvir = getattr(halo, "M_vir", getattr(halo, "_M_vir", np.nan))
        con = getattr(halo, "con", getattr(halo, "_con", np.nan))

        rows.append({
            "halo": halo,
            "Mvir": float(Mvir),
            "con": float(con),
            "r_star_kpc": r_star,
            "nu_num_kms": nu_kms,
            "nu_analytic_kms": nu_analytic_kms,
            "sig_kappa5_num": sig_kappa5_num,
            "sig_delta5_num": sig_delta5_num,
            "sig_avg1_num": sig_avg1_num,
            "sig_delta1_num": sig_delta1_num,
            "sig_kappa5_analytic": sig_kappa5_analytic,
            "sig_delta5_analytic": sig_delta5_analytic,
        })

    return rows


def print_benchmark_table(rows):
    print("\n=== RSIDM constant-σ/m approximations ===")
    header = (
        f"{'Mvir':>12}  {'c':>8}  {'r*[kpc]':>9}  {'nu_num':>10}  {'nu_an':>10}  "
        f"{'K5(num)':>12}  {'delta5(num)':>12}  {'K5(an)':>12}  {'delta5(an)':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['Mvir']:12.4e}  "
            f"{row['con']:8.3f}  "
            f"{row['r_star_kpc']:9.3g}  "
            f"{row['nu_num_kms']:10.4f}  "
            f"{row['nu_analytic_kms']:10.4f}  "
            f"{row['sig_kappa5_num']:12.5g}  "
            f"{row['sig_delta5_num']:12.5g}  "
            f"{row['sig_kappa5_analytic']:12.5g}  "
            f"{row['sig_delta5_analytic']:12.5g}"
        )


def plot_sigma_eff_vs_halo_mass(rows, savepath=None):
    """
    Plot effective sigma/m vs halo mass.
    Keeps both:
      - full K5 result
      - approximation result
    and compares numerical nu(r_s) with analytic velocity estimate.
    """
    Mvir = np.array([row["Mvir"] for row in rows])

    sig_kappa5_num = np.array([row["sig_kappa5_num"] for row in rows])
    sig_delta5_num = np.array([row["sig_delta5_num"] for row in rows])

    sig_kappa5_analytic = np.array([row["sig_kappa5_analytic"] for row in rows])
    sig_delta5_analytic = np.array([row["sig_delta5_analytic"] for row in rows])

    plt.figure(figsize=(8, 5.5))

    plt.plot(Mvir, sig_kappa5_num, marker='o', label=r'Full $K_5$ (numerical $\nu(r_s)$)')
    # plt.plot(Mvir, sig_delta5_num, marker='s', linestyle='--', label=r'Approx. $K_5$ proxy (numerical $\nu(r_s)$)')

    plt.plot(Mvir, sig_kappa5_analytic, marker='^', label=r'Full $K_5$ (analytic $\sigma_{1D}$)')
    # plt.plot(Mvir, sig_delta5_analytic, marker='d', linestyle='--', label=r'Approx. $K_5$ proxy (analytic $\sigma_{1D}$)')

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$M_{\rm vir}\,[M_\odot]$")
    plt.ylabel(r"Effective $\sigma/m\;[{\rm cm^2/g}]$")
    plt.title(r"Effective RSIDM $\sigma/m$ mapped onto constant-SIDM benchmark")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # --- LUCA BENCHMARK
    # sigma_m0 = 0.007,
    # m_GeV = 0.02,
    # L = 0,
    # Gamma = 9.5e-12,
    # vR_kms = 200.0,

    # --- NEW PROPOSITION
    # sigma_m0 = 0.1,
    # m_GeV = 0.055,
    # L = 0,
    # Gamma = 9e-13,
    # vR_kms = 28.0

    bench = dict(
        sigma_m0 = 0.141,
        m_GeV = 0.0301,
        L = 0,
        Gamma = 1.93e-13,
        vR_kms = 55.8,
    )
    # M_vir_arr = np.logspace(6.0, 12.0, num=500)
    M_vir_arr = np.logspace(8.0, 10.2, num=12)
    # M_vir_arr = np.array([10**8.4, 10**8.6, 10**8.8, 1e9, 10**9.2, 10**9.4, 10**9.6, 10**9.8, 1e10, 1e11])
    c_arr = aux.DMc_gravo(M_vir_arr, 0, cfg.const_h)

    halos = [NFWProfile(_M_vir=M_vir_arr[i], _con=c_arr[i]) for i in range(len(M_vir_arr))]

    rows = compute_benchmark_rows(
        halos,
        **bench,
        rstar_mode="rs",
        vstar="mean",
        epsrel=1e-10,
    )

    print_benchmark_table(rows)


    best_row = find_mass_of_max_sigma(rows, sigma_key="sig_kappa5_num")

    print("\n=== Maximum effective cross section ===")
    print(f"Mvir               = {best_row['Mvir']:.6e} Msun")
    print(f"Mvir               = {np.log10(best_row['Mvir']):.6e} log10(Msun)")
    print(f"concentration      = {best_row['con']:.4f}")
    print(f"nu_num_kms         = {best_row['nu_num_kms']:.6f} km/s")
    print(f"sig_kappa5_num     = {best_row['sig_kappa5_num']:.6g} cm^2/g")
    print(f"sig_delta5_num     = {best_row['sig_delta5_num']:.6g} cm^2/g")
    print(f"sig_kappa5_analytic= {best_row['sig_kappa5_analytic']:.6g} cm^2/g")
    print(f"sig_delta5_analytic= {best_row['sig_delta5_analytic']:.6g} cm^2/g")

    plot_sigma_eff_vs_halo_mass(
        rows,
        # savepath="sigma_eff_vs_Mvir.png"
    )

# SCANN MASS
# sigma_m0 = 0.141,
# m_GeV = 0.0301,
# L = 0,
# Gamma = 1.93e-13,
# vR_kms = 55.8,

# SCANN MASS
# sigma_m0 = 0.141,
# m_GeV = 6.81,
# L = 0,
# Gamma = 1.93e-6,
# vR_kms = 55.8,
