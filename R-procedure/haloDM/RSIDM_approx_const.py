import numpy as np
import math
from NFWProfile import NFWProfile, sigma
from RSIMD import resonant_factor, resonant_factor_k1, sigma_m_eff_p5, sigma_m_eff_p1
import config as cfg
import auxiliaryFunctions as aux  # helpful functions

# ----------------------------
# Units: GeV^-3 -> (cm^2 / g)
# Mathematica used division by 4578.17, which is 1/GEV_M3_TO_CM2_PER_G.
# ----------------------------
GEV_M2_TO_CM2 = 0.389379e-27
GEV_TO_G = 1.78266192e-24
GEV_M3_TO_CM2_PER_G = GEV_M2_TO_CM2 / GEV_TO_G  # ~2.185e-4


def sigma1D_NFW_analytic(halo):
    rs = get_rs_kpc(halo)
    rho_s = halo.rho_s  # assuming M_sun/kpc^3
    G = 4.30091e-6      # kpc km^2 s^-2 M_sun^-1

    # result in km/s
    return 1.10 * math.sqrt(G * rho_s * rs**2)

# ----------------------------
# Helpers: get r_s robustly
# ----------------------------
def get_rs_kpc(halo):
    for attr in ["r_s", "rs", "R_s", "Rs"]:
        if hasattr(halo, attr):
            return float(getattr(halo, attr))
    # last resort: if halo stores virial radius and concentration
    if hasattr(halo, "Rvir") and hasattr(halo, "con"):
        return float(getattr(halo, "Rvir")) / float(getattr(halo, "con"))
    raise AttributeError("Cannot infer r_s from halo. Add halo.r_s (kpc) or adjust get_rs_kpc().")


# ----------------------------
# Delta-function (single-velocity) approximations
# ----------------------------
def sigma_m_delta_p5_from_nu(
    nu_kms: float,
    sigma_m0: float, m_GeV: float, L: int, Gamma: float, vR_kms: float,
    vstar="mean",  # "mean" or "rms" or float (km/s)
):
    """
    Single-velocity proxy for p=5:
      sigma_delta,5 = sigma_m0 + pref * resonant_factor(v*) / v*^5
    """
    # convert to dimensionless nu, vR
    nu = nu_kms / 3.0e5
    vR = vR_kms / 3.0e5

    # choose v*
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

    pref = (256.0 * math.pi) * (GEV_M3_TO_CM2_PER_G) / (m_GeV ** 3)
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

    pref = (256.0 * math.pi) * (GEV_M3_TO_CM2_PER_G) / (m_GeV ** 3)
    K = resonant_factor_k1(v, L, Gamma, vR)
    return sigma_m0 + pref * K / v


# ----------------------------
# Benchmark + halo list runner
# ----------------------------
def benchmark_table_for_halos(
    halos,
    # RSIDM benchmark
    sigma_m0=0.008, m_GeV=0.02, L=0, Gamma=6e-12, vR_kms=85.0,
    # where to evaluate nu(r)
    rstar_mode="rs",  # "rs" or float kpc
    # how to define v* for delta approx
    vstar="mean",     # "mean" or "rms" or float km/s
    # numerical precision for full averages
    epsrel=1e-10,
):
    rows = []

    for halo in halos:
        # radius choice
        if rstar_mode == "rs":
            r_star = get_rs_kpc(halo)
        elif isinstance(rstar_mode, (int, float)):
            r_star = float(rstar_mode)
        else:
            raise ValueError("rstar_mode must be 'rs' or a float kpc")

        # velocity dispersion at r_star
        nu_kpcGyr = sigma(halo, r_star)
        nu_kms = nu_kpcGyr * cfg.kpcGyr_to_kms

        # (1) kappa-weighted (p=5) full average
        sig_kappa5 = sigma_m_eff_p5(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            epsrel=epsrel
        )

        # (2) "single matching" delta proxy (p=5)
        sig_delta5 = sigma_m_delta_p5_from_nu(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            vstar=vstar
        )

        # (optional) collision-rate average and its delta proxy too
        sig_avg1 = sigma_m_eff_p1(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            epsrel=epsrel
        )
        sig_delta1 = sigma_m_delta_p1_from_nu(
            nu_kms=nu_kms,
            sigma_m0=sigma_m0, m_GeV=m_GeV, L=L, Gamma=Gamma, vR_kms=vR_kms,
            vstar=vstar
        )

        rows.append((halo, r_star, nu_kms, sig_kappa5, sig_delta5, sig_avg1, sig_delta1))

    # Print a readable table
    print("\n=== RSIDM constant-σ/m approximations ===")
    print(f"Benchmark: sigma_m0={sigma_m0}, m_GeV={m_GeV}, L={L}, Gamma={Gamma}, vR_kms={vR_kms}")
    print(f"nu evaluated at: r*={rstar_mode} ; delta approx uses v*='{vstar}'")
    print()
    header = f"{'halo':>14}  {'r*[kpc]':>9}  {'nu*[km/s]':>10}  {'sigma_kappa(p=5)':>17}  {'sigma_delta(p=5)':>17}  {'sigma_avg(p=1)':>15}  {'sigma_delta(p=1)':>15}"
    print(header)
    print("-"*len(header))

    for (halo, r_star, nu_kms, sk5, sd5, sa1, sd1) in rows:
        # try to show a compact halo label
        Mvir = getattr(halo, "M_vir", getattr(halo, "_M_vir", None))
        con  = getattr(halo, "con",   getattr(halo, "_con",  None))
        if Mvir is not None and con is not None:
            hlabel = f"M={float(Mvir):.2e},c={float(con):.2f}"
        else:
            hlabel = halo.__class__.__name__

        print(f"{hlabel:>14}  {r_star:9.3g}  {nu_kms:10.6g}  {sk5:17.6g}  {sd5:17.6g}  {sa1:15.6g}  {sd1:15.6g}")

    return rows


if __name__ == "__main__":
    # 1) Choose your benchmark RSIDM point
    bench = dict(sigma_m0=0.008, m_GeV=0.02, L=0, Gamma=6e-12, vR_kms=85.0)

    M_vir_arr = np.array([10**8.4, 10**8.6, 10**8.8, 1e9, 10**9.2, 10**9.4, 10**9.6, 10**9.8, 1e10, 1e11])
    c_arr = aux.DMc_gravo(M_vir_arr, 0, cfg.const_h)  # [dimensionless] concentration of DM
    print("M_vir_arr:", M_vir_arr)
    print("c_arr:", c_arr)

    # 2) Add halos (edit these to your target set)
    halos = [NFWProfile(_M_vir=M_vir_arr[i], _con=c_arr[i]) for i in range(0, len(M_vir_arr))]

    # 3) Compute kappa(p=5) and single-matching for each halo
    benchmark_table_for_halos(
        halos,
        **bench,
        rstar_mode="rs",   # nu at r_s
        vstar="mean",      # delta approx uses v* = <v_rel> = (4/sqrt(pi))*nu
        # epsrel=1e-18,
    )

    # 4) Compute analytic approximation
    print()
    for i in range(len(M_vir_arr)):
        sigma_1d_analytic = sigma1D_NFW_analytic(halos[i])
        print("Mvir:", M_vir_arr[i], "sigma_1d_analytic:", sigma_1d_analytic)
