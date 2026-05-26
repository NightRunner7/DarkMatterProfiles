import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import quad

from NFWProfile import NFWProfile, sigma
from TruncatedNFWProfile import TruncatedNFWProfile
import config as cfg
import auxiliaryFunctions as aux


# ============================================================
# 1. Basic halo utilities
# ============================================================

def get_rs_kpc(halo):
    """Try several common attribute names for r_s."""
    for attr in ["r_s", "rs", "R_s", "Rs", "_r_s", "_rs"]:
        if hasattr(halo, attr):
            return float(getattr(halo, attr))

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

    raise AttributeError(
        "Cannot infer r_s from halo. Add halo.r_s or adjust get_rs_kpc()."
    )


def sigma1D_NFW_analytic(halo):
    """
    Analytic estimate of characteristic 1D velocity dispersion for an NFW halo.

    Returns
    -------
    float
        sigma_1D in km/s.
    """
    rs = get_rs_kpc(halo)
    rho_s = halo.rho_s  # assumed M_sun / kpc^3

    G = 4.30091e-6      # kpc km^2 s^-2 M_sun^-1

    return 1.10 * math.sqrt(G * rho_s * rs**2)


def sigma1D_NFW_analytic_test():
    """
    Analytic estimate of characteristic 1D velocity dispersion for an NFW halo.

    Returns
    -------
    float
        sigma_1D in km/s.
    """
    rs = 0.21
    rho_s =  10**(7.57) # assumed M_sun / kpc^3

    G = 4.30091e-6      # kpc km^2 s^-2 M_sun^-1

    return 1.10 * math.sqrt(G * rho_s * rs**2)

nu_test = sigma1D_NFW_analytic_test()
print("nu_test:", nu_test)

# ============================================================
# 2. Read tabulated sigma(v_rel) model
# ============================================================

class TabulatedCrossSection:
    """
    Treat a CSV file as the velocity-dependent SIDM cross-section model.

    The file should have two columns:

        v_rel [km/s], sigma/m [cm^2/g]

    Example:
        0.1, 10.67
        0.102, 10.67
        ...
    """

    def __init__(self, filename, log_interp=True, extrapolation="edge"):
        data = np.loadtxt(filename, delimiter=",")

        self.v_kms = np.asarray(data[:, 0], dtype=float)
        self.sigma_m = np.asarray(data[:, 1], dtype=float)

        mask = (
            np.isfinite(self.v_kms)
            & np.isfinite(self.sigma_m)
            & (self.v_kms > 0)
            & (self.sigma_m > 0)
        )

        self.v_kms = self.v_kms[mask]
        self.sigma_m = self.sigma_m[mask]

        order = np.argsort(self.v_kms)
        self.v_kms = self.v_kms[order]
        self.sigma_m = self.sigma_m[order]

        self.v_min = float(self.v_kms[0])
        self.v_max = float(self.v_kms[-1])

        self.log_interp = log_interp
        self.extrapolation = extrapolation

        if log_interp:
            x = np.log(self.v_kms)
            y = np.log(self.sigma_m)

            if extrapolation == "edge":
                self._interp = interp1d(
                    x,
                    y,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(y[0], y[-1]),
                )
            elif extrapolation == "extrapolate":
                self._interp = interp1d(
                    x,
                    y,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            else:
                raise ValueError("extrapolation must be 'edge' or 'extrapolate'")

        else:
            x = self.v_kms
            y = self.sigma_m

            if extrapolation == "edge":
                self._interp = interp1d(
                    x,
                    y,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(y[0], y[-1]),
                )
            elif extrapolation == "extrapolate":
                self._interp = interp1d(
                    x,
                    y,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            else:
                raise ValueError("extrapolation must be 'edge' or 'extrapolate'")

    def __call__(self, v_rel_kms):
        """
        Return sigma(v_rel)/m in cm^2/g.

        Parameters
        ----------
        v_rel_kms : float or array
            Relative velocity in km/s.
        """
        v = np.asarray(v_rel_kms, dtype=float)

        if np.any(v <= 0):
            # For safety. The physical interpolation is only for positive v.
            v = np.maximum(v, 1e-300)

        if self.log_interp:
            return np.exp(self._interp(np.log(v)))
        else:
            return self._interp(v)


# ============================================================
# 3. Maxwellian K_p average from tabulated sigma(v)
# ============================================================

# def sigma_eff_from_table(
#     nu_kms,
#     sigma_of_v,
#     p=5,
#     x_min=0.0,
#     x_max=60.0,
#     epsabs=0.0,
#     epsrel=1e-8,
# ):
#     """
#     Compute effective cross section:
#
#         sigma_eff_p(nu)
#         =
#         < sigma(v_rel) v_rel^p > / < v_rel^p >
#
#     using the relative Maxwellian distribution.
#
#     The constants in f_rel cancel, so after changing variable
#
#         x = v_rel / nu,
#
#     the weight is
#
#         x^(p+2) exp(-x^2/4).
#
#     Parameters
#     ----------
#     nu_kms : float
#         1D velocity dispersion in km/s.
#
#     sigma_of_v : callable
#         Function sigma_of_v(v_rel_kms), returning sigma/m in cm^2/g.
#
#     p : int or float
#         Velocity weighting power. For heat conduction use p=5.
#
#     Returns
#     -------
#     float
#         sigma_eff/m in cm^2/g.
#     """
#     if nu_kms <= 0:
#         return np.nan
#
#     def weight_x(x):
#         return x ** (p + 2) * math.exp(-0.25 * x * x)
#
#     def numerator_integrand(x):
#         v_rel = nu_kms * x
#         return float(sigma_of_v(v_rel)) * weight_x(x)
#
#     def denominator_integrand(x):
#         return weight_x(x)
#
#     num, _ = quad(
#         numerator_integrand,
#         x_min,
#         x_max,
#         epsabs=epsabs,
#         epsrel=epsrel,
#         limit=300,
#     )
#
#     den, _ = quad(
#         denominator_integrand,
#         x_min,
#         x_max,
#         epsabs=epsabs,
#         epsrel=epsrel,
#         limit=300,
#     )
#
#     return num / den

def sigma_eff_from_table(nu_kms, sigma_table, p=5, n_grid=5000):
    """
    Stable K_p average using a fixed log-spaced velocity grid.
    """
    if nu_kms <= 0:
        return np.nan

    vmin = sigma_table.v_min
    vmax = sigma_table.v_max

    v = np.logspace(np.log10(vmin), np.log10(vmax), n_grid)
    sig = sigma_table(v)

    # Relative Maxwellian shape, constants cancel
    weight = v**(p + 2) * np.exp(-v**2 / (4.0 * nu_kms**2))

    num = np.trapz(sig * weight, v)
    den = np.trapz(weight, v)

    return num / den

def sigma_point_proxy_from_table(nu_kms, sigma_of_v, p=5):
    """
    Single-velocity proxy.

    For the p-weighted Maxwellian average, the integrand peak is roughly at

        v_peak = sqrt(2(p+2)) * nu.

    For p=5:
        v_peak = sqrt(14) * nu.

    This is useful only for intuition, not as the main benchmark.
    """
    if nu_kms <= 0:
        return np.nan

    v_peak = math.sqrt(2.0 * (p + 2.0)) * nu_kms
    return float(sigma_of_v(v_peak))


# ============================================================
# 4. Compute benchmark rows
# ============================================================

def compute_benchmark_rows_from_table(
    halos,
    sigma_table,
    rstar_mode="rs",
    p=5,
    epsrel=1e-8,
):
    """
    Compute sigma_eff(M) from a tabulated sigma(v_rel) model.

    Parameters
    ----------
    halos : list
        List of NFWProfile objects.

    sigma_table : TabulatedCrossSection
        Object giving sigma(v_rel)/m from the CSV file.

    rstar_mode : str or float
        - "rs": use r = r_s
        - float: use this physical radius in kpc

    p : int
        Velocity weighting. For heat conduction use p=5.

    Returns
    -------
    rows : list of dict
        One row per halo.
    """
    rows = []

    for halo in halos:

        if rstar_mode == "rs":
            r_star = get_rs_kpc(halo)
        elif isinstance(rstar_mode, (int, float)):
            r_star = float(rstar_mode)
        else:
            raise ValueError("rstar_mode must be 'rs' or a float in kpc")

        # Numerical velocity scale from your NFW profile
        nu_kpcGyr = sigma(halo, r_star)
        nu_num_kms = nu_kpcGyr * cfg.kpcGyr_to_kms

        # Analytic Gilman-style estimate
        nu_analytic_kms = sigma1D_NFW_analytic(halo)

        # Full K_p averages
        sig_eff_num = sigma_eff_from_table(
            nu_kms=nu_num_kms,
            sigma_table=sigma_table,
            p=p,
            # epsrel=epsrel,
        )

        sig_eff_analytic = sigma_eff_from_table(
            nu_kms=nu_analytic_kms,
            sigma_table=sigma_table,
            p=p,
            # epsrel=epsrel,
        )

        # Single-velocity proxies
        sig_point_num = sigma_point_proxy_from_table(
            nu_kms=nu_num_kms,
            sigma_of_v=sigma_table,
            p=p,
        )

        sig_point_analytic = sigma_point_proxy_from_table(
            nu_kms=nu_analytic_kms,
            sigma_of_v=sigma_table,
            p=p,
        )

        Mvir = getattr(halo, "M_vir", getattr(halo, "_M_vir", np.nan))
        con = getattr(halo, "con", getattr(halo, "_con", np.nan))

        rows.append({
            "halo": halo,
            "Mvir": float(Mvir),
            "log10Mvir": float(np.log10(Mvir)),
            "con": float(con),
            "r_star_kpc": float(r_star),
            "nu_num_kms": float(nu_num_kms),
            "nu_analytic_kms": float(nu_analytic_kms),
            "sig_eff_num": float(sig_eff_num),
            "sig_eff_analytic": float(sig_eff_analytic),
            "sig_point_num": float(sig_point_num),
            "sig_point_analytic": float(sig_point_analytic),
        })

    return rows


def find_mass_of_max_sigma(rows, sigma_key="sig_eff_num"):
    """
    Find the halo mass where selected sigma value is maximal.
    """
    if len(rows) == 0:
        raise ValueError("rows is empty")

    sigma_arr = np.array([row[sigma_key] for row in rows], dtype=float)

    if np.all(np.isnan(sigma_arr)):
        raise ValueError(f"All values in {sigma_key} are NaN")

    imax = np.nanargmax(sigma_arr)
    return rows[imax]


# ============================================================
# 5. Printing and plotting
# ============================================================

def print_benchmark_table(rows):
    print("\n=== Tabulated RSIDM mapped to effective constant sigma/m ===")

    header = (
        f"{'log10M':>8}  "
        f"{'Mvir':>12}  "
        f"{'c':>8}  "
        f"{'r*[kpc]':>10}  "
        f"{'nu_num':>10}  "
        f"{'nu_an':>10}  "
        f"{'K5(num)':>12}  "
        f"{'K5(an)':>12}  "
        f"{'point(num)':>12}  "
        f"{'point(an)':>12}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['log10Mvir']:8.3f}  "
            f"{row['Mvir']:12.4e}  "
            f"{row['con']:8.3f}  "
            f"{row['r_star_kpc']:10.4g}  "
            f"{row['nu_num_kms']:10.4f}  "
            f"{row['nu_analytic_kms']:10.4f}  "
            f"{row['sig_eff_num']:12.5g}  "
            f"{row['sig_eff_analytic']:12.5g}  "
            f"{row['sig_point_num']:12.5g}  "
            f"{row['sig_point_analytic']:12.5g}"
        )


def plot_sigma_v_table(sigma_table, savepath=None):
    """
    Plot input sigma(v_rel)/m table.
    """
    plt.figure(figsize=(8, 5.5))

    plt.plot(
        sigma_table.v_kms,
        sigma_table.sigma_m,
        lw=2,
        label=r"Input table: $\sigma(v_{\rm rel})/m$",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$v_{\rm rel}\,[{\rm km/s}]$")
    plt.ylabel(r"$\sigma(v_{\rm rel})/m\,[{\rm cm^2/g}]$")
    plt.title(r"Tabulated velocity-dependent cross section")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()


def plot_sigma_eff_vs_halo_mass(rows, savepath=None):
    """
    Plot effective sigma/m vs halo mass.
    """
    Mvir = np.array([row["Mvir"] for row in rows])

    sig_eff_num = np.array([row["sig_eff_num"] for row in rows])
    sig_eff_an = np.array([row["sig_eff_analytic"] for row in rows])

    sig_point_num = np.array([row["sig_point_num"] for row in rows])
    sig_point_an = np.array([row["sig_point_analytic"] for row in rows])

    plt.figure(figsize=(8, 5.5))

    plt.plot(
        Mvir,
        sig_eff_num,
        marker="o",
        lw=2,
        label=r"Full $K_5$ average, numerical $\nu(r_s)$",
    )

    plt.plot(
        Mvir,
        sig_eff_an,
        marker="^",
        lw=2,
        label=r"Full $K_5$ average, analytic $\sigma_{1D}$",
    )

    plt.plot(
        Mvir,
        sig_point_num,
        marker="s",
        ls="--",
        alpha=0.7,
        label=r"Point proxy, $v=\sqrt{14}\nu(r_s)$",
    )

    plt.plot(
        Mvir,
        sig_point_an,
        marker="d",
        ls="--",
        alpha=0.7,
        label=r"Point proxy, $v=\sqrt{14}\sigma_{1D}$",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$M_{\rm vir}\,[M_\odot]$")
    plt.ylabel(r"Effective $\sigma/m\,[{\rm cm^2/g}]$")
    plt.title(r"Tabulated RSIDM mapped onto constant-SIDM benchmark")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()


def plot_velocity_mapping(rows, savepath=None):
    """
    Plot characteristic velocities as a function of halo mass.
    This helps interpret which part of sigma(v) each halo probes.
    """
    Mvir = np.array([row["Mvir"] for row in rows])

    nu_num = np.array([row["nu_num_kms"] for row in rows])
    nu_an = np.array([row["nu_analytic_kms"] for row in rows])

    vpeak_num = np.sqrt(14.0) * nu_num
    vpeak_an = np.sqrt(14.0) * nu_an

    plt.figure(figsize=(8, 5.5))

    plt.plot(
        Mvir,
        vpeak_num,
        marker="o",
        lw=2,
        label=r"$v_{\rm peak}=\sqrt{14}\nu(r_s)$",
    )

    plt.plot(
        Mvir,
        vpeak_an,
        marker="^",
        lw=2,
        label=r"$v_{\rm peak}=\sqrt{14}\sigma_{1D}$",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$M_{\rm vir}\,[M_\odot]$")
    plt.ylabel(r"Characteristic $v_{\rm rel}\,[{\rm km/s}]$")
    plt.title(r"Velocity region probed by each halo mass")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# 6. Main example
# ============================================================

if __name__ == "__main__":

    # Path to your Gilman/collaborator table
    sigma_file = "Crossv_May7.csv"

    sigma_table = TabulatedCrossSection(
        sigma_file,
        log_interp=True,
        extrapolation="edge",
    )

    # First look at the input sigma(v_rel)
    plot_sigma_v_table(
        sigma_table,
        # savepath="input_sigma_vs_vrel.png",
    )

    # --------------------------------------------------------
    # Mass scan
    # --------------------------------------------------------

    # Option A: broad scan to see the shape
    # M_vir_arr = np.logspace(6.0, 10.5, num=120)

    # Option B: Gilman-style benchmark masses
    M_vir_arr = np.array([1e7, 10**7.5, 1e8, 10**8.5, 1e9])

    # Option C: focused scan around the expected peak
    # M_vir_arr = np.logspace(7.0, 8.5, num=80)

    # c_arr = aux.DMc_gravo(M_vir_arr, 0, cfg.const_h)
    c_arr = np.array([21.21, 19.81, 18.42, 17.05, 15.69])

    # other data we needed
    r200_arr = np.array([4.55, 6.68, 9.81, 14.4, 21.1])
    log_rho_s_arr = np.array([7.57, 7.50, 7.42, 7.33, 7.24])

    halos = [
        NFWProfile(_M_vir=M_vir_arr[i], _con=c_arr[i])
        # TruncatedNFWProfile(_M_vir=M_vir_arr[i], _con=c_arr[i])
        # TruncatedNFWProfile(_M_vir=M_vir_arr[i], _con=c_arr[i], _r200=r200_arr[i], _log_rho_s=log_rho_s_arr[i])
        for i in range(len(M_vir_arr))
    ]

    # for i in range(0,len(c_arr)):
    #     print("r200:", f"{halos[i].r200:.2f}", "log rho_s:", f"{np.log10(halos[i].rho_s):.2f}",
    #           "epsilon:",  f"{halos[i].eps_d:.2f}")

    rows = compute_benchmark_rows_from_table(
        halos,
        sigma_table=sigma_table,
        rstar_mode="rs",
        p=5,
        epsrel=1e-8,
    )

    print_benchmark_table(rows)

    best_row = find_mass_of_max_sigma(rows, sigma_key="sig_eff_num")

    print("\n=== Maximum effective cross section ===")
    print(f"Mvir                 = {best_row['Mvir']:.6e} Msun")
    print(f"log10(Mvir/Msun)     = {best_row['log10Mvir']:.6f}")
    print(f"concentration        = {best_row['con']:.4f}")
    print(f"r_star               = {best_row['r_star_kpc']:.6g} kpc")
    print(f"nu_num               = {best_row['nu_num_kms']:.6f} km/s")
    print(f"nu_analytic          = {best_row['nu_analytic_kms']:.6f} km/s")
    print(f"K5 sigma_eff num     = {best_row['sig_eff_num']:.6g} cm^2/g")
    print(f"K5 sigma_eff analytic= {best_row['sig_eff_analytic']:.6g} cm^2/g")
    print(f"point proxy num      = {best_row['sig_point_num']:.6g} cm^2/g")
    print(f"point proxy analytic = {best_row['sig_point_analytic']:.6g} cm^2/g")

    plot_sigma_eff_vs_halo_mass(
        rows,
        # savepath="sigma_eff_vs_Mvir_from_table.png",
    )

    plot_velocity_mapping(
        rows,
        # savepath="velocity_mapping_vs_Mvir.png",
    )