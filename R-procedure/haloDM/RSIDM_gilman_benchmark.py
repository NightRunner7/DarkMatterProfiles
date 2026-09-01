import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

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
        # sigma_1D in km/s.
    """
    Vmax = halo.Vmax  # kpc/Gyr
    KPC_PER_GYR_TO_KM_PER_S = 0.9777922217
    Vmax_km_s = Vmax * KPC_PER_GYR_TO_KM_PER_S  # km/s


    sigma_1D = Vmax_km_s*0.64

    rs = get_rs_kpc(halo)
    rho_s = halo.rho_s  # assumed M_sun / kpc^3
    G = 4.30091e-6      # kpc km^2 s^-2 M_sun^-1

    # Krzysztof results
    # sigma_1D = 1.10 * math.sqrt(G * rho_s * rs**2)

    # Luca results
    # sigma_1D = 1.05 * math.sqrt(G * rho_s * rs**2)

    return sigma_1D


def sigma1D_NFW_analytic_test(index=0):
    """
    Analytic estimate of characteristic 1D velocity dispersion for an NFW halo.

    Returns
    -------
    float
        sigma_1D in km/s.
    """
    rs_arr = [0.21, 0.34, 0.53,  0.84,  1.35]
    log_rho_s = [7.57, 7.50, 7.42, 7.33, 7.24]

    rs = rs_arr[index]
    rho_s = 10**(log_rho_s[index])  # assumed M_sun / kpc^3

    G = 4.30091e-6      # kpc km^2 s^-2 M_sun^-1

    return 1.10 * math.sqrt(G * rho_s * rs**2)
    # return 1.05 * math.sqrt(G * rho_s * rs**2)

nu_test = sigma1D_NFW_analytic_test()
print("M_vir=7.0, nu=", sigma1D_NFW_analytic_test(index=0))
print("M_vir=7.5, nu=", sigma1D_NFW_analytic_test(index=1))
print("M_vir=8.0, nu=", sigma1D_NFW_analytic_test(index=2))
print("M_vir=8.5, nu=", sigma1D_NFW_analytic_test(index=3))
print("M_vir=9.0, nu=", sigma1D_NFW_analytic_test(index=4))

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

def sigma_eff_from_table(
    nu_kms,
    sigma_table,
    p=5,
    n_grid=5000,
    x_min=1e-4,
    x_max=30.0,
):
    """
    Compute K_p over a broad range in x = v_rel / nu.
    """
    if nu_kms <= 0:
        return np.nan

    x = np.logspace(
        np.log10(x_min),
        np.log10(x_max),
        n_grid,
    )

    v_rel = nu_kms * x
    sig = sigma_table(v_rel)

    weight = x ** (p + 2) * np.exp(-x**2 / 4.0)

    num = np.trapz(sig * weight, x)
    den = np.trapz(weight, x)

    # num = np.trapezoid(sig * weight, x)
    # den = np.trapezoid(weight, x)

    return num / den

# def sigma_eff_from_table(nu_kms, sigma_table, p=5, n_grid=5000):
#     """
#     Stable K_p average using a fixed log-spaced velocity grid.
#     """
#     if nu_kms <= 0:
#         return np.nan
#
#     vmin = sigma_table.v_min
#     vmax = sigma_table.v_max
#
#     v = np.logspace(np.log10(vmin), np.log10(vmax), n_grid)
#     sig = sigma_table(v)
#
#     # Relative Maxwellian shape, constants cancel
#     weight = v**(p + 2) * np.exp(-v**2 / (4.0 * nu_kms**2))
#
#     num = np.trapz(sig * weight, v)
#     den = np.trapz(weight, v)
#
#     return num / den

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
    tabel_vol=False,
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
    index =0
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

        # ### SIGMA_1D VALUES GIVEN FROM LUCA
        nuValues = [1.9912, 2.1818, 2.3906, 2.6192, 2.8694, 3.1434, 3.4434,
                    3.7717, 4.1311, 4.5245, 4.9550, 5.4260, 5.9415, 6.5054, 7.1224,
                    7.7972, 8.5354, 10.2255, 11.1908, 12.2461, 13.3998, 14.6607]
        # nu_analytic_kms=nuValues[index]
        index=index+1

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
        rho_s = getattr(halo, "rho_s", getattr(halo, "_rho_s", np.nan))

        if tabel_vol:
            rows.append({
                "log10Mvir": float(np.log10(Mvir)),
                "con": float(con),
                "r_s_kpc": float(r_star),
                "log_rho_s": float(np.log10(rho_s)),
                "nu_analytic_kms": float(nu_analytic_kms),
                # "sig_eff_m": float(sig_eff_num),
                "sig_eff_m": float(sig_eff_analytic),

            })
        else:
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

def print_benchmark_table(rows, tabel_vol=False):
    print("\n=== Tabulated RSIDM mapped to effective constant sigma/m ===")

    if tabel_vol:
        sigma_label = r"$\sigma_{1D}$"

        header = (
            f"{'log10M':>8}  "
            f"{'c':>8}  "
            f"{'r[kpc]':>10}  "
            f"{'log10rho_s':>10}  "
            f"{sigma_label:>10}  "
            f"{'K5(an)':>12}  "
        )

        print(header)
        print("-" * len(header))

        for row in rows:
            print(
                f"{row['log10Mvir']:8.3f}  "
                f"{row['con']:8.3f}  "
                f"{row['r_s_kpc']:10.4g}  "
                f"{row['log_rho_s']:8.3f}  "
                f"{row['nu_analytic_kms']:12.5g}  "
                f"{row['sig_eff_m']:12.5g}  "
            )
    else:
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
    Plot input sigma(v_rel)/m table in a cleaner publication-style format.
    """

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    ax.plot(
        sigma_table.v_kms,
        sigma_table.sigma_m,
        lw=3.0,
        color="black",
        label=r"Input table (Camilo): $\sigma(v_{\rm rel})/m$",
        zorder=3,
    )

    ax.plot(
        points_v_values,
        points_sigma_values,
        "o",
        lw=3.0,
        color="red",
        label=r"Input table (Krzystzof): $\sigma(v_{\rm rel})/m$",
        zorder=3,
    )


    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"$v_{\rm rel}\ [{\rm km\,s^{-1}}]$",
        fontsize=15,
    )

    ax.set_ylabel(
        r"$\sigma(v_{\rm rel})/m\ [{\rm cm^2\,g^{-1}}]$",
        fontsize=15,
    )

    ax.set_title(
        r"Velocity-dependent self-interaction cross section",
        fontsize=16,
        pad=12,
    )

    # Tick styling
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=13,
        length=7,
        width=1.2,
        direction="in",
        top=True,
        right=True,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        length=4,
        width=1.0,
        direction="in",
        top=True,
        right=True,
    )

    # Better log minor ticks
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))

    # Grid
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.5,
        alpha=0.25,
    )

    # Legend
    ax.legend(
        fontsize=12,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        loc="best",
    )

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

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

    # --------------------------------------------------------
    # Load my points
    # --------------------------------------------------------
    data_file = "Gilma_plot_points.csv"

    v_values = []
    sigma_values = []

    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            x_str, y_str = line.split(",")

            # x = float(x_str.strip().replace(",", "."))
            # y = float(y_str.strip().replace(",", "."))
            x = float(x_str.strip())
            y = float(y_str.strip())


            v_values.append(x)
            sigma_values.append(y)

    points_v_values = np.array(v_values)
    points_sigma_values = np.array(sigma_values)

    # --------------------------------------------------------
    # Load Camilo points
    # --------------------------------------------------------

    # Path to your Gilman/collaborator table
    sigma_file = "Crossv_May7.csv"
    # sigma_file = "Gilma_plot_points.csv"

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
    # ### Option A: Gilman-style benchmark masses
    # M_vir_arr = np.logspace(6.0, 10.5, num=120)
    M_vir_arr = np.array([1e7, 10**7.5, 1e8, 10**8.5, 1e9])
    # M_vir_arr = np.array([1e7, 1e8])

    # c_arr = aux.DMc_gravo(M_vir_arr, 0, cfg.const_h)
    # c_arr = np.array([21.21, 18.42])
    c_arr = np.array([21.21,  19.81, 18.42, 17.05, 15.69])

    # other data we needed
    r200_arr = np.array([4.55, 6.68, 9.81, 14.4, 21.1])
    log_rho_s_arr = np.array([7.57, 7.50, 7.42, 7.33, 7.24])
    rs_kpc = np.array([0.21, 0.34, 0.53, 0.84, 1.35])

    # ### Option 0: Taking into account error in c-M relation
    M_vir_arr = np.array([1e7, 10**7.5, 1e8, 10**8.5, 1e9])
    c_arr = np.array([21.196, 19.816, 18.436, 17.056, 15.676])
    c_plus = 10**(0.16) * c_arr
    c_minus = 10**(-0.16) * c_arr
    print("c_plus:", c_plus)
    print("c_minus:", c_minus)

    # ### Option A: Gilman-style benchmark masses (extrapolation)
    # M_vir_arr = np.array([10**7.0, 10**7.5, 10**8.0, 10**8.5, 10**9.0])
    # c_arr = np.array([21.1960, 19.8160, 18.4360, 17.0560, 15.6760])

    # M_vir_arr = np.array([10**6.5, 10**6.625, 10**6.750, 10**6.875,
    #                       10**7.0, 10**7.125, 10**7.250, 10**7.375, 10**7.5, 10**7.625, 10**7.750, 10**7.875,
    #                       10**8.0, 10**8.125, 10**8.250, 10**8.375, 10**8.5, 10**8.625, 10**8.750, 10**8.875,
    #                       10**9.0])
    # c_arr = np.array([22.5760, 22.2310, 21.8860, 21.5410,
    #                   21.1960, 20.8510, 20.5060, 20.1610, 19.8160, 19.4710, 19.1260, 18.7810,
    #                   18.4360, 18.0910, 17.7460, 17.4010, 17.0560, 16.7110, 16.3660, 16.0210,
    #                   15.6760])

    # ### COMPARISON TO LUCA
    # M_vir_arr = [
    #     10**6.500, 10**6.625, 10**6.750, 10**6.875, 10**7.000,
    #     10**7.125, 10**7.250, 10**7.375, 10**7.500, 10**7.625,
    #     10**7.750, 10**7.875, 10**8.000, 10**8.125, 10**8.250,
    #     10**8.375, 10**8.500, 10**8.750, 10**8.875, 10**9.000,
    #     10**9.125, 10**9.250
    # ]

    # ### LUCA CONCENTRATION
    # c_arr = [
    #     22.590, 22.242, 21.895, 21.547, 21.200,
    #     20.852, 20.505, 20.157, 19.810, 19.462,
    #     19.115, 18.768, 18.420, 18.073, 17.725,
    #     17.378, 17.030, 16.335, 15.988, 15.640,
    #     15.293, 14.945
    # ]
    # ### MY CONCENTRATION
    # c_arr = [
    #     22.576, 22.231, 21.886, 21.541, 21.196,
    #     20.851, 20.506, 20.161, 19.816, 19.471,
    #     19.126, 18.781, 18.436, 18.091, 17.746,
    #     17.401, 17.056, 16.366, 16.021, 15.676,
    #     15.331, 14.986
    # ]

    # ### LUCA R_S
    # rs_kpc = [
    #     0.1379, 0.1541, 0.1723, 0.1927, 0.2156,
    #     0.2413, 0.2701, 0.3024, 0.3387, 0.3795,
    #     0.4253, 0.4767, 0.5346, 0.5998, 0.6731,
    #     0.7557, 0.8488, 1.0721, 1.2057, 1.3566,
    #     1.5271, 1.7200
    # ]
    # r200_arr = [rs_kpc[i]*c_arr[i] for i in range(0, len(c_arr))]
    # ### LUCA RHO_S
    # log_rho_s_arr = [
    #     7.6394, 7.6220, 7.6044, 7.5865, 7.5683,
    #     7.5498, 7.5310, 7.5120, 7.4926, 7.4729,
    #     7.4528, 7.4324, 7.4117, 7.3906, 7.3691,
    #     7.3472, 7.3249, 7.2789, 7.2553, 7.2311,
    #     7.2065, 7.1813
    # ]

    halos = [
        # NFWProfile(_M_vir=M_vir_arr[i], _con=c_arr[i])
        # TruncatedNFWProfile(_M_vir=M_vir_arr[i], _con=c_arr[i])
        # TruncatedNFWProfile(_M_vir=M_vir_arr[i], _con=c_arr[i], _r200=r200_arr[i], _log_rho_s=log_rho_s_arr[i])

        # Gilma c-base
        TruncatedNFWProfile(_M_vir=M_vir_arr[i], _con=c_minus[i])

        # Gilman t-base
        # TruncatedNFWProfile(_M_vir=M_vir_arr[i],
        #                     _con=c_arr[i],
        #                     _r200=r200_arr[i],
        #                     _r_s=rs_kpc[i],
        #                     _log_rho_s=log_rho_s_arr[i],
        #                     strict_consistency=False)
        for i in range(len(M_vir_arr))
    ]

    # for i in range(0,len(c_arr)):
    #     print("r200:", f"{halos[i].r200:.2f}", "log rho_s:", f"{np.log10(halos[i].rho_s):.2f}",
    #           "epsilon:",  f"{halos[i].eps_d:.2f}")

    rows = compute_benchmark_rows_from_table(
        halos,
        sigma_table=sigma_table,
        rstar_mode="rs",
        tabel_vol=True,
        p=5,
        epsrel=1e-8,
    )

    print_benchmark_table(rows, tabel_vol=True)

    # best_row = find_mass_of_max_sigma(rows, sigma_key="sig_eff_num")
    #
    # print("\n=== Maximum effective cross section ===")
    # print(f"Mvir                 = {best_row['Mvir']:.6e} Msun")
    # print(f"log10(Mvir/Msun)     = {best_row['log10Mvir']:.6f}")
    # print(f"concentration        = {best_row['con']:.4f}")
    # print(f"r_star               = {best_row['r_star_kpc']:.6g} kpc")
    # print(f"nu_num               = {best_row['nu_num_kms']:.6f} km/s")
    # print(f"nu_analytic          = {best_row['nu_analytic_kms']:.6f} km/s")
    # print(f"K5 sigma_eff num     = {best_row['sig_eff_num']:.6g} cm^2/g")
    # print(f"K5 sigma_eff analytic= {best_row['sig_eff_analytic']:.6g} cm^2/g")
    # print(f"point proxy num      = {best_row['sig_point_num']:.6g} cm^2/g")
    # print(f"point proxy analytic = {best_row['sig_point_analytic']:.6g} cm^2/g")
    #
    # plot_sigma_eff_vs_halo_mass(
    #     rows,
    #     # savepath="sigma_eff_vs_Mvir_from_table.png",
    # )
    #
    # plot_velocity_mapping(
    #     rows,
    #     # savepath="velocity_mapping_vs_Mvir.png",
    # )

    for i in range(0,len(halos)):
        tau_kappa = halos[i].tau(beta=0.85, sigma_eff=float(rows[i]['sig_eff_m']))
        print("tau_kappa:", tau_kappa)
        # print("sigma_eff:", float(rows[i]['sig_eff_m']))