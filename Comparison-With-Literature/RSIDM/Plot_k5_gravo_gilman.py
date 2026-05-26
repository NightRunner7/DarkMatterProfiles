import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import units as uni
import numpy as np
import matplotlib as mpl
from RSIMD import build_sigma_m_eff_p5_interpolator
import config as cfg
from scipy.interpolate import interp1d
# ##################################### SETTINGS ##################################### #
plot_save = False
output_plot_dir = "Results"

OUR_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_cSIDM = os.path.join(OUR_DIR, 'Data_cSIDM', "Gilman_benchmark_2")
# DATA_rSIDM = os.path.join(OUR_DIR, 'Data_rSIDM')
DATA_rSIDM = os.path.join(OUR_DIR, 'Data_rSIDM', "Gilman_benchmark_2")

# --- select files: cSIDM
SIDM_file_gilman  = "CSIDM_Gilman_M7.0_c21.21_sigma24.95_beta0.85.csv"
SIDM_file_1  = "CSIDM_Gilman_M7.0_c21.21_sigma30.545_beta0.85.csv"
SIDM_file_2  = "CSIDM_Gilman_M7.5_c19.81_sigma46.855_beta0.85.csv"
SIDM_file_3  = "CSIDM_Gilman_M8.0_c18.42_sigma28.378_beta0.85.csv"
SIDM_file_4  = "CSIDM_Gilman_M8.5_c17.05_sigma14.986_beta0.85.csv"
SIDM_file_5  = "CSIDM_Gilman_M9.0_c15.69_sigma9.19_beta0.85.csv"

SIDM_file_arr = [SIDM_file_1, SIDM_file_2, SIDM_file_3, SIDM_file_4, SIDM_file_5]

# --- select files: rSIDM
RSIDM_file_1  = "RSIDM_Gilman_M7.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_2  = "RSIDM_Gilman_M7.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_3  = "RSIDM_Gilman_M8.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_4  = "RSIDM_Gilman_M8.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_5  = "RSIDM_Gilman_M9.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"

RSIDM_file_arr = [RSIDM_file_1, RSIDM_file_2, RSIDM_file_3, RSIDM_file_4, RSIDM_file_5]

# --- choose of file
select_file = 0

# --- SELECT SIDM
SIDM_file = SIDM_file_arr[select_file]

# --- SELECT RSIDM
RSIDM_file = RSIDM_file_arr[select_file]

# --- select parameters values
# sigma_m_arr = [35.1, 45.8, 27.1, 14.7, 9.15]
sigma_m_arr = [30.545, 46.855, 28.378, 14.986, 9.19]
sigma_m=sigma_m_arr[select_file]

M_power_arr = [7.0, 7.5, 8.0, 8.5, 9.0]
M_power = M_power_arr[select_file]
# ##################################### SET DATA TO CLASS ############################################################ #
# --- SIDM
gravoEvolution_SIDM = gravoF.create_gravothermalData_from_file(SIDM_file, DATA_cSIDM, beta=0.75)
gravoEvolution_SIDM.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM.put_extra_parameters(10 ** M_power, sigma_m)

# --- RSIDM
gravoEvolution_RSIDM = gravoF.create_gravothermalData_from_file(RSIDM_file, DATA_rSIDM, beta=0.75)
gravoEvolution_RSIDM.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM.put_extra_parameters(10**M_power, sigma_m)

# _, time_collapse, time_step_collapse = gravoEvolution_SIDM.find_collapse(elements=2, fixed_limit=10 ** 10)
# print(f"time collapse: {time_collapse} [Gyr], time step: {time_step_collapse}")

# ##################################### TAKE SIGMA FROM FILE ######################################################### #
def build_sigma_m_eff_p5_interpolator_from_table(
    csv_file,
    *,
    nu_kms_min=0.05,
    nu_kms_max=1.0e4,
    N_nu=5000,
    N_v=8000,
    extrapolation="edge",
):
    """
    Build K5-averaged effective sigma/m interpolator from a tabulated
    velocity-dependent cross section.

    Input CSV:
        column 1: v_rel [km/s]
        column 2: sigma(v_rel)/m [cm^2/g]

    Returns
    -------
    sigma_eff_from_nu_kpcGyr : callable
        Input nu in kpc/Gyr.

    sigma_eff_from_nu_kms : callable
        Input nu in km/s.

    (nu_grid_kms, sigma_eff_grid) : tuple
        Grid used for plotting K5(nu).
    """

    data = np.loadtxt(csv_file, delimiter=",")

    v_data = np.asarray(data[:, 0], dtype=float)
    sig_data = np.asarray(data[:, 1], dtype=float)

    mask = (
        np.isfinite(v_data)
        & np.isfinite(sig_data)
        & (v_data > 0)
        & (sig_data > 0)
    )

    v_data = v_data[mask]
    sig_data = sig_data[mask]

    order = np.argsort(v_data)
    v_data = v_data[order]
    sig_data = sig_data[order]

    v_min = float(v_data[0])
    v_max = float(v_data[-1])

    # log-log interpolation of raw sigma(v_rel)
    logv_data = np.log(v_data)
    logsig_data = np.log(sig_data)

    if extrapolation == "edge":
        raw_interp = interp1d(
            logv_data,
            logsig_data,
            kind="linear",
            bounds_error=False,
            fill_value=(logsig_data[0], logsig_data[-1]),
        )
    elif extrapolation == "extrapolate":
        raw_interp = interp1d(
            logv_data,
            logsig_data,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
    else:
        raise ValueError("extrapolation must be 'edge' or 'extrapolate'")

    def sigma_raw_from_v_kms(v_kms):
        v = np.asarray(v_kms, dtype=float)
        v = np.maximum(v, 1.0e-300)
        return np.exp(raw_interp(np.log(v)))

    # fixed velocity grid for stable K5 averaging
    v_grid = np.logspace(np.log10(v_min), np.log10(v_max), N_v)
    sig_grid_raw = sigma_raw_from_v_kms(v_grid)

    def sigma_eff_single_nu(nu_kms):
        """
        K5 average:

            <sigma(v_rel) v_rel^5> / <v_rel^5>

        with relative Maxwellian kernel:

            f_rel ~ v_rel^2 exp[-v_rel^2/(4 nu^2)]

        Therefore the integration weight in dv is:

            v_rel^7 exp[-v_rel^2/(4 nu^2)]
        """
        if not np.isfinite(nu_kms) or nu_kms <= 0:
            return np.nan

        weight = v_grid**7 * np.exp(-v_grid**2 / (4.0 * nu_kms**2))

        den = np.trapz(weight, v_grid)
        if den <= 0 or not np.isfinite(den):
            return np.nan

        num = np.trapz(sig_grid_raw * weight, v_grid)
        return num / den

    # build K5(nu) grid
    nu_grid_kms = np.logspace(
        np.log10(nu_kms_min),
        np.log10(nu_kms_max),
        N_nu,
    )

    sigma_eff_grid = np.array(
        [sigma_eff_single_nu(nu) for nu in nu_grid_kms],
        dtype=float,
    )

    good = (
        np.isfinite(nu_grid_kms)
        & np.isfinite(sigma_eff_grid)
        & (nu_grid_kms > 0)
        & (sigma_eff_grid > 0)
    )

    nu_grid_kms = nu_grid_kms[good]
    sigma_eff_grid = sigma_eff_grid[good]

    lognu_grid = np.log(nu_grid_kms)
    logsig_eff_grid = np.log(sigma_eff_grid)

    sigma_eff_interp = interp1d(
        lognu_grid,
        logsig_eff_grid,
        kind="linear",
        bounds_error=False,
        fill_value=(logsig_eff_grid[0], logsig_eff_grid[-1]),
    )

    def sigma_eff_from_nu_kms(nu_kms):
        nu = np.asarray(nu_kms, dtype=float)
        nu = np.maximum(nu, 1.0e-300)
        return np.exp(sigma_eff_interp(np.log(nu)))

    def sigma_eff_from_nu_kpcGyr(nu_kpcGyr):
        nu_kms = np.asarray(nu_kpcGyr, dtype=float) * cfg.kpcGyr_to_kms
        return sigma_eff_from_nu_kms(nu_kms)

    return sigma_eff_from_nu_kpcGyr, sigma_eff_from_nu_kms, (
        nu_grid_kms,
        sigma_eff_grid,
    )

# ##################################### PLOTTING COMPARISON ########################################################## #
def extract_central_velocity_track(gravo, *, stop_at_collapse=True, n_core_bins=1):
    tarr = np.asarray(gravo.data["time-no-repetition"], dtype=float)
    tarr = tarr[np.isfinite(tarr)]
    tarr = tarr[tarr > 0]

    if stop_at_collapse:
        try:
            _, t_collapse, _ = gravo.find_collapse(elements=2, fixed_limit=10**10)
            tarr = tarr[tarr <= t_collapse]
        except Exception:
            pass

    nu0_code = gravo._nu0_rs_per_gyr()   # effectively [1/Gyr] or "r_s/Gyr scale"
    r_s_kpc = gravo.parameters["r_s"]    # [kpc]

    times_used = []
    nu_core_hat = []
    seen_steps = set()

    for t in tarr:
        r_hat, nu_hat, step, t_val = gravo.return_veldis_hat_profile(
            time_argument=float(t),
            time_step_bool=False,
            return_time_step=True,
        )

        if int(step) in seen_steps:
            continue
        seen_steps.add(int(step))

        nu_hat = np.asarray(nu_hat, dtype=float)
        nu_hat = nu_hat[np.isfinite(nu_hat)]
        if len(nu_hat) == 0:
            continue

        n_use = min(max(1, n_core_bins), len(nu_hat))
        nu_core_hat.append(np.mean(nu_hat[:n_use]))
        times_used.append(t_val)

    times_gyr = np.asarray(times_used, dtype=float)
    nu_core_hat = np.asarray(nu_core_hat, dtype=float)

    nu_core_kpcGyr = nu_core_hat * nu0_code * r_s_kpc
    nu_core_kms = nu_core_kpcGyr * cfg.kpcGyr_to_kms

    return times_gyr, nu_core_hat, nu_core_kpcGyr, nu_core_kms

def plot_core_track_on_resonance(
    nu_grid_kms,
    sigma_grid,
    times_gyr,
    nu_core_kms,
    sigma_eff_from_nu_kms,
    *,
    title=r"Core trajectory on the resonant $K_5(\nu)$ profile",
    cmap_name="viridis",
    # figsize=(9.0, 6.2),
    figsize=(10.8, 6.8),

):
    """
    Plot the static resonant K5(nu) curve and overlay the time evolution
    of the core velocity dispersion.
    """
    K_core = sigma_eff_from_nu_kms(nu_core_kms)

    fig, ax = plt.subplots(figsize=figsize)

    # static background curve
    ax.loglog(nu_grid_kms, sigma_grid, lw=2.4, color="black", zorder=1)

    # trajectory line
    ax.plot(nu_core_kms, K_core, lw=1.2, alpha=0.6, color="tab:green", zorder=2)

    # colored by log10 time
    cvals = np.log10(times_gyr)
    norm = mpl.colors.Normalize(vmin=float(np.min(cvals)), vmax=float(np.max(cvals)))
    cmap = mpl.cm.get_cmap(cmap_name)

    sc = ax.scatter(
        nu_core_kms,
        K_core,
        c=cvals,
        cmap=cmap,
        norm=norm,
        s=36,
        edgecolors="none",
        zorder=3,
    )

    # start / end markers
    ax.scatter(
        nu_core_kms[0], K_core[0],
        s=95, marker="o",
        facecolors="none", edgecolors="red",
        linewidths=1.8, zorder=4, label="start"
    )
    ax.scatter(
        nu_core_kms[-1], K_core[-1],
        s=95, marker="s",
        facecolors="none", edgecolors="blue",
        linewidths=1.8, zorder=4, label="end"
    )

    # dashed line
    x_position = 2.0
    y_position = 1.2  # 0.55
    ax.axhline(y=sigma_m, linestyle='--', color='grey', alpha=0.7, lw=2.4)
    ax.text(
        x_position, sigma_m * y_position,
        r'$\langle \sigma/m \rangle_{\mathrm{eff}}$',
        fontsize=14,
        color='grey'
    )

    ax.set_xlabel(r"$\nu_{\rm core}\ [\mathrm{km/s}]$", fontsize=16)
    ax.set_ylabel(r"$K_5(\nu)\ [\mathrm{cm^2/g}]$", fontsize=16)
    ax.set_title(title, fontsize=18, pad=10)

    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=12)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(t\,[\mathrm{Gyr}])$", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    ax.legend(fontsize=12, frameon=True)

    plt.tight_layout()
    plt.show()

    return fig, ax

# Build resonant curve
sigma_profile_file = os.path.join(OUR_DIR, "Crossv_May7.csv")

sigma_eff_from_nu_kpcGyr, sigma_eff_from_nu_kms, (nu_grid, sigma_grid) = \
    build_sigma_m_eff_p5_interpolator_from_table(
        sigma_profile_file,
        nu_kms_min=0.05,
        nu_kms_max=8000.0,
        N_nu=5000,
        N_v=8000,
        extrapolation="edge",
    )

# Extract core track
time_core, nu_core_hat, nu_core_rs_per_gyr, nu_core_kms = extract_central_velocity_track(
    gravoEvolution_RSIDM,
    stop_at_collapse=False,
    n_core_bins=2,   # or 1 if you prefer strictly central
)

# Plot
plot_core_track_on_resonance(
    nu_grid,
    sigma_grid,
    time_core,
    nu_core_kms,
    sigma_eff_from_nu_kms,
    title=rf"RSIDM: core trajectory on the resonant $K_5(\nu)$ profile, $M=10^{{{M_power}}}\,M_\odot$"
)