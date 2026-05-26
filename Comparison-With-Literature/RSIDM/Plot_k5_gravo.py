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
# ##################################### SETTINGS ##################################### #
plot_save = False
output_plot_dir = "Results"

OUR_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_cSIDM = os.path.join(OUR_DIR, 'Data_cSIDM', "New_benchmark_1")
# DATA_rSIDM = os.path.join(OUR_DIR, 'Data_rSIDM')
DATA_rSIDM = os.path.join(OUR_DIR, 'Data_rSIDM', "New_benchmark_1")

# --- select files: cSIDM
# SIDM_file_1  = "CSIDM_M8.4_sigma0.0106_beta0.75.csv"
# SIDM_file_2  = "CSIDM_M8.6_sigma0.421_beta0.75.csv"
# SIDM_file_3  = "CSIDM_M8.8_sigma15.2_beta0.75.csv"
# SIDM_file_4  = "CSIDM_M9.0_sigma176_beta0.75.csv"
# SIDM_file_5  = "CSIDM_M9.2_sigma845_beta0.75.csv"
# SIDM_file_6  = "CSIDM_M9.4_sigma2095_beta0.75.csv"
# SIDM_file_7  = "CSIDM_M9.6_sigma3148_beta0.75.csv"
# SIDM_file_8  = "CSIDM_M9.8_sigma3345_beta0.75.csv"
# SIDM_file_9  = "CSIDM_M10.0_sigma2500_beta0.75.csv"
# SIDM_file_10 = "CSIDM_M11.0_sigma63.86_beta0.75.csv"

SIDM_file_1  = "CSIDM_M8.0_sigma0.155_beta0.75.csv"
SIDM_file_2  = "CSIDM_M8.2_sigma0.64_beta0.75.csv"
SIDM_file_3  = "CSIDM_M8.4_sigma5.9_beta0.75.csv"
SIDM_file_4  = "CSIDM_M8.6_sigma27.9_beta0.75.csv"
SIDM_file_5  = "CSIDM_M8.8_sigma69.3_beta0.75.csv"
SIDM_file_6  = "CSIDM_M9.0_sigma104.0_beta0.75.csv"
SIDM_file_7  = "CSIDM_M9.2_sigma108.0_beta0.75.csv"
SIDM_file_8  = "CSIDM_M9.4_sigma84.4_beta0.75.csv"
SIDM_file_9  = "CSIDM_M9.6_sigma52.9_beta0.75.csv"
SIDM_file_10 = "CSIDM_M9.8_sigma28.2_beta0.75.csv"
SIDM_file_11 = "CSIDM_M10.0_sigma13.3_beta0.75.csv"
SIDM_file_12 = "CSIDM_M10.2_sigma5.75_beta0.75.csv"

SIDM_file_arr = [SIDM_file_1, SIDM_file_2, SIDM_file_3, SIDM_file_4, SIDM_file_5,
                 SIDM_file_6, SIDM_file_7, SIDM_file_8, SIDM_file_9, SIDM_file_10,
                 SIDM_file_11, SIDM_file_12]

# --- select files: rSIDM
# RSIDM_file_1  = "RSIDM_M8.4_beta0.75_Nradi400.csv"
# RSIDM_file_2  = "RSIDM_M8.6_beta0.75_Nradi400.csv"
# RSIDM_file_3  = "RSIDM_M8.8_beta0.75_Nradi400.csv"
# RSIDM_file_4  = "RSIDM_M9.0_beta0.75_Nradi400.csv"
# RSIDM_file_5  = "RSIDM_M9.2_beta0.75_Nradi400.csv"
# RSIDM_file_6  = "RSIDM_M9.4_beta0.75_Nradi400.csv"
# RSIDM_file_7  = "RSIDM_M9.6_beta0.75_Nradi400.csv"
# RSIDM_file_8  = "RSIDM_M9.8_beta0.75_Nradi400.csv"
# RSIDM_file_9  = "RSIDM_M10.0_beta0.75_Nradi400.csv"
# RSIDM_file_10 = "RSIDM_M11.0_beta0.75_Nradi400.csv"

# RSIDM_file_1  = "RSIDM_M8.4_beta0.75_Nradi700.csv"
# RSIDM_file_2  = "RSIDM_M8.6_beta0.75_Nradi700.csv"
# RSIDM_file_3  = "RSIDM_M8.8_beta0.75_Nradi700.csv"
# RSIDM_file_4  = "RSIDM_M9.0_beta0.75_Nradi700.csv"
# RSIDM_file_5  = "RSIDM_M9.2_beta0.75_Nradi700.csv"
# RSIDM_file_6  = "RSIDM_M9.4_beta0.75_Nradi700.csv"
# RSIDM_file_7  = "RSIDM_M9.6_beta0.75_Nradi700.csv"
# RSIDM_file_8  = "RSIDM_M9.8_beta0.75_Nradi700.csv"
# RSIDM_file_9  = "RSIDM_M10.0_beta0.75_Nradi700.csv"
# RSIDM_file_10 = "RSIDM_M11.0_beta0.75_Nradi700.csv"

RSIDM_file_1  = "RSIDM_M8.0_beta0.75_RadiiPerDec100_Ndec5.0deltaT-4.0.csv"
RSIDM_file_2  = "RSIDM_M8.2_beta0.75_RadiiPerDec100_Ndec5.0deltaT-4.0.csv"
RSIDM_file_3  = "RSIDM_M8.4_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_4  = "RSIDM_M8.6_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_5  = "RSIDM_M8.8_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_6  = "RSIDM_M9.0_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_7  = "RSIDM_M9.2_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_8  = "RSIDM_M9.4_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_9  = "RSIDM_M9.6_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_10 = "RSIDM_M9.8_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_11 = "RSIDM_M10.0_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_12 = "RSIDM_M10.2_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"


RSIDM_file_arr = [RSIDM_file_1, RSIDM_file_2, RSIDM_file_3, RSIDM_file_4, RSIDM_file_5,
                  RSIDM_file_6, RSIDM_file_7, RSIDM_file_8, RSIDM_file_9, RSIDM_file_10,
                  RSIDM_file_11, RSIDM_file_12]

# --- choose of file
select_file = 11

# --- SELECT SIDM
SIDM_file = SIDM_file_arr[select_file]

# --- SELECT RSIDM
RSIDM_file = RSIDM_file_arr[select_file]

# --- select parameters values
# sigma_m_arr = [0.0106, 0.421, 15.2, 176, 845, 2095, 3148, 3345, 2500, 63.86]
# sigma_m_arr = [0.155, 0.64, 5.9, 27.9, 69.3, 104.0, 108.0, 84.4, 52.9, 28.2, 13.3, 5.75]
sigma_m_arr = [35.1, 45.8, 27.1, 14.7, 9.15]
sigma_m=sigma_m_arr[select_file]

# M_power_arr = [8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 11.0]
# M_power_arr = [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2]
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
sigma_eff_from_nu_kpcGyr, sigma_eff_from_nu_kms, (nu_grid, sigma_grid) = \
    build_sigma_m_eff_p5_interpolator(
        # sigma_m0=0.008,
        # m_GeV=0.02,
        # L=0,
        # Gamma=6e-12,
        # vR_kms=85.0,
        # nu_kms_min=2,
        # nu_kms_max=8000.0,
        # N=5000
        sigma_m0=0.141,
        m_GeV=0.0301,
        L=0,
        Gamma=1.93e-13,
        vR_kms=55.8,
        nu_kms_min=2,
        nu_kms_max=8000.0,
        N=5000
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