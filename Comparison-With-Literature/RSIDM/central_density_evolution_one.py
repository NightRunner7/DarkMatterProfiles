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
beta=0.85

OUR_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_cSIDM = os.path.join(OUR_DIR, 'Data_cSIDM/Gilman_benchmark_3')
DATA_rSIDM = os.path.join(OUR_DIR, 'Data_rSIDM')
DATA_rSIDM_nr700 = os.path.join(OUR_DIR, "Data_rSIDM", "Gilman_benchmark_3")
DATA_rSIDM_nr700_vol2 = os.path.join(OUR_DIR, "Data_rSIDM", "Gilman_benchmark_2")
DATA_rSIDM_impact = os.path.join(OUR_DIR, 'Data-impact-initial-parameters')

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

SIDM_file_gilman  = "CSIDM_Gilman_M7.0_c21.21_sigma24.95_beta0.85.csv"
SIDM_file_1  = "CSIDM_Gilman_M7.0_c21.21_sigma30.545_beta0.85.csv"
SIDM_file_2  = "CSIDM_Gilman_M7.5_c19.81_sigma46.855_beta0.85.csv"
SIDM_file_3  = "CSIDM_Gilman_M8.0_c18.42_sigma28.378_beta0.85.csv"
SIDM_file_4  = "CSIDM_Gilman_M8.5_c17.05_sigma14.986_beta0.85.csv"
SIDM_file_5  = "CSIDM_Gilman_M9.0_c15.69_sigma9.19_beta0.85.csv"


SIDM_file_arr = [SIDM_file_1, SIDM_file_2, SIDM_file_3, SIDM_file_4, SIDM_file_5,
                 SIDM_file_6, SIDM_file_7, SIDM_file_8, SIDM_file_9, SIDM_file_10,
                 SIDM_file_11, SIDM_file_12]
# --- select files: rSIDM
RSIDM_file_1  = "RSIDM_M8.4_beta0.75_Nradi400.csv"
RSIDM_file_2  = "RSIDM_M8.6_beta0.75_Nradi400.csv"
RSIDM_file_3  = "RSIDM_M8.8_beta0.75_Nradi400.csv"
RSIDM_file_4  = "RSIDM_M9.0_beta0.75_Nradi400.csv"
RSIDM_file_5  = "RSIDM_M9.2_beta0.75_Nradi400.csv"
RSIDM_file_6  = "RSIDM_M9.4_beta0.75_Nradi400.csv"
RSIDM_file_7  = "RSIDM_M9.6_beta0.75_Nradi400.csv"
RSIDM_file_8  = "RSIDM_M9.8_beta0.75_Nradi400.csv"
RSIDM_file_9  = "RSIDM_M10.0_beta0.75_Nradi400.csv"
RSIDM_file_10 = "RSIDM_M11.0_beta0.75_Nradi400.csv"

# RSIDM_file_1_nr700  = "RSIDM_M8.4_beta0.75_Nradi700.csv"
# RSIDM_file_2_nr700  = "RSIDM_M8.6_beta0.75_Nradi700.csv"
# RSIDM_file_3_nr700  = "RSIDM_M8.8_beta0.75_Nradi700.csv"
# RSIDM_file_4_nr700  = "RSIDM_M9.0_beta0.75_Nradi700.csv"
# RSIDM_file_5_nr700  = "RSIDM_M9.2_beta0.75_Nradi700.csv"
# RSIDM_file_6_nr700  = "RSIDM_M9.4_beta0.75_Nradi700.csv"
# RSIDM_file_7_nr700  = "RSIDM_M9.6_beta0.75_Nradi700.csv"
# RSIDM_file_8_nr700  = "RSIDM_M9.8_beta0.75_Nradi700.csv"
# RSIDM_file_9_nr700  = "RSIDM_M10.0_beta0.75_Nradi700.csv"
# RSIDM_file_10_nr700 = "RSIDM_M11.0_beta0.75_Nradi700.csv"

RSIDM_file_1_nr700  = "RSIDM_M8.0_beta0.75_RadiiPerDec100_Ndec5.0deltaT-4.0.csv"
RSIDM_file_2_nr700  = "RSIDM_M8.2_beta0.75_RadiiPerDec100_Ndec5.0deltaT-4.0.csv"
RSIDM_file_3_nr700  = "RSIDM_M8.4_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_4_nr700  = "RSIDM_M8.6_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_5_nr700  = "RSIDM_M8.8_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_6_nr700  = "RSIDM_M9.0_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_7_nr700  = "RSIDM_M9.2_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_8_nr700  = "RSIDM_M9.4_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_9_nr700  = "RSIDM_M9.6_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_10_nr700 = "RSIDM_M9.8_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_11_nr700 = "RSIDM_M10.0_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"
RSIDM_file_12_nr700 = "RSIDM_M10.2_beta0.75_RadiiPerDec100_Ndec5.0_deltaT-4.0.csv"

RSIDM_file_1_nr700  = "RSIDM_Gilman_M7.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_2_nr700  = "RSIDM_Gilman_M7.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_3_nr700  = "RSIDM_Gilman_M8.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_4_nr700  = "RSIDM_Gilman_M8.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_5_nr700  = "RSIDM_Gilman_M9.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"

RSIDM_file_arr = [RSIDM_file_1, RSIDM_file_2, RSIDM_file_3, RSIDM_file_4, RSIDM_file_5,
                  RSIDM_file_6, RSIDM_file_7, RSIDM_file_8, RSIDM_file_9, RSIDM_file_10]

RSIDM_file_nr700_arr = [RSIDM_file_1_nr700, RSIDM_file_2_nr700, RSIDM_file_3_nr700, RSIDM_file_4_nr700,
                        RSIDM_file_5_nr700, RSIDM_file_6_nr700, RSIDM_file_7_nr700, RSIDM_file_8_nr700,
                        RSIDM_file_9_nr700, RSIDM_file_10_nr700, RSIDM_file_11_nr700, RSIDM_file_12_nr700]

# --- choose of file
select_file = 0
elements = 3

# --- SELECT SIDM
SIDM_file = SIDM_file_arr[select_file]

# --- SELECT RSIDM
# RSIDM_file = RSIDM_file_arr[select_file]
RSIDM_file_nr700 = RSIDM_file_nr700_arr[select_file]

# --- select parameters values
# sigma_m_arr = [0.0106, 0.421, 15.2, 176, 845, 2095, 3148, 3345, 2500, 63.86]
# sigma_m_arr = [0.155, 0.64, 5.9, 27.9, 69.3, 104.0, 108.0, 84.4, 52.9, 28.2, 13.3, 5.75]
sigma_m_arr = [30.545, 46.855, 28.378, 14.986, 9.19]
sigma_m=sigma_m_arr[select_file]

# M_power_arr = [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2]
M_power_arr = [7.0, 7.5, 8.0, 8.5, 9.0]
M_power = M_power_arr[select_file]

# ##################################### SET DATA TO CLASS ############################################################ #
# --- SIDM
gravoEvolution_SIDM = gravoF.create_gravothermalData_from_file(SIDM_file, DATA_cSIDM, beta=beta)
gravoEvolution_SIDM.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM.put_extra_parameters(10 ** M_power, sigma_m)

gravoEvolution_SIDM_gilman = gravoF.create_gravothermalData_from_file(SIDM_file_gilman, DATA_cSIDM, beta=beta)
gravoEvolution_SIDM_gilman.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM_gilman.put_extra_parameters(10 ** 7.0, 24.95)


# --- RSIDM
# # r_min=1e-2
# gravoEvolution_RSIDM = gravoF.create_gravothermalData_from_file(RSIDM_file, DATA_rSIDM, beta=0.75)
# gravoEvolution_RSIDM.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
# gravoEvolution_RSIDM.put_extra_parameters(10**M_power, sigma_m)
# # r_min=1e-4
# gravoEvolution_RSIDM_nr700 = gravoF.create_gravothermalData_from_file(RSIDM_file_nr700, DATA_rSIDM_nr700, beta=0.75)
# gravoEvolution_RSIDM_nr700.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
# gravoEvolution_RSIDM_nr700.put_extra_parameters(10**M_power, sigma_m)
# # various
# file_arr = ["RSIDM_M9.4_beta0.75_RadiiPerDec100_Ndec5.0deltaT-4.0.csv",
#             "RSIDM_M9.4_beta0.75_RadiiPerDec80_Ndec5.0deltaT-4.0.csv",
#             "RSIDM_M9.4_beta0.75_RadiiPerDec60_Ndec5.0deltaT-4.0.csv",
#             "RSIDM_M9.4_beta0.75_RadiiPerDec100_Ndec5.0deltaT-3.0.csv",
#             "RSIDM_M9.4_beta0.75_RadiiPerDec80_Ndec5.0deltaT-3.0.csv",
#             "RSIDM_M9.4_beta0.75_RadiiPerDec60_Ndec5.0deltaT-3.0.csv"
#             ]
# file = file_arr[1]
# gravoEvolution_RSIDM_Impact = gravoF.create_gravothermalData_from_file(file, DATA_rSIDM_impact, beta=0.75)
# gravoEvolution_RSIDM_Impact.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
# gravoEvolution_RSIDM_Impact.put_extra_parameters(10**M_power, sigma_m)
#

# r_min=1e-3
gravoEvolution_RSIDM = gravoF.create_gravothermalData_from_file(RSIDM_file_nr700, DATA_rSIDM_nr700, beta=beta)
gravoEvolution_RSIDM.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM.put_extra_parameters(10**M_power, sigma_m)


# _, time_collapse, time_step_collapse = gravoEvolution_SIDM.find_collapse(elements=elements, fixed_limit=10 ** 10)
# print(f"time collapse: {time_collapse} [Gyr], time step: {time_step_collapse}")

# ##################################### PLOTTING COMPARISON ########################################################## #
def rescale_time(model, _sigma_eff, beta=0.75):
    r_s = model.parameters["r_s"]
    rho_s = model.parameters["rho_s"]
    sigma_m_SI = _sigma_eff * 1e-4 * 1e3  # Convert from [cm^2/g] to [m^2/kg]
    sigma_m_SU = sigma_m_SI * cfg.kpc_SI ** -2 * cfg.M_solar_SI  # Convert to [kpc^2 * M_sun^-1]
    G_SU = cfg.const_G_starUnits
    time_c = (150/beta) * (1/(r_s * rho_s) * 1/sigma_m_SU) * (4 * np.pi * G_SU * rho_s) ** (-1/2)
    return time_c

def apply_logx_linear_y_grid(_ax):
    """
    Apply grid and tick settings for:
    log-scale x axis and linear-scale y axis.
    """

    # Grid
    _ax.grid(which='major', alpha=0.45)
    _ax.grid(which='minor', alpha=0.15)

    # -------------------------
    # X axis (LOG)
    # -------------------------
    locmaj_x = mticker.LogLocator(base=10, numticks=12)
    locmin_x = mticker.LogLocator(
        base=10.0,
        subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        numticks=12
    )

    _ax.xaxis.set_major_locator(locmaj_x)
    _ax.xaxis.set_minor_locator(locmin_x)
    _ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # -------------------------
    # Y axis (LINEAR)
    # -------------------------

    # major ticks
    _ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

    # minor ticks
    _ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))

    # -------------------------
    # Tick style
    # -------------------------
    _ax.tick_params(
        'both',
        direction='in',
        top=True,
        right=True,
        length=10,
        width=1,
        which='major',
        zorder=301
    )

    _ax.tick_params(
        'both',
        direction='in',
        top=True,
        right=True,
        length=5,
        width=1,
        which='minor',
        zorder=301
    )

def apply_scale_grid(
    _ax,
    xscale="log",
    yscale="log",
    grid=True,
    minor_grid=True,
    major_grid=True,
    log_numticks=12,
):
    """
    Apply scale, grid, and tick settings to a Matplotlib axis.

    Parameters
    ----------
    _ax : matplotlib.axes.Axes
        Axis to which the settings will be applied.

    xscale : {"log", "linear"}
        Scale for the x-axis.

    yscale : {"log", "linear"}
        Scale for the y-axis.

    grid : bool
        If True, apply grid settings.

    minor_grid : bool
        If True, show minor grid lines.

    major_grid : bool
        If True, show major grid lines.

    log_numticks : int
        Maximum number of ticks for log-scale axes.
    """

    allowed_scales = {"log", "linear"}

    if xscale not in allowed_scales:
        raise ValueError(f"xscale must be one of {allowed_scales}, got {xscale}")

    if yscale not in allowed_scales:
        raise ValueError(f"yscale must be one of {allowed_scales}, got {yscale}")

    # ------------------------------------------------------------------
    # Set axis scales
    # ------------------------------------------------------------------
    _ax.set_xscale(xscale)
    _ax.set_yscale(yscale)

    # ------------------------------------------------------------------
    # Grid settings
    # ------------------------------------------------------------------
    if grid:
        if minor_grid:
            _ax.grid(which="minor", alpha=0.2)
        if major_grid:
            _ax.grid(which="major", alpha=0.4)

    # ------------------------------------------------------------------
    # X-axis tick settings
    # ------------------------------------------------------------------
    if xscale == "log":
        locmaj_x = mticker.LogLocator(base=10, numticks=log_numticks)
        locmin_x = mticker.LogLocator(
            base=10.0,
            subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            numticks=log_numticks,
        )

        _ax.xaxis.set_major_locator(locmaj_x)
        _ax.xaxis.set_minor_locator(locmin_x)
        _ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    else:
        _ax.xaxis.set_major_locator(mticker.AutoLocator())
        _ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        _ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # ------------------------------------------------------------------
    # Y-axis tick settings
    # ------------------------------------------------------------------
    if yscale == "log":
        locmaj_y = mticker.LogLocator(base=10, numticks=log_numticks)
        locmin_y = mticker.LogLocator(
            base=10.0,
            subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            numticks=log_numticks,
        )

        _ax.yaxis.set_major_locator(locmaj_y)
        _ax.yaxis.set_minor_locator(locmin_y)
        _ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    else:
        _ax.yaxis.set_major_locator(mticker.AutoLocator())
        _ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        _ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # ------------------------------------------------------------------
    # Tick length and direction
    # ------------------------------------------------------------------
    _ax.tick_params(
        "both",
        direction="in",
        top=True,
        right=True,
        length=10,
        width=1,
        which="major",
        zorder=301,
    )

    _ax.tick_params(
        "both",
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1,
        which="minor",
        zorder=301,
    )

def apply_log_scale_grid(_ax):
    """
    Apply grid and log scale settings to a Matplotlib axis.

    Parameters:
        _ax (matplotlib.axes.Axes): The axis to which the settings will be applied.
    """
    # Grid settings
    _ax.grid(which='minor', alpha=0.2)
    _ax.grid(which='major', alpha=0.4)

    # Log-scale tick marks for x-axis
    locmaj_x = mticker.LogLocator(base=10, numticks=12)
    locmin_x = mticker.LogLocator(base=10.0,
                                  subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                  numticks=12)
    _ax.xaxis.set_major_locator(locmaj_x)
    _ax.xaxis.set_minor_locator(locmin_x)
    _ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # Log-scale tick marks for y-axis
    locmaj_y = mticker.LogLocator(base=10, numticks=12)
    locmin_y = mticker.LogLocator(base=10.0,
                                  subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                  numticks=12)
    _ax.yaxis.set_major_locator(locmaj_y)
    _ax.yaxis.set_minor_locator(locmin_y)
    _ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Tick length and direction
    _ax.tick_params('both', direction='in', top=True, right=True, length=10,
                    width=1, which='major', zorder=301)
    _ax.tick_params('both', direction='in', top=True, right=True, length=5,
                    width=1, which='minor', zorder=301)

######################################## NU PROFILE ####################################################################
sigma_eff_from_nu_kpcGyr, sigma_eff_from_nu_kms, (nu_grid, sigma_grid) = \
    build_sigma_m_eff_p5_interpolator(
        # sigma_m0=0.008,
        # m_GeV=0.02,
        # L=0,
        # Gamma=6e-12,
        # vR_kms=85.0,
        # nu_kms_min=0.1,
        # nu_kms_max=3000.0,
        # N=400
        sigma_m0=0.141,
        m_GeV=0.0301,
        L=0,
        Gamma=1.93e-13,
        vR_kms=55.8,
        nu_kms_min=0.1,
        nu_kms_max=3000.0,
        N=400

    )

########################################################### CSIDM ######################################################
kn_c = gravoEvolution_SIDM.return_knudsen_core_evolution(
    sigma_m=sigma_m,
    elements_rho=1,
    elements_vel=1,
    use_core_average=False,
    stop_at_collapse=False,
)

kn_r = gravoEvolution_RSIDM.return_knudsen_core_evolution(
    sigma_m=sigma_eff_from_nu_kms,
    elements_rho=1,
    elements_vel=1,
    sigma_input_vel_unit="km/s",
    use_core_average=False,
    stop_at_collapse=False,
)

# kn_r_nr700 = gravoEvolution_RSIDM_nr700.return_knudsen_core_evolution(
#     sigma_m=sigma_eff_from_nu_kms,
#     elements_rho=1,
#     elements_vel=1,
#     sigma_input_vel_unit="km/s",
#     use_core_average=False,
#     stop_at_collapse=False,
#     # stop_at_collapse=True,
# )


fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(kn_c["time"],  kn_c["Kn_core"], label="cSIDM")
ax.loglog(kn_r["time"], kn_r["Kn_core"], label="RSIDM")
# ax.loglog(kn_r_nr700["time"], kn_r_nr700["Kn_core"], label="RSIDM, precise")

ax.axhline(1.0, ls="--", color="gray", alpha=0.8)

ax.set_xlabel("time [Gyr]")
ax.set_ylabel(r"$\mathrm{Kn}_{\rm core}$")
ax.grid(which="minor", alpha=0.2)
ax.grid(which="major", alpha=0.4)
ax.legend()

plt.show()


################################################## NEW PLOT ############################################################
# # ------------------ DATA COLLECTION ------------------ #
# # sidm_models = [
# #     # SIDM_1, SIDM_2, SIDM_3,
# #     SIDM_4, SIDM_5, SIDM_6, SIDM_7, SIDM_8, SIDM_9,
# #     # SIDM_4, SIDM_5, SIDM_6, SIDM_7, SIDM_8
# #     # SIDM_10
# # ]
# #
# # rsidm_models = [
# #     # rSIDM_1, rSIDM_2, rSIDM_3,
# #     rSIDM_4, rSIDM_5, rSIDM_6, rSIDM_7, rSIDM_8, rSIDM_9,
# #     # rSIDM_4, rSIDM_5, rSIDM_6, rSIDM_7, rSIDM_8
# #     # rSIDM_10
# # ]
# #
# # masses_log10 = [
# #                 # 8.4, 8.6, 8.8,
# #                 9.0, 9.2, 9.4, 9.6, 9.8, 10.0,
# #                 # 9.0, 9.2, 9.4, 9.6, 9.8,
# #                 # 11.0
# #                 ]
# #
# # sigma_eff_arr = [
# #     # 0.0106, 0.421, 15.2,
# #     119.0, 845, 2095, 3148, 3345, 2500,
# #     # 119.0, 845, 2095, 3148, 3345
# #     # 63.86
# # ]
#
# # WORKS ONLY WHEN cSIDM REACHES UNIVERSALITY
# markers = ['o', 's', '^']
#
# # ordered colormap
# # cmap = mpl.cm.cividis
# # cmap = mpl.cm.viridis
# cmap = mpl.cm.plasma
#
# # norm = mpl.colors.Normalize(vmin=min(M_power), vmax=max(M_power))
# # colors = [cmap(norm(m)) for m in M_power]
#
# from matplotlib.lines import Line2D
#
# # ------------------ PLOT ------------------ #
# fig, ax = plt.subplots(figsize=(10.5, 6.8))
# ax.set_xscale('log')
# ax.set_yscale('log')
#
# # ax.set_xlim(1e-2, 3e1)
# # ax.set_xlim(1e-2, 2e1)
#
# # FOR Halo masses below 9.0
# # ax.set_xlim(2e-1, 3e0)
# # ax.set_xlim(1e0, 2e5)
#
# ax.set_ylim(1e0, 3e3)
#
# ax.set_xlabel(r'$t\,[\mathrm{Gyr}]$', fontsize=18)
# # ax.set_xlabel(
# #     r'$\hat{t} \equiv t/t_c,\quad '
# #     r't_c = \frac{150}{\beta\, r_s \, \rho_s \, (\sigma/m)_{\rm eff} \, \sqrt{4\pi G \rho_s}}$',
# #     fontsize=18)
# ax.set_ylabel(r'$\hat{\rho}_c \equiv \rho_c/\rho_s$', fontsize=18)
# ax.set_title(r'Core-density evolution: cSIDM vs rSIDM', fontsize=18)
#
# ax.tick_params(labelsize=13)
# apply_log_scale_grid(ax)
#
# t, rho = gravoEvolution_SIDM.return_rho_core_evolution(elements=elements)
# time_c = rescale_time(gravoEvolution_SIDM, sigma_m)
# ax.plot(
#     t, rho,
#     # t / time_c, rho,
#     linestyle='--',
#     linewidth=2.3,
#     # marker=markers[i],
#     # markersize=5,
#     # markevery=35,
#     # markerfacecolor='white',
#     # markeredgewidth=1.2,
#     color="blue",
#     alpha=0.8,
#     label=rf'$10^{{{M_power:.1f}}}\,M_\odot$'
# )
#
# t, rho = gravoEvolution_RSIDM.return_rho_core_evolution(elements=elements, index=2)
# time_c = rescale_time(gravoEvolution_RSIDM, sigma_m)
# ax.plot(
#     t, rho,
#     # t / time_c, rho,
#     linestyle='-',
#     linewidth=2.3,
#     color="blue",
#     alpha=0.95
# )
#
# t, rho = gravoEvolution_RSIDM_nr700.return_rho_core_evolution(elements=elements)
# time_c = rescale_time(gravoEvolution_RSIDM_nr700, sigma_m)
# ax.plot(
#     t, rho,
#     # t / time_c, rho,
#     linestyle='-',
#     linewidth=2.3,
#     color="black",
#     alpha=0.95
# )
#
#
# # legend 1: halo mass
# leg1 = ax.legend(
#     title=r'Halo mass',
#     fontsize=10,
#     title_fontsize=11,
#     loc='upper left',
#     bbox_to_anchor=(0.02, 0.78),
#     frameon=True,
#     framealpha=0.9
# )
# ax.add_artist(leg1)
#
# # legend 2: model type
# style_handles = [
#     Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='cSIDM'),
#     Line2D([0], [0], color='black', lw=2.3, linestyle='-', label='rSIDM'),
# ]
#
#
# # -- Handle vertical lines
# idx = np.argmin(np.abs(np.array(kn_r["Kn_core"]) - 1.0))
# closest_kn = kn_r["Kn_core"][idx]
# print("closet kn to 1.0:", closest_kn)
# closest_time = kn_r["time"][idx]
# # plt.axvline(x=closest_time/time_c, linestyle="-.", linewidth=2, alpha=0.8, color="grey")
# plt.axvline(x=closest_time, linestyle="-.", linewidth=2, alpha=0.8, color="grey")
#
# idx = np.argmin(np.abs(np.array(kn_r_nr700["Kn_core"]) - 1.0))
# closest_kn = kn_r_nr700["Kn_core"][idx]
# print("closet kn to 1.0:", closest_kn)
# closest_time = kn_r_nr700["time"][idx]
# # plt.axvline(x=closest_time/time_c, linestyle="-.", linewidth=2, alpha=0.8, color="grey")
# plt.axvline(x=closest_time, linestyle="-.", linewidth=2, alpha=0.8, color="black")
#
# ax.legend(
#     handles=style_handles,
#     title='Model',
#     fontsize=10,
#     title_fontsize=11,
#     loc='upper left',
#     frameon=True,
#     framealpha=0.9
# )
#
# plt.tight_layout()
# if plot_save:
#     plt.savefig(f'./{output_plot_dir}/core_density_evolution_csidm.png', dpi=300, bbox_inches='tight')
# else:
#     plt.show()


########################################################################################################################
fig, ax = plt.subplots(figsize=(10.5, 6.8))
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e0, 3e3)

ax.set_xlabel(r'$t\,[\mathrm{Gyr}]$', fontsize=18)
ax.set_ylabel(r'$\hat{\rho}_c \equiv \rho_c/\rho_s$', fontsize=18)
ax.set_title(r'Core-density evolution: sensitivity to central-bin definition', fontsize=17)

ax.tick_params(labelsize=13)
apply_scale_grid(ax, xscale="linear", yscale="log")

# --- cSIDM ---
t_csidm, rho_csidm = gravoEvolution_SIDM.return_rho_core_evolution(elements=elements)
ax.plot(
    t_csidm, rho_csidm,
    ls='--', lw=2.2, color='0.5',
    label='cSIDM'
)

if select_file == 0:
    t_csidm, rho_csidm = gravoEvolution_SIDM_gilman.return_rho_core_evolution(elements=elements)
    ax.plot(
        t_csidm, rho_csidm,
        ls='--', lw=2.2, color='black',
        label='cSIDM (Gilman)'
    )


# # --- rSIDM high-res ---
# t_hi, rho_hi = gravoEvolution_RSIDM_nr700.return_rho_core_evolution(elements=elements, index=2)
# ax.plot(
#     t_hi, rho_hi,
#     ls='-', lw=3.5, color='black',
#     label=r'rSIDM, $r_{\min}=10^{-4}$'
# )

# --- rSIDM low-res, first nonzero bin ---
t_lo1, rho_lo1 = gravoEvolution_RSIDM.return_rho_core_evolution(elements=elements, index=2)
ax.plot(
    t_lo1, rho_lo1,
    ls='-', lw=2.4, color='tab:blue',
    label=r'rSIDM, $r_{\min}=10^{-2}$, second nonzero bin'
)

# --- rSIDM low-res, second nonzero bin ---
# t_lo2, rho_lo2 = gravoEvolution_RSIDM.return_rho_core_evolution(elements=elements, index=2)
# idx = np.argmin(np.abs(rho_lo2 - 10**5))
# ax.plot(
#     t_lo2[:idx], rho_lo2[:idx],
#     ls='-', lw=2.4, color='tab:red',
#     label=r'rSIDM, $r_{\min}=10^{-2}$, second nonzero bin'
# )

# # --- rSIDM low-res (different combination), first nonzero bin ---
# t_lo2, rho_lo2 = gravoEvolution_RSIDM_Impact.return_rho_core_evolution(elements=elements, index=2)
# # idx = np.argmin(np.abs(rho_lo2 - 10**5))
# ax.plot(
#     t_lo2, rho_lo2,
#     ls='-', lw=2.4, color='tab:red',
#     label=r'rSIDM, IMPACT, first nonzero bin'
# )


# Optional: annotate halo mass directly instead of separate legend
ax.text(
    0.03, 0.05,
    rf'$M = 10^{{{M_power:.1f}}}\,M_\odot$',
    transform=ax.transAxes,
    fontsize=12,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.8')
)

# Optional: Kn=1 lines (subtle)

ax.legend(fontsize=10.5, loc='upper left', frameon=True, framealpha=0.92)

plt.tight_layout()
plt.show()


########################################################################################################################
# fig, ax = plt.subplots(figsize=(10.5, 6.8))
# ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_ylim(1e0, 1e2)
#
# ax.set_xlabel(r'$t\,[\mathrm{Gyr}]$', fontsize=18)
# ax.set_ylabel(r'$|\hat{\rho}_c^{high} - \hat{\rho}_c^{low}|/\hat{\rho}_c^{high}$', fontsize=18)
# ax.set_title(r'Core-density evolution: sensitivity to central-bin definition', fontsize=17)
#
# ax.tick_params(labelsize=13)
# apply_log_scale_grid(ax)
#
# # --- relative error ---
#
# # --- rSIDM high-res ---
# t_hi, rho_hi = gravoEvolution_RSIDM_nr700.return_rho_core_evolution(elements=elements, index=2)
# idx = len(t_hi)
# t_lo2, rho_lo2 = gravoEvolution_RSIDM_Impact.return_rho_core_evolution(elements=elements, index=2)
# # t_lo2, rho_lo2 = gravoEvolution_RSIDM.return_rho_core_evolution(elements=elements, index=2)
#
# ax.plot(
#     t_hi, abs(rho_hi-rho_lo2[:idx])/(rho_hi),
#     # t_hi, abs(np.log10(rho_hi) - np.log10(rho_lo2[:idx])),
#     ls='-', lw=3.5, color='black',
#     label=r'rSIDM, $r_{\min}^{high}=10^{-4}$, $r_{\min}^{low}=10^{-e}$'
# )
#
# # Optional: annotate halo mass directly instead of separate legend
# ax.text(
#     0.03, 0.05,
#     rf'$M = 10^{{{M_power:.1f}}}\,M_\odot$',
#     transform=ax.transAxes,
#     fontsize=12,
#     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.8')
# )
#
# # Optional: Kn=1 lines (subtle)
#
# ax.legend(fontsize=10.5, loc='upper left', frameon=True, framealpha=0.92)
#
# plt.tight_layout()
# plt.show()

gravoEvolution_RSIDM_vol2 = gravoF.create_gravothermalData_from_file(RSIDM_file_nr700, DATA_rSIDM_nr700_vol2, beta=beta)
gravoEvolution_RSIDM_vol2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM_vol2.put_extra_parameters(10**M_power, sigma_m)


fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.set_xscale('log')
ax.set_yscale('log')

data_new = gravoEvolution_RSIDM.return_data_at_fixed_time(86, time_step_bool=False)
data_old = gravoEvolution_RSIDM_vol2.return_data_at_fixed_time(86, time_step_bool=False)

print("OLD ONE")
print(data_old["rho"][0], data_old["rho"][1], data_old["rho"][2], data_old["rho"][3])
print(data_old["r"][0], data_old["r"][1], data_old["r"][2], data_old["r"][3])
print("NEW ONE")
print(data_new["rho"][0], data_new["rho"][1], data_new["rho"][2], data_new["rho"][3])
print(data_new["r"][0], data_new["r"][1], data_new["r"][2], data_new["r"][3])

# --- Create plot
ax.plot(data_new["r"], data_new["rho"], label='RSIDM, Truncated NFW')
ax.plot(data_old["r"], data_old["rho"], label='RSIDM, NFW')

# plt.ylim(-6.2, 2.2)
plt.ylabel(r'$\log\rho(r)$', fontsize=18)
plt.xlabel(r'$\log(r)$', fontsize=18)
# plt.title('Density profile on:' + "%10.2e" % (10 ** time_point) + ' Gyr', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.show()
