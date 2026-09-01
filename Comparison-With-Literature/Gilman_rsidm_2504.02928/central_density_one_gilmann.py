import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import numpy as np
import matplotlib as mpl
import config as cfg
from TruncatedNFWProfile import TruncatedNFWProfile
# ##################################### SETTINGS ##################################### #
# --- Manual settings
plot_save = False
make_gilman_plot = True
make_scanning_mass_plot = False
output_plot_dir = "Results"
beta = 0.85
beta_to_scale = 0.75

# --- choose of file
select_file = 0
elements = 3

# --- Set paths
OUR_DIR = os.path.dirname(os.path.abspath(__file__))
# BENCHMARK: GILMANN (HALO PARAMETERS)
DATA_cSIDM_ben3 = os.path.join(OUR_DIR, 'Data', "cSIDM", "Gilman_benchmark_3")
# Halo parameters: AYUKI (HALO PARAMETERS)
DATA_cSIDM_ben1 = os.path.join(OUR_DIR, 'Data', "cSIDM", "Gilman_benchmark_1")
# Halo parameters: GILMANN TABEL (HALO PARAMETERS)
DATA_cSIDM_gilman = os.path.join(OUR_DIR, 'Data', "cSIDM", "Gilman_tabel")

# BENCHMARK: GILMANN (HALO PARAMETERS)
Data_rSIMD_ben3 = os.path.join(OUR_DIR, "Data", "rSIDM", "Gilman_benchmark_3")
# Halo parameters: AYUKI (HALO PARAMETERS)
Data_rSIMD_ben1 = os.path.join(OUR_DIR, "Data", "rSIDM", "Gilman_benchmark_1")


# DATA_cSIDM_ben3 = os.path.join(OUR_DIR, 'Data_cSIDM/Gilman_benchmark_3')
# # Halo parameters: AYUKI (HALO PARAMETERS)
# DATA_cSIDM_ben1 = os.path.join(OUR_DIR, 'Data_cSIDM/Gilman_benchmark_1')
#
# # BENCHMARK: GILMANN (HALO PARAMETERS)
# Data_rSIMD_ben3 = os.path.join(OUR_DIR, "Data_rSIDM", "Gilman_benchmark_3")
# # Halo parameters: AYUKI (HALO PARAMETERS)
# Data_rSIMD_ben1 = os.path.join(OUR_DIR, "Data_rSIDM", "Gilman_benchmark_1")

# --- select files: cSIDM
# SIDM_file_gilman  = "CSIDM_Gilman_M7.0_c21.21_sigma24.95_beta0.85.csv"
# BENCHMARK: AYUKI
SIDM_file_1_ben1  = "CSIDM_M7.0_sigma35.1_beta0.75.csv"
SIDM_file_2_ben1  = "CSIDM_M7.5_sigma45.8_beta0.75.csv"
SIDM_file_3_ben1  = "CSIDM_M8.0_sigma27.1_beta0.75.csv"
SIDM_file_4_ben1  = "CSIDM_M8.5_sigma14.7_beta0.75.csv"
SIDM_file_5_ben1  = "CSIDM_M9.0_sigma9.15_beta0.75.csv"
# BENCHMARK: GILMANN
SIDM_file_1_ben3  = "CSIDM_Gilman_M7.0_c21.21_sigma30.545_beta0.85.csv"
SIDM_file_2_ben3  = "CSIDM_Gilman_M7.5_c19.81_sigma46.855_beta0.85.csv"
SIDM_file_3_ben3  = "CSIDM_Gilman_M8.0_c18.42_sigma28.378_beta0.85.csv"
SIDM_file_4_ben3  = "CSIDM_Gilman_M8.5_c17.05_sigma14.986_beta0.85.csv"
SIDM_file_5_ben3  = "CSIDM_Gilman_M9.0_c15.69_sigma9.19_beta0.85.csv"

gilman_sigma_m_arr = [28.482, 46.571, 28.334, 15.097, 9.136]
CSIDM_table1_file1 = "CSIDM_Gilman_M7.0_c21.21_sigma28.482_beta0.85.csv"
CSIDM_table1_file2 = "CSIDM_Gilman_M7.5_c19.81_sigma46.571_beta0.85.csv"
CSIDM_table1_file3 = "CSIDM_Gilman_M8.0_c18.42_sigma28.334_beta0.85.csv"
CSIDM_table1_file4 = "CSIDM_Gilman_M8.5_c17.05_sigma15.097_beta0.85.csv"
CSIDM_table1_file5 = "CSIDM_Gilman_M9.0_c15.69_sigma9.137_beta0.85.csv"

gilman_sigma_m_arr_vol2 = [24.759, 47.358, 30.905, 16.233, 9.701]
CSIDM_table2_file1 = "CSIDM_Gilman_M7.0_c21.21_sigma24.759_beta0.85.csv"
CSIDM_table2_file2 = "CSIDM_Gilman_M7.5_c19.81_sigma47.358_beta0.85.csv"
CSIDM_table2_file3 = "CSIDM_Gilman_M8.0_c18.42_sigma30.905_beta0.85.csv"
CSIDM_table2_file4 = "CSIDM_Gilman_M8.5_c17.05_sigma16.233_beta0.85.csv"
CSIDM_table2_file5 = "CSIDM_Gilman_M9.0_c15.69_sigma9.701_beta0.85.csv"

SIDM_file_ben1_arr = [SIDM_file_1_ben1, SIDM_file_2_ben1, SIDM_file_3_ben1, SIDM_file_4_ben1, SIDM_file_5_ben1]
SIDM_file_ben3_arr = [SIDM_file_1_ben3, SIDM_file_2_ben3, SIDM_file_3_ben3, SIDM_file_4_ben3, SIDM_file_5_ben3]
CSIDM_table_file_arr1 = [CSIDM_table1_file1, CSIDM_table1_file2, CSIDM_table1_file3, CSIDM_table1_file4, CSIDM_table1_file5]
CSIDM_table_file_arr2 = [CSIDM_table2_file1, CSIDM_table2_file2, CSIDM_table2_file3, CSIDM_table2_file4, CSIDM_table2_file5]

# --- select files: rSIDM
# BENCHMARK: AYUKI
RSIDM_file_1_ben1  = "RSIDM_M7.0_beta0.75_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_2_ben1  = "RSIDM_M7.5_beta0.75_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_3_ben1  = "RSIDM_M8.0_beta0.75_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_4_ben1  = "RSIDM_M8.5_beta0.75_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_5_ben1  = "RSIDM_M9.0_beta0.75_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
# BENCHMARK: GILMANN
RSIDM_file_1_ben3  = "RSIDM_Gilman_M7.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_2_ben3  = "RSIDM_Gilman_M7.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_3_ben3  = "RSIDM_Gilman_M8.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_4_ben3  = "RSIDM_Gilman_M8.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
RSIDM_file_5_ben3  = "RSIDM_Gilman_M9.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
# BENCHMARK: GILMAN TABLE
RSIDM_table_file1 = "RSIDM_Gilman_M7.0.csv"
RSIDM_table_file2 = "RSIDM_Gilman_M7.5.csv"
RSIDM_table_file3 = "RSIDM_Gilman_M8.0.csv"
RSIDM_table_file4 = "RSIDM_Gilman_M8.5.csv"
RSIDM_table_file5 = "RSIDM_Gilman_M9.0.csv"

RSIDM_file_ben1_arr = [RSIDM_file_1_ben1, RSIDM_file_2_ben1, RSIDM_file_3_ben1, RSIDM_file_4_ben1, RSIDM_file_5_ben1]
RSIDM_file_ben3_arr = [RSIDM_file_1_ben3, RSIDM_file_2_ben3, RSIDM_file_3_ben3, RSIDM_file_4_ben3, RSIDM_file_5_ben3]
RSIDM_table_file_arr = [RSIDM_table_file1, RSIDM_table_file2, RSIDM_table_file3, RSIDM_table_file4, RSIDM_table_file5]

# --- SELECT SIDM
SIDM_file_ben3 = SIDM_file_ben3_arr[select_file]
SIDM_file_ben1 = SIDM_file_ben1_arr[select_file]

# --- SELECT RSIDM
RSIDM_file_ben3 = RSIDM_file_ben3_arr[select_file]
RSIDM_file_ben1 = RSIDM_file_ben1_arr[select_file]

# --- select parameters values
sigma_m_arr = [30.545, 46.855, 28.378, 14.986, 9.19]
sigma_m=sigma_m_arr[select_file]

M_power_arr = [7.0, 7.5, 8.0, 8.5, 9.0]
M_power = M_power_arr[select_file]

con_arr=[21.21, 19.81, 18.42, 17.05, 15.69]
con=con_arr[select_file]

# ##################################### SET DATA TO CLASS ############################################################ #
# --- SIDM
gravoEvolution_SIDM_ben3 = gravoF.create_gravothermalData_from_file(SIDM_file_ben3, DATA_cSIDM_ben3, beta=beta)
gravoEvolution_SIDM_ben3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM_ben3.put_extra_parameters(10 ** M_power, sigma_m)

gravoEvolution_SIDM_ben1 = gravoF.create_gravothermalData_from_file(SIDM_file_ben1, DATA_cSIDM_ben1, beta=beta)
gravoEvolution_SIDM_ben1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM_ben1.put_extra_parameters(10 ** 7.0, 24.95)

# --- RSIDM
gravoEvolution_RSIDM_ben3 = gravoF.create_gravothermalData_from_file(RSIDM_file_ben3, Data_rSIMD_ben3, beta=beta)
gravoEvolution_RSIDM_ben3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM_ben3.put_extra_parameters(10**M_power, sigma_m)

gravoEvolution_RSIDM_ben1 = gravoF.create_gravothermalData_from_file(RSIDM_file_ben1, Data_rSIMD_ben1, beta=beta)
gravoEvolution_RSIDM_ben1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM_ben1.put_extra_parameters(10**M_power, sigma_m)

halo_rSIDM = TruncatedNFWProfile(_M_vir=10**M_power, _con=con)
tau_kappa = halo_rSIDM.tau(beta=beta, sigma_eff=sigma_m)

# _, time_collapse, time_step_collapse = gravoEvolution_SIDM.find_collapse(elements=elements, fixed_limit=10 ** 10)
# print(f"time collapse: {time_collapse} [Gyr], time step: {time_step_collapse}")

# ##################################### PLOTTING COMPARISON ########################################################## #
def rescale_time(model, _sigma_eff, beta=0.85, input_r_s=0.0, input_rho_s=0.0, manual=False):
    if manual:
        r_s = input_r_s  # [kpc]
        rho_s = 10**(input_rho_s)  # [Msun * kcp^-3]
    else:
        r_s = model.parameters["r_s"]
        rho_s = model.parameters["rho_s"]
    sigma_m_SI = _sigma_eff * 1e-4 * 1e3  # Convert from [cm^2/g] to [m^2/kg]
    sigma_m_SU = sigma_m_SI * cfg.kpc_SI ** -2 * cfg.M_solar_SI  # Convert to [kpc^2 * M_sun^-1]
    G_SU = cfg.const_G_starUnits
    time_c = (150/beta) * (1/(r_s * rho_s) * 1/sigma_m_SU) * (4 * np.pi * G_SU * rho_s) ** (-1/2)
    # print("====================================================================================")
    # print("==================== RESCAL TIME FUNCTION ==========================================")
    # print("_sigma_eff:", _sigma_eff)
    # print("beta:", beta)
    # print("r_s:", r_s)
    # print("log_rho_s:", np.log10(rho_s))
    # print("sigma_m_SU:", sigma_m_SU)
    # print("G_SU:", G_SU)
    # print("time_c:", time_c)
    return time_c


time_c = rescale_time(gravoEvolution_RSIDM_ben1, sigma_m)
print("time_c:",time_c)

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

########################################################################################################################
fig, ax = plt.subplots(figsize=(10.5, 6.8))
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e0, 3e3)

ax.set_xlabel(r'$t\,[\mathrm{Gyr}]$', fontsize=18)
ax.set_xlabel(r'$t = T/\tau_{\kappa}$', fontsize=18)
ax.set_ylabel(r'$\hat{\rho}_c \equiv \rho_c/\rho_s$', fontsize=18)
ax.set_title(r'Core-density evolution: sensitivity to halo parameters (c-M relation)', fontsize=17)

ax.tick_params(labelsize=13)
apply_scale_grid(ax, xscale="linear", yscale="log")


# --- cSIDM ---
t_csidm_1, rho_csidm_1 = gravoEvolution_SIDM_ben3.return_rho_core_evolution(elements=elements)
t_collapse_1 = t_csidm_1[-1]
ax.plot(
    t_csidm_1/t_collapse_1,
    # t_csidm_1,
    rho_csidm_1,
    ls='--', lw=2.2, color='tab:blue',
    label='cSIDM (Gilman halo)'
)

t_csidm_2, rho_csidm_2 = gravoEvolution_SIDM_ben1.return_rho_core_evolution(elements=elements)
t_collapse_2 = t_csidm_2[-1]
t_csidm_scale_1 = t_csidm_2*(beta/beta_to_scale)**(-1)
ax.plot(
    t_csidm_2 / t_collapse_2,
    # t_csidm_scale,
    rho_csidm_2,
    ls='--', lw=2.2, color='tab:red',
    label='cSIDM (Old halo)'
)
print("cSIDM collapse:", t_collapse_1)

# --- rSIDM low-res, first nonzero bin ---
t_lo1, rho_lo1 = gravoEvolution_RSIDM_ben3.return_rho_core_evolution(elements=elements, index=2)
ax.plot(
    t_lo1/t_collapse_1,
    # t_lo1,
    rho_lo1,
    ls='-', lw=2.4, color='tab:blue',
    label='rSIDM (Gilman halo)'
    # label=r'rSIDM, $r_{\min}=10^{-2}$, second nonzero bin'
)

t_lo2, rho_lo2 = gravoEvolution_RSIDM_ben1.return_rho_core_evolution(elements=elements, index=2)
t_lo2_scale=t_lo2*(beta/beta_to_scale)**(-1)
ax.plot(
    t_lo2 / t_collapse_2,
    # t_lo2_scale,
    rho_lo2,
    ls='-', lw=2.4, color='tab:red',
    label='rSIDM (Old halo)'
    # label=r'rSIDM, $r_{\min}=10^{-2}$, second nonzero bin'
)


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

print("rSIDM collapse:", t_collapse_1)
print("tau_kappa:", tau_kappa)

# ############################## GILMAN PLOT ######################################################################### #
if make_gilman_plot:
    # --- Digitize Fig 3 ()
    # Path to your file
    file_path = "Gilman_Fig3_digitize.csv"

    # Read file without assuming a normal single-row header
    raw = pd.read_csv(file_path, header=None)

    # First row contains dataset names, second row contains X/Y labels
    dataset_names = raw.iloc[0].tolist()
    data = raw.iloc[2:].reset_index(drop=True)

    # Convert all data values to numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    # Helper function: read one X/Y pair starting from a given column
    def read_dataset(x_col, y_col):
        tmp = data[[x_col, y_col]].dropna()

        return {
            "t": tmp[x_col].to_numpy(),
            "rho": tmp[y_col].to_numpy(),
        }

    rho_8 = read_dataset(0, 1)
    rho_9 = read_dataset(2, 3)
    rho_7 = read_dataset(4, 5)
    rho_self_similar = read_dataset(6, 7)

    # Optional: collect them together
    rho_data = {
        "M8p0": rho_8,
        "M9p0": rho_9,
        "M7p0": rho_7,
        "Self-similar": rho_self_similar
    }
    # --- GILMAN TABLE
    gilman_M_power_arr = [7.0, 7.5, 8.0, 8.5, 9.0]
    # gilman_sigma_m_arr = [28.482, 46.571, 28.334, 15.097, 9.136]
    gilman_sigma_m_arr_vol2 = [24.759, 47.358, 30.905, 16.233, 9.700]
    gilman_sigma_m_arr_rsidm = [24.759, 47.358, 30.905, 16.233, 9.700]
    # gilman_sigma_m_arr_rsidm_c = [26.425, 47.414, 30.63, 16.094, 9.7489]
    gilman_r200_arr = np.array([4.55, 6.68, 9.81, 14.4, 21.1])       # [kpc]
    gilman_log_rho_s_arr = np.array([7.57, 7.50, 7.42, 7.33, 7.24])  # [Msun * pc^-3]
    gilman_c_arr = np.array([21.21,  19.81, 18.42, 17.05, 15.69])
    gilman_rs_arr = [0.220, 0.340, 0.530, 0.840, 1.350]

    # --- SIDM (C-Base)
    gravoEvolution_SIDM_1 = gravoF.create_gravothermalData_from_file(SIDM_file_ben3_arr[0], DATA_cSIDM_ben3, beta=beta)
    gravoEvolution_SIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_SIDM_1.put_extra_parameters(10 ** M_power_arr[0], sigma_m_arr[0])

    gravoEvolution_SIDM_2 = gravoF.create_gravothermalData_from_file(SIDM_file_ben3_arr[1], DATA_cSIDM_ben3, beta=beta)
    gravoEvolution_SIDM_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_SIDM_2.put_extra_parameters(10 ** M_power_arr[1], sigma_m_arr[1])

    gravoEvolution_SIDM_3 = gravoF.create_gravothermalData_from_file(SIDM_file_ben3_arr[2], DATA_cSIDM_ben3, beta=beta)
    gravoEvolution_SIDM_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_SIDM_3.put_extra_parameters(10 ** M_power_arr[2], sigma_m_arr[2])

    gravoEvolution_SIDM_4 = gravoF.create_gravothermalData_from_file(SIDM_file_ben3_arr[3], DATA_cSIDM_ben3, beta=beta)
    gravoEvolution_SIDM_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_SIDM_4.put_extra_parameters(10 ** M_power_arr[3], sigma_m_arr[3])

    gravoEvolution_SIDM_5 = gravoF.create_gravothermalData_from_file(SIDM_file_ben3_arr[4], DATA_cSIDM_ben3, beta=beta)
    gravoEvolution_SIDM_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_SIDM_5.put_extra_parameters(10 ** M_power_arr[4], sigma_m_arr[4])

    # --- SIDM (T-Base)
    # select_dir = CSIDM_table_file_arr1
    # select_cross_section = gilman_sigma_m_arr
    select_dir = CSIDM_table_file_arr2
    select_cross_section = gilman_sigma_m_arr_vol2

    cSIDM_1_gilmann = gravoF.create_gravothermalData_from_file(select_dir[0], DATA_cSIDM_gilman, beta=beta)
    cSIDM_1_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    cSIDM_1_gilmann.put_extra_parameters(10 ** M_power_arr[0], select_cross_section[0])

    cSIDM_2_gilmann = gravoF.create_gravothermalData_from_file(select_dir[1], DATA_cSIDM_gilman, beta=beta)
    cSIDM_2_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    cSIDM_2_gilmann.put_extra_parameters(10 ** M_power_arr[1], select_cross_section[1])

    cSIDM_3_gilmann = gravoF.create_gravothermalData_from_file(select_dir[2], DATA_cSIDM_gilman, beta=beta)
    cSIDM_3_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    cSIDM_3_gilmann.put_extra_parameters(10 ** M_power_arr[2], select_cross_section[2])

    cSIDM_4_gilmann = gravoF.create_gravothermalData_from_file(select_dir[3], DATA_cSIDM_gilman, beta=beta)
    cSIDM_4_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    cSIDM_4_gilmann.put_extra_parameters(10 ** M_power_arr[3], select_cross_section[3])

    cSIDM_5_gilmann = gravoF.create_gravothermalData_from_file(select_dir[4], DATA_cSIDM_gilman, beta=beta)
    cSIDM_5_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    cSIDM_5_gilmann.put_extra_parameters(10 ** M_power_arr[4], select_cross_section[4])

    # --- RSIDM (T-Base)
    rSIDM_1_gilmann = gravoF.create_gravothermalData_from_file(RSIDM_table_file_arr[0], Data_rSIMD_ben3, beta=beta)
    rSIDM_1_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_1_gilmann.put_extra_parameters(10 ** gilman_M_power_arr[0], gilman_sigma_m_arr_rsidm[0])

    rSIDM_2_gilmann = gravoF.create_gravothermalData_from_file(RSIDM_table_file_arr[1], Data_rSIMD_ben3, beta=beta)
    rSIDM_2_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_2_gilmann.put_extra_parameters(10 ** gilman_M_power_arr[1], gilman_sigma_m_arr_rsidm[1])

    rSIDM_3_gilmann = gravoF.create_gravothermalData_from_file(RSIDM_table_file_arr[2], Data_rSIMD_ben3, beta=beta)
    rSIDM_3_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_3_gilmann.put_extra_parameters(10 ** gilman_M_power_arr[2], gilman_sigma_m_arr_rsidm[2])

    rSIDM_4_gilmann = gravoF.create_gravothermalData_from_file(RSIDM_table_file_arr[3], Data_rSIMD_ben3, beta=beta)
    rSIDM_4_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_4_gilmann.put_extra_parameters(10 ** gilman_M_power_arr[3], gilman_sigma_m_arr_rsidm[3])

    rSIDM_5_gilmann = gravoF.create_gravothermalData_from_file(RSIDM_table_file_arr[4], Data_rSIMD_ben3, beta=beta)
    rSIDM_5_gilmann.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_5_gilmann.put_extra_parameters(10 ** gilman_M_power_arr[4], gilman_sigma_m_arr_rsidm[4])

    # --- RSIDM (C-Base)
    gravoEvolution_RSIDM_1 = gravoF.create_gravothermalData_from_file(RSIDM_file_ben3_arr[0], Data_rSIMD_ben3, beta=beta)
    gravoEvolution_RSIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_RSIDM_1.put_extra_parameters(10**M_power_arr[0], sigma_m_arr[0])

    gravoEvolution_RSIDM_2 = gravoF.create_gravothermalData_from_file(RSIDM_file_ben3_arr[1], Data_rSIMD_ben3, beta=beta)
    gravoEvolution_RSIDM_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_RSIDM_2.put_extra_parameters(10**M_power_arr[1], sigma_m_arr[1])

    gravoEvolution_RSIDM_3 = gravoF.create_gravothermalData_from_file(RSIDM_file_ben3_arr[2], Data_rSIMD_ben3, beta=beta)
    gravoEvolution_RSIDM_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_RSIDM_3.put_extra_parameters(10**M_power_arr[2], sigma_m_arr[2])

    gravoEvolution_RSIDM_4 = gravoF.create_gravothermalData_from_file(RSIDM_file_ben3_arr[3], Data_rSIMD_ben3, beta=beta)
    gravoEvolution_RSIDM_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_RSIDM_4.put_extra_parameters(10**M_power_arr[3], sigma_m_arr[3])

    gravoEvolution_RSIDM_5 = gravoF.create_gravothermalData_from_file(RSIDM_file_ben3_arr[4], Data_rSIMD_ben3, beta=beta)
    gravoEvolution_RSIDM_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    gravoEvolution_RSIDM_5.put_extra_parameters(10**M_power_arr[4], sigma_m_arr[4])

    # ### Plot part
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e0, 1e2)

    ax.set_xlabel(r'$t=T/\tau_{\kappa}$', fontsize=18)
    # ax.set_xlabel(r'$t=T$', fontsize=18)
    ax.set_ylabel(r'$\hat{\rho}_c \equiv \rho_c/\rho_s$', fontsize=18)
    ax.set_title(r'Core-density evolution: deviations from the cSIDM self-similar track', fontsize=17)

    ax.tick_params(labelsize=13)
    apply_scale_grid(ax, xscale="linear", yscale="log")

    # --- cSIDM ---
    # ------------------------------------------------------------
    # Lista symulacji, które chcesz potraktować jako jeden obszar
    # ------------------------------------------------------------
    sidm_runs = [
        gravoEvolution_SIDM_1,
        gravoEvolution_SIDM_2,
        gravoEvolution_SIDM_3,
        gravoEvolution_SIDM_4,
        gravoEvolution_SIDM_5,
    ]

    # ------------------------------------------------------------
    # Wspólna siatka czasu: t / t_end
    # ------------------------------------------------------------
    t_grid = np.linspace(0.0, 1.0, 500)

    rho_curves = []

    index = int(0)
    for run in sidm_runs:
        t, rho = run.return_rho_core_evolution(elements=elements)

        # reskalowanie czasu
        # t_scaled = t / t[-1]

        halo_rSIDM = TruncatedNFWProfile(_M_vir=10**M_power_arr[index], _con=con_arr[index])
        tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=sigma_m_arr[index])
        index = int(index+1)
        t_scaled = t / tau_kappa
        # t_scaled = t

        # sortowanie na wszelki wypadek
        order = np.argsort(t_scaled)
        t_scaled = t_scaled[order]
        rho = rho[order]

        # interpolacja na wspólną siatkę
        rho_interp = np.interp(t_grid, t_scaled, rho)

        rho_curves.append(rho_interp)

    rho_curves = np.array(rho_curves)

    # ------------------------------------------------------------
    # Obwiednia: minimum i maksimum spośród krzywych
    # ------------------------------------------------------------
    rho_min = np.min(rho_curves, axis=0)
    rho_max = np.max(rho_curves, axis=0)

    # ------------------------------------------------------------
    # Opcjonalnie: linia centralna, np. mediana
    # ------------------------------------------------------------
    rho_med = np.median(rho_curves, axis=0)

    # ------------------------------------------------------------
    # Rysowanie obszaru zamiast wszystkich osobnych krzywych
    # ------------------------------------------------------------
    # ax.fill_between(
    #     t_grid,
    #     rho_min,
    #     rho_max,
    #     alpha=0.95,
    #     label=r"cSIDM parameter range",
    # )

    # ax.plot(
    #     t_grid,
    #     rho_med,
    #     color="black",
    #     ls="-",
    #     lw=2.0,
    #     label=r"cSIDM self-similar"
    # )
    # # Gilman
    # ax.plot(
    #     rho_data["Self-similar"]["t"],
    #     rho_data["Self-similar"]["rho"],
    #     color="green",
    #     ls="--",
    #     lw=2.0,
    #     label=r"cSIDM self-similar (Gilman)"
    # )
    Mass_index=1

    gilman_C_csidm_model_arr = [gravoEvolution_SIDM_1, gravoEvolution_SIDM_2, gravoEvolution_SIDM_3, gravoEvolution_SIDM_4, gravoEvolution_SIDM_5]
    gilman_C_csidm_model = gilman_C_csidm_model_arr[Mass_index]
    gilman_T_csidm_model_arr = [cSIDM_1_gilmann, cSIDM_2_gilmann, cSIDM_3_gilmann, cSIDM_4_gilmann, cSIDM_5_gilmann]
    gilman_T_csidm_model = gilman_T_csidm_model_arr[Mass_index]

    gilman_C_rsidm_model_arr = [gravoEvolution_RSIDM_1, gravoEvolution_RSIDM_2, gravoEvolution_RSIDM_3, gravoEvolution_RSIDM_4, gravoEvolution_RSIDM_5]
    gilman_C_rsidm_model = gilman_C_rsidm_model_arr[Mass_index]
    gilman_T_rsidm_model_arr = [rSIDM_1_gilmann, rSIDM_2_gilmann, rSIDM_3_gilmann, rSIDM_4_gilmann, rSIDM_5_gilmann]
    gilman_T_rsidm_model = gilman_T_rsidm_model_arr[Mass_index]

    # t_lo1, rho_lo1 = gilman_C_csidm_model.return_rho_core_evolution(elements=elements, index=2)
    # t_scale = t_lo1 / t_lo1[-1]
    # halo_rSIDM = TruncatedNFWProfile(_M_vir=10**M_power_arr[Mass_index], _con=con_arr[Mass_index])
    # tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=sigma_m_arr[Mass_index])
    # ax.plot(
    #     # t_scale, rho_lo1,
    #     # t_lo1, rho_lo1,
    #     t_lo1/tau_kappa, rho_lo1,
    #     ls='-', lw=2.4, color="black",
    #     label=r"cSIDM self-similar"
    # )

    t_lo1, rho_lo1 = gilman_T_csidm_model.return_rho_core_evolution(elements=elements, index=2)
    t_scale = t_lo1 / t_lo1[-1]
    halo_rSIDM = TruncatedNFWProfile(_M_vir=10**gilman_M_power_arr[Mass_index],
                                     _con=gilman_c_arr[Mass_index],
                                     _r200=gilman_r200_arr[Mass_index],
                                     _log_rho_s=gilman_log_rho_s_arr[Mass_index])
    tau_kappa = rescale_time(halo_rSIDM, gilman_sigma_m_arr_vol2[Mass_index],
                             beta=0.85, input_r_s=gilman_rs_arr[Mass_index], input_rho_s=gilman_log_rho_s_arr[Mass_index], manual=True)
    # halo_rSIDM = TruncatedNFWProfile(_M_vir=10**gilman_M_power_arr[Mass_index], _con=gilman_c_arr[Mass_index])
    # tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=gilman_sigma_m_arr_vol2[Mass_index])
    print("our tau_kappa", )
    ax.plot(
        # t_scale, rho_lo1,
        # t_lo1, rho_lo1,
        t_lo1/tau_kappa, rho_lo1,
        ls=':', lw=2.4, color="black",
        # label=r"cSIDM self-similar (T-Base, $\sigma_{1D}=1.05 \cdot \sigma_s$)"
        label = r"cSIDM self-similar (T-Base)"
    )


    # --- rSIDM low-res, first nonzero bin ---
    # Kolory bardziej zbliżone do oryginalnego rysunku
    color_1 = "#8b1e1e"   # dark red / brown
    color_2 = "#ff2ca3"   # magenta / pink
    color_3 = "#1f2aa6"   # deep blue

    # --- M200 = 10^7 Msun ---
    # t_lo1, rho_lo1 = gilman_C_rsidm_model.return_rho_core_evolution(elements=elements, index=2)
    # t_scale = t_lo1 / t_lo1[-1]
    # halo_rSIDM = TruncatedNFWProfile(_M_vir=10**M_power_arr[Mass_index], _con=con_arr[Mass_index])
    # tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=sigma_m_arr[Mass_index])
    # tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=select_cross_section[Mass_index])
    # ax.plot(
    #     # t_scale, rho_lo1,
    #     # t_lo1, rho_lo1,
    #     t_lo1/tau_kappa, rho_lo1,
    #     ls='-', lw=2.4, color=color_1,
    #     # label=r"$M_{200}=10^{7.0}\,M_\odot$ (C-base)"
    #     # ls='-', lw=2.4, color="blue",
    #     label=r"$M_{200}=10^{7.0}\,M_\odot$ (T-base, $\sigma_{1D}=1.05 \cdot \sigma_s$)"
    # )

    # GILMAN TABLE
    t_lo1, rho_lo1 = gilman_T_rsidm_model.return_rho_core_evolution(elements=elements, index=2)
    t_scale = t_lo1 / t_lo1[-1]
    halo_rSIDM = TruncatedNFWProfile(_M_vir=10**gilman_M_power_arr[Mass_index],
                                     _con=gilman_c_arr[Mass_index],
                                     _r200=gilman_r200_arr[Mass_index],
                                     _log_rho_s=gilman_log_rho_s_arr[Mass_index])
    tau_kappa = rescale_time(halo_rSIDM, gilman_sigma_m_arr_rsidm[Mass_index],
                             beta=0.85, input_r_s=gilman_rs_arr[Mass_index], input_rho_s=gilman_log_rho_s_arr[Mass_index], manual=True)
    ax.plot(
        # t_scale, rho_lo1,
        # t_lo1, rho_lo1,
        t_lo1/tau_kappa, rho_lo1,
        ls='-', lw=2.4, color="blue",
        # label=r"$M_{200}=10^{7.0}\,M_\odot$ (T-base, $\sigma_{1D}=1.05 \cdot \sigma_s$)"
        label=r"$M_{200}=10^{7.0}\,M_\odot$ (T-base)"
    )

    #
    # Gilman results
    # ax.plot(
    #     rho_data["M7p0"]["t"], rho_data["M7p0"]["rho"],
    #     ls='--', lw=2.4, color=color_1,
    #     label=r"$M_{200}=10^{7}\,M_\odot$"
    # )

    # --- M200 = 10^8 Msun ---
    # t_lo1, rho_lo1 = gravoEvolution_RSIDM_3.return_rho_core_evolution(elements=elements, index=2)
    # t_scale = t_lo1 / t_lo1[-1]
    # halo_rSIDM = TruncatedNFWProfile(_M_vir=10**M_power_arr[2], _con=con_arr[2])
    # tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=sigma_m_arr[2])
    # ax.plot(
    #     # t_scale, rho_lo1,
    #     t_lo1/tau_kappa, rho_lo1,
    #     # t_lo1, rho_lo1,
    #     ls='-', lw=2.4, color=color_2,
    #     label=r"$M_{200}=10^{8}\,M_\odot$ (C-base)"
    # )

    # GILMAN TABLE
    # t_lo1, rho_lo1 = rSIDM_3_gilmann.return_rho_core_evolution(elements=elements, index=2)
    # t_scale = t_lo1 / t_lo1[-1]
    # halo_rSIDM = TruncatedNFWProfile(_M_vir=10**gilman_M_power_arr[2],
    #                                  _con=gilman_c_arr[2],
    #                                  _r200=gilman_r200_arr[2],
    #                                  _log_rho_s=gilman_log_rho_s_arr[2])
    # tau_kappa = rescale_time(halo_rSIDM, gilman_sigma_m_arr[2],
    #                          beta=0.85, input_r_s=gilman_rs_arr[2], input_rho_s=gilman_log_rho_s_arr[2], manual=True)
    # ax.plot(
    #     # t_scale, rho_lo1,
    #     t_lo1/tau_kappa, rho_lo1,
    #     # t_lo1, rho_lo1,
    #     ls='-', lw=2.4, color="grey",
    #     label=r"$M_{200}=10^{8.0}\,M_\odot$ (T-base)"
    # )


    # Gilman results
    # ax.plot(
    #     rho_data["M8p0"]["t"], rho_data["M8p0"]["rho"],
    #     ls='--', lw=2.4, color=color_2,
    #     label=r"$M_{200}=10^{8}\,M_\odot$"
    # )

    # --- M200 = 10^9 Msun ---
    # t_lo1, rho_lo1 = gravoEvolution_RSIDM_5.return_rho_core_evolution(elements=elements, index=2)
    # t_scale = t_lo1 / t_lo1[-1]
    # halo_rSIDM = TruncatedNFWProfile(_M_vir=10**M_power_arr[4], _con=con_arr[4])
    # tau_kappa = rescale_time(halo_rSIDM, gilman_sigma_m_arr[4],
    #                          beta=0.85, input_r_s=gilman_rs_arr[4], input_rho_s=gilman_log_rho_s_arr[4], manual=True)
    # ax.plot(
    #     t_lo1, rho_lo1,
    #     ls='-', lw=2.4, color=color_3,
    #     label=r"$M_{200}=10^{9}\,M_\odot$ (C-base)"
    # )
    #
    # # GILMAN TABLE
    # t_lo1, rho_lo1 = rSIDM_5_gilmann.return_rho_core_evolution(elements=elements, index=2)
    # t_scale = t_lo1 / t_lo1[-1]
    # halo_rSIDM = TruncatedNFWProfile(_M_vir=10**gilman_M_power_arr[4],
    #                                  _con=gilman_c_arr[4],
    #                                  _r200=gilman_r200_arr[4],
    #                                  _log_rho_s=gilman_log_rho_s_arr[4])
    # tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=gilman_sigma_m_arr[4])
    # ax.plot(
    #     # t_scale, rho_lo1,
    #     # t_lo1/tau_kappa, rho_lo1,
    #     t_lo1, rho_lo1,
    #     ls='-', lw=2.4, color="grey",
    #     label=r"$M_{200}=10^{9.0}\,M_\odot$ (T-base)"
    # )

    # Gilman results
    # ax.plot(
    #     rho_data["M9p0"]["t"], rho_data["M9p0"]["rho"],
    #     ls='--', lw=2.4, color=color_3,
    #     label=r"$M_{200}=10^{8}\,M_\odot$"
    # )



    ax.legend(
        loc="upper center",
        frameon=False,
        fontsize=14,
        handlelength=1.8,
        handletextpad=0.5,
    )

    plt.tight_layout()
    plt.show()


# ########################################### SCANNIN MASS ########################################################### #
if make_scanning_mass_plot:
    # --- select files: cSIDM
    # BENCHMARK: GILMAN (EXTRAPOLATION)
    # SIDM_file1_extra = "CSIDM_Gilman_M6.0_c23.96_sigma10.713_beta0.85.csv"
    # SIDM_file2_extra = "CSIDM_Gilman_M6.2_c23.4_sigma10.899_beta0.85.csv"
    # SIDM_file3_extra = "CSIDM_Gilman_M6.4_c22.85_sigma11.512_beta0.85.csv"
    # SIDM_file4_extra = "CSIDM_Gilman_M6.6_c22.3_sigma13.645_beta0.85.csv"
    # SIDM_file5_extra = "CSIDM_Gilman_M6.8_c21.75_sigma19.634_beta0.85.csv"
    SIDM_file1_extra = "CSIDM_Gilman_M7.0_c21.21_sigma30.545_beta0.85.csv"
    SIDM_file2_extra = "CSIDM_Gilman_M7.1_c20.92_sigma35.434_beta0.85.csv"
    SIDM_file3_extra = "CSIDM_Gilman_M7.2_c20.64_sigma41.033_beta0.85.csv"
    SIDM_file4_extra = "CSIDM_Gilman_M7.3_c20.37_sigma45.166_beta0.85.csv"
    SIDM_file5_extra = "CSIDM_Gilman_M7.4_c20.09_sigma47.236_beta0.85.csv"
    SIDM_file6_extra = "CSIDM_Gilman_M7.5_c19.81_sigma46.855_beta0.85.csv"

    # --- select files: rSIDM
    # BENCHMARK: GILMAN (EXTRAPOLATION)
    # RSIDM_file1_extra = "RSIDM_Gilman_M6.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    # RSIDM_file2_extra = "RSIDM_Gilman_M6.2_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    # RSIDM_file3_extra = "RSIDM_Gilman_M6.4_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    # RSIDM_file4_extra = "RSIDM_Gilman_M6.6_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    # RSIDM_file5_extra = "RSIDM_Gilman_M6.8_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    RSIDM_file1_extra = "RSIDM_Gilman_M7.0_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    RSIDM_file2_extra = "RSIDM_Gilman_M7.1_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    RSIDM_file3_extra = "RSIDM_Gilman_M7.2_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    RSIDM_file4_extra = "RSIDM_Gilman_M7.3_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    RSIDM_file5_extra = "RSIDM_Gilman_M7.4_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"
    RSIDM_file6_extra = "RSIDM_Gilman_M7.5_beta0.85_RadiiPerDec100_Ndec4.0_deltaT-4.0.csv"

    # --- select parameters values
    # sigma_m_arr = [10.713, 10.899, 11.512, 13.645, 19.634]
    # M_power_arr = [6.0, 6.2, 6.4, 6.6, 6.8]
    sigma_m_arr = [30.545, 35.434, 41.033, 45.166, 47.236, 46.855]
    M_power_arr = [7.0, 7.1, 7.2, 7.3, 7.4, 7.5]
    con_arr=[21.21, 20.92, 20.64, 20.37, 20.09, 19.81]


    # --- SIDM
    SIDM_1 = gravoF.create_gravothermalData_from_file(SIDM_file1_extra, DATA_cSIDM_ben3, beta=beta)
    SIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    SIDM_1.put_extra_parameters(10 ** M_power_arr[0], sigma_m_arr[0])

    SIDM_2 = gravoF.create_gravothermalData_from_file(SIDM_file2_extra, DATA_cSIDM_ben3, beta=beta)
    SIDM_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    SIDM_2.put_extra_parameters(10 ** M_power_arr[1], sigma_m_arr[1])

    SIDM_3 = gravoF.create_gravothermalData_from_file(SIDM_file3_extra, DATA_cSIDM_ben3, beta=beta)
    SIDM_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    SIDM_3.put_extra_parameters(10 ** M_power_arr[2], sigma_m_arr[2])

    SIDM_4 = gravoF.create_gravothermalData_from_file(SIDM_file4_extra, DATA_cSIDM_ben3, beta=beta)
    SIDM_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    SIDM_4.put_extra_parameters(10 ** M_power_arr[3], sigma_m_arr[3])

    SIDM_5 = gravoF.create_gravothermalData_from_file(SIDM_file5_extra, DATA_cSIDM_ben3, beta=beta)
    SIDM_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    SIDM_5.put_extra_parameters(10 ** M_power_arr[4], sigma_m_arr[4])

    SIDM_6 = gravoF.create_gravothermalData_from_file(SIDM_file6_extra, DATA_cSIDM_ben3, beta=beta)
    SIDM_6.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    SIDM_6.put_extra_parameters(10 ** M_power_arr[5], sigma_m_arr[5])

    # --- RSIDM
    rSIDM_1 = gravoF.create_gravothermalData_from_file(RSIDM_file1_extra, Data_rSIMD_ben3, beta=beta)
    rSIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_1.put_extra_parameters(10 ** M_power_arr[0], sigma_m_arr[0])

    rSIDM_2 = gravoF.create_gravothermalData_from_file(RSIDM_file2_extra, Data_rSIMD_ben3, beta=beta)
    rSIDM_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_2.put_extra_parameters(10 ** M_power_arr[1], sigma_m_arr[1])

    rSIDM_3 = gravoF.create_gravothermalData_from_file(RSIDM_file3_extra, Data_rSIMD_ben3, beta=beta)
    rSIDM_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_3.put_extra_parameters(10 ** M_power_arr[2], sigma_m_arr[2])

    rSIDM_4 = gravoF.create_gravothermalData_from_file(RSIDM_file4_extra, Data_rSIMD_ben3, beta=beta)
    rSIDM_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_4.put_extra_parameters(10 ** M_power_arr[3], sigma_m_arr[3])

    rSIDM_5 = gravoF.create_gravothermalData_from_file(RSIDM_file5_extra, Data_rSIMD_ben3, beta=beta)
    rSIDM_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_5.put_extra_parameters(10 ** M_power_arr[4], sigma_m_arr[4])

    rSIDM_6 = gravoF.create_gravothermalData_from_file(RSIDM_file6_extra, Data_rSIMD_ben3, beta=beta)
    rSIDM_6.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
    rSIDM_6.put_extra_parameters(10 ** M_power_arr[5], sigma_m_arr[5])

    # ------------------ DATA COLLECTION ------------------ #
    sidm_models = [
        SIDM_1, SIDM_2, SIDM_3, SIDM_4, SIDM_5, SIDM_6
    ]

    rsidm_models = [
        rSIDM_1, rSIDM_2, rSIDM_3, rSIDM_4, rSIDM_5, rSIDM_6
    ]


    # WORKS ONLY WHEN cSIDM REACHES UNIVERSALITY
    markers = ['o', 's', '^']

    # ordered colormap
    # cmap = mpl.cm.cividis
    # cmap = mpl.cm.viridis
    cmap = mpl.cm.plasma

    norm = mpl.colors.Normalize(vmin=min(M_power_arr), vmax=max(M_power_arr))
    colors = [cmap(norm(m)) for m in M_power_arr]

    from matplotlib.lines import Line2D

    # ------------------ PLOT ------------------ #
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e0, 3e3)

    ax.set_xlabel(r'$t\,[\mathrm{Gyr}]$', fontsize=18)
    ax.set_ylabel(r'$\hat{\rho}_c \equiv \rho_c/\rho_s$', fontsize=18)
    # ax.set_title(r'Core-density evolution: sensitivity to halo parameters (c-M relation)', fontsize=17)
    ax.set_title(r'Core-density evolution: cSIDM vs rSIDM', fontsize=18)

    ax.tick_params(labelsize=13)
    apply_scale_grid(ax, xscale="linear", yscale="log")

    index=0
    for i, (model, mlog, color, sigma_eff) in enumerate(zip(sidm_models, M_power_arr, colors, sigma_m_arr)):
        t, rho = model.return_rho_core_evolution(elements=2)
        time_c = rescale_time(model, sigma_eff)
        halo_rSIDM = TruncatedNFWProfile(_M_vir=10**M_power_arr[index], _con=con_arr[index])
        tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=sigma_m_arr[index])
        index=index+1

        ax.plot(
            # t, rho,
            t / tau_kappa, rho,
            linestyle='--',
            linewidth=2.3,
            # marker=markers[i],
            # markersize=5,
            # markevery=35,
            # markerfacecolor='white',
            # markeredgewidth=1.2,
            color=color,
            alpha=0.8,
            label=rf'$10^{{{mlog:.1f}}}\,M_\odot$'
        )

    index=0
    for model, mlog, color, sigma_eff in zip(rsidm_models, M_power_arr, colors, sigma_m_arr):
        t, rho = model.return_rho_core_evolution(elements=2)
        # time_c = rescale_time(model, sigma_eff)
        halo_rSIDM = TruncatedNFWProfile(_M_vir=10**M_power_arr[index], _con=con_arr[index])
        tau_kappa = halo_rSIDM.tau(beta=0.85, sigma_eff=sigma_m_arr[index])
        index=index+1

        ax.plot(
            # t, rho,
            t / tau_kappa, rho,
            linestyle='-',
            linewidth=2.3,
            color=color,
            alpha=0.95
        )

    # legend 1: halo mass
    leg1 = ax.legend(
        title=r'Halo mass',
        fontsize=10,
        title_fontsize=11,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.78),
        frameon=True,
        framealpha=0.9
    )
    ax.add_artist(leg1)

    # legend 2: model type
    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='cSIDM'),
        Line2D([0], [0], color='black', lw=2.3, linestyle='-', label='rSIDM'),
    ]
    ax.legend(
        handles=style_handles,
        title='Model',
        fontsize=10,
        title_fontsize=11,
        loc='upper left',
        frameon=True,
        framealpha=0.9
    )

    plt.tight_layout()
    if plot_save:
        plt.savefig(f'./{output_plot_dir}/core_density_evolution_csidm.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()