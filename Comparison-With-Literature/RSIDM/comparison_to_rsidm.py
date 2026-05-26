import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import RprocedureData as rprocF
from collections import defaultdict
import numpy as np
import units as uni


def read_fast_v2(path: str):
    """Read t, r, rho from messy whitespace/CSV-ish file; ignore garbage lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            arr = np.fromstring(s.replace(",", " "), sep=" ")
            if arr.size >= 2:
                rows.append(arr[:2])  # t, rho

    if not rows:
        raise ValueError("No numeric rows found.")
    data = np.vstack(rows)
    return data[:, 0], data[:, 1]


def read_fast(path: str):
    """Read t, r, rho from messy whitespace/CSV-ish file; ignore garbage lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            arr = np.fromstring(s.replace(",", " "), sep=" ")
            if arr.size >= 3:
                rows.append(arr[:3])  # t, r, rho

    if not rows:
        raise ValueError("No numeric rows found.")
    data = np.vstack(rows)
    return data[:, 0], data[:, 1], data[:, 2]

def central_density_evolution_blocks(t, r, rho, n_inner=3, combine="median", tol=0.0):
    """
    Assumes file structure: consecutive blocks with constant time t, each with many radii.
    Picks n_inner smallest radii within each block and combines their rho.
    """
    t = np.asarray(t); r = np.asarray(r); rho = np.asarray(rho)

    # keep finite, positive values for log scales
    m = np.isfinite(t) & np.isfinite(r) & np.isfinite(rho) & (r > 0) & (rho > 0)
    t, r, rho = t[m], r[m], rho[m]
    if t.size == 0:
        return np.array([]), np.array([])

    # find block boundaries
    if tol == 0.0:
        change = np.where(t[1:] != t[:-1])[0] + 1
    else:
        change = np.where(np.abs(t[1:] - t[:-1]) > tol)[0] + 1

    starts = np.r_[0, change]
    ends   = np.r_[change, t.size]

    times, rho_c = [], []

    for s, e in zip(starts, ends):
        rr = r[s:e]
        rh = rho[s:e]
        if rr.size == 0:
            continue

        k = min(n_inner, rr.size)

        # pick k smallest radii (fast)
        idx = np.argpartition(rr, k - 1)[:k]
        inner_rho = rh[idx]

        if combine == "median":
            rc = np.median(inner_rho)
        elif combine == "mean":
            rc = np.mean(inner_rho)
        else:
            raise ValueError("combine must be 'median' or 'mean'")

        times.append(t[s])
        rho_c.append(rc)

    return np.asarray(times), np.asarray(rho_c)



# ##################################### SETTINGS ##################################### #
plot_save = False
flag_sigma = False
output_plot_dir = "Results"

# --- the base directory will be our directory
OUR_DIR = os.path.dirname(os.path.abspath(__file__))

# --- select gravothermal directory
# gravothermal
DATA_GRAVOTHERMAL = os.path.join(OUR_DIR, 'Data_Extraction')
# isothermal
DATA_ISOTHERMAL_NFW_LoDen_DIR = os.path.join(OUR_DIR, 'Data_Extraction')
DATA_ISOTHERMAL_NFW_HiDen_DIR = os.path.join(OUR_DIR, 'Data_Extraction')

# --- select files:
gravothermal_file_1 = 'ρσ_sol_M_10._t_20_sigma_beta_0.75.csv'
path_gravothermal_file_1 = './Data_Extraction/ρσ_sol_M_10._t_20_sigma_beta_0.75.csv'

# gravothermal_file_1 = 'ρσ_sol_M_11._t_20_sigma_beta_0.75.csv'
# path_gravothermal_file_1 = './Data_Extraction/ρσ_sol_M_11._t_20_sigma_beta_0.75.csv'

# our Isothermal simulation results
isothermal_loDen_file = 'Riso_M_10.00000_t_0.359_sigma_2264.0_con_13.320.csv'
isothermal_HiDen_file = 'RisoHiDens_M_10.00000_t_0.359_sigma_m_2264.0_con_13.320.csv'

# isothermal_loDen_file = 'Riso_M_11.00000_t_10.131_sigma_70.0_con_10.556.csv'
# isothermal_HiDen_file = 'RisoHiDens_M_11.00000_t_10.131_sigma_m_70.0_con_10.556.csv'


# --- proportion parameter
# proportionOur = 1.32  # can be changed
proportionOur = 2.0  # can be changed
proportionJiang = 2.0  # must be

# ##################################### SET DATA TO CLASS ############################################################ #
# ------------------------------- GRAVOTHERMAL ------------------------------- #
# --- file with beta = 1.1
gravoEvolution_1 = gravoF.create_gravothermalData_from_file(gravothermal_file_1, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)


# ------------------------------- ISOTHERMAL (our results) ------------------------------- #
isoEvolution_our = dict()
# --- low dense
isoEvolution_our['LoDen'] = rprocF.create_RprocedureData_from_file(isothermal_loDen_file, DATA_ISOTHERMAL_NFW_LoDen_DIR)
isoEvolution_our['LoDen'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class

# --- high dense
isoEvolution_our['HiDen'] = rprocF.create_RprocedureData_from_file(isothermal_HiDen_file, DATA_ISOTHERMAL_NFW_HiDen_DIR)
isoEvolution_our['HiDen'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class

# --- mirror dense
isoEvolution_our['Mirror'] = rprocF.create_RprocedureDataMirror_from_file(isothermal_HiDen_file,
                                                                          DATA_ISOTHERMAL_NFW_HiDen_DIR,
                                                                          proportion=proportionOur)
isoEvolution_our['Mirror'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class


# ##################################### TAKE SIMULATION PARAMETERS ################################################### #
# --- from our simulation
ourIsoParameters = isoEvolution_our['LoDen'].return_parameters()
rho_s = ourIsoParameters["rho_s"]
r_s = ourIsoParameters["r_s"]
sigma_m = ourIsoParameters["sigma_m"]
ourTimeSteps = isoEvolution_our['LoDen'].time_steps


# ##################################### PLOTTING COMPARISON ########################################################## #
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

# ------------------ PLOT: ISOTHERMAL VS GRAVOTHERMAL: US ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\hat{\rho} = \rho/\rho_{s}$', fontsize=18)
plt.xlabel(r'$t$ [Gyr]',
           fontsize=18)
plt.title(r'RSIDM vs Isothermal, $10^{11}\, M_{\odot}$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA

# --- The evolution of central density for R-procedure simulation: OUR

iso_lowDen_central_time_l, iso_lowDen_central_rho_l = isoEvolution_our['LoDen'].return_central_rho()
ax.plot(iso_lowDen_central_time_l, iso_lowDen_central_rho_l,
        label='Isothermal (low dense)')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_our['HiDen'].return_central_rho()
ax.plot(iso_HigDen_central_time_l, iso_HigDen_central_rho_l,
        label='Isothermal (high dense)')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_our['Mirror'].return_central_rho()
ax.plot(iso_HigDen_central_time_l, iso_HigDen_central_rho_l,
        "--", label=f'Isothermal (mirror)')

# --- RSIDM
N_INNER = 3                                        # <- like in your code

t, r, rho = read_fast(path_gravothermal_file_1)
# if your file stores t = log10(Gyr) like in the big class
t = 10.0 ** t

times, rhoc = central_density_evolution_blocks(t, r, rho,
                                               n_inner=1,
                                               combine="median",
                                               tol=0.0)

# drop nans if any
m = np.isfinite(times) & np.isfinite(rhoc) & (times > 0) & (rhoc > 0)
times, rhoc = times[m], rhoc[m]
# times = 1.3 * times
ax.plot(times, rhoc,
        "-", label=f'Gravothermal: RSIDM')

# --- sigma effective
if flag_sigma == True:
    path_sigma_eff = './Data_Extraction/GravoCodePython_simres10_eff.txt'
    t, rho = read_fast_v2(path_sigma_eff)
    time = 10.0**t
    ax.plot(time, rho, color="grey", ls="-",
            label=r'Gravothermal: $\sigma_{\mathrm{eff}} \sim 2264\, \mathrm{cm}^2/\mathrm{g}$')


# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'M11' + '.png', dpi=300)
else:
    plt.show()


# ------------------ PLOT: ISOTHERMAL VS GRAVOTHERMAL: US (TILDA-time) ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\hat{\rho} = \rho/\rho_{s}$', fontsize=18)
plt.xlabel(r'$\tilde{t}$ [Dimensionless]',
           fontsize=18)
plt.title(r'RSIDM vs Isothermal, $10^{11}\, M_{\odot}$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA

# --- The evolution of central density for R-procedure simulation: OUR

iso_lowDen_central_time_l, iso_lowDen_central_rho_l = isoEvolution_our['LoDen'].return_central_rho()
time_tilda = uni.time_tilda(iso_lowDen_central_time_l, rho_s, r_s, sigma_m)  # [dimensionless]
ax.plot(time_tilda, iso_lowDen_central_rho_l,
        label='Isothermal (low dense)')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_our['HiDen'].return_central_rho()
time_tilda = uni.time_tilda(iso_HigDen_central_time_l, rho_s, r_s, sigma_m)  # [dimensionless]
ax.plot(time_tilda, iso_HigDen_central_rho_l,
        label='Isothermal (high dense)')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_our['Mirror'].return_central_rho()
time_tilda = uni.time_tilda(iso_HigDen_central_time_l, rho_s, r_s, sigma_m)  # [dimensionless]
ax.plot(time_tilda, iso_HigDen_central_rho_l,
        "--", label=f'Isothermal (mirror)')

# --- RSIDM
N_INNER = 3                                        # <- like in your code

t, r, rho = read_fast(path_gravothermal_file_1)
# if your file stores t = log10(Gyr) like in the big class
t = 10.0 ** t

times, rhoc = central_density_evolution_blocks(t, r, rho,
                                               n_inner=1,
                                               combine="median",
                                               tol=0.0)

# drop nans if any
m = np.isfinite(times) & np.isfinite(rhoc) & (times > 0) & (rhoc > 0)
times, rhoc = times[m], rhoc[m]
# times = 1.3 * times

time_tilda = uni.time_tilda(times, rho_s, r_s, sigma_m)  # [dimensionless]
ax.plot(time_tilda, rhoc,
        "-", label=f'Gravothermal: RSIDM')

# --- sigma effective
if flag_sigma == True:
    path_sigma_eff = './Data_Extraction/GravoCodePython_simres10_eff.txt'
    t, rho = read_fast_v2(path_sigma_eff)
    time = 10.0**t
    time_tilda = uni.time_tilda(time, rho_s, r_s, sigma_m)  # [dimensionless]

    ax.plot(time_tilda, rho, color="grey", ls="-",
            label=r'Gravothermal: $\sigma_{\mathrm{eff}} \sim 2264\, \mathrm{cm}^2/\mathrm{g}$')


# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'M11_tilde' + '.png', dpi=300)
else:
    plt.show()
