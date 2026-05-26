"""
Shorts summary, presents results of mirror method and simulation that we already had.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import RprocedureData as rprocF
import units as uni
import config as cfg
from NFWProfile import NFWProfile  # CDM profile (halo)

# ##################################### SETTINGS ##################################### #
plot_save = True
output_directory = ""

# --- gravothermal
gravo_file_1 = "./Input/Gravothermal/ρσ_sol_M_10.699_t_2051.1_sigma_1.0_beta_0.3.csv"
gravo_file_2 = "./Input/Gravothermal/ρσ_sol_M_10.699_t_2051.1_sigma_1.0_beta_0.4.csv"
gravo_file_3 = "./Input/Gravothermal/ρσ_sol_M_10.699_t_2200.42_sigma_1.0_beta_0.5.csv"


# --- gravothermal + Rprocedure
gravoAndRproc_file = "./Input/RprocedureAndGravothermal/Riso_minDen_sol_M_10.699_t_2200.43_sigma_1.0_beta_0.5.csv"

# --- Rprocedure
# Riso_file = "./Input/Rprocedure/Riso_M_10.69897_t_855.204_sigma_1.0_con_11.322.csv"
Riso_file = "./Input/Rprocedure/Riso_M_10.69897_t_880.172_sigma_1.0_con_11.322.csv"

# --- Rprocedure High dense (to mirror method)
# Riso_HiDens_file = "./Input/Rprocedure/RisoHiDens_M_10.69897_t_855.204_sigma_1.0_con_11.322.csv"
Riso_HiDens_file = "./Input/Rprocedure/RisoHiDens_M_10.69897_t_880.880_sigma_1.0_con_11.322.csv"

# stable solution of high solution. It is estimated, where high solution end below the line of our accuracy
# of calculation
r_1_at_coll = 1.99  # [r_s], where collapse should appear
proportion = 2.0  # proportion between tmerge and tcollapse

# ##################################### IMPORT DATA ##################################### #
# ------------------------------- GRAVOTHERMAL ------------------------------- #
# --- FILE 1
#  create map, which contains all data coming from file created by mathematica code
gravo_data_1 = pd.read_csv(gravo_file_1, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravo_data_1 = gravo_data_1.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
NFW_class_1 = gravoF.GravothermalData(gravo_data_1['t'], gravo_data_1['r'], gravo_data_1['rho'],
                                      gravo_data_1['velDis'], beta=0.3)

NFW_class_1.put_the_name("NFW")  # put the name to differentiate data in class

# --- FILE 2
#  create map, which contains all data coming from file created by mathematica code
gravo_data_2 = pd.read_csv(gravo_file_2, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravo_data_2 = gravo_data_2.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
NFW_class_2 = gravoF.GravothermalData(gravo_data_2['t'], gravo_data_2['r'], gravo_data_2['rho'],
                                      gravo_data_2['velDis'], beta=0.4)

NFW_class_2.put_the_name("NFW")  # put the name to differentiate data in class

# --- FILE 3
#  create map, which contains all data coming from file created by mathematica code
gravo_data_3 = pd.read_csv(gravo_file_3, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravo_data_3 = gravo_data_3.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
NFW_class_3 = gravoF.GravothermalData(gravo_data_3['t'], gravo_data_3['r'], gravo_data_3['rho'],
                                      gravo_data_3['velDis'], beta=0.5)

NFW_class_3.put_the_name("NFW")  # put the name to differentiate data in class

# ------------------------------- GRAVOTHERMAL AND R-PROCEDURE ------------------------------- #
#  create map, which contains all data coming from file created by mathematica code: rho
gravoRpro_data_1 = pd.read_csv(gravoAndRproc_file, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravoRpro_data_1 = gravoRpro_data_1.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
ISO_class_1 = gravoF.GravothermalData(gravoRpro_data_1['t'], gravoRpro_data_1['r'], gravoRpro_data_1['rho'],
                                      gravoRpro_data_1['velDis'], beta=0.5)

ISO_class_1.put_the_name("Isothermal")  # put the name to differentiate data in class

# ------------------------------- RPROCEDURE: ISO + NFW ------------------------------- #
#  create map, which contains all data coming from file created by mathematica code: rho
isoAndNFW_data = pd.read_csv(Riso_file, sep='\t', names=['t', 'r', 'rho', 'mass', 'velDis',
                                                         'central-rho', 'central-velDis',
                                                         'names', 'values'])

# create class, which will contain all the necessary data
RISO_class = rprocF.RprocedureData(isoAndNFW_data['t'], isoAndNFW_data['r'], isoAndNFW_data['rho'],
                                   isoAndNFW_data['velDis'], isoAndNFW_data['names'], isoAndNFW_data['values'])

RISO_class.put_the_name("Isothermal")  # put the name to differentiate data in class

# ------------------------------- RPROCEDURE: ISO + NFW (High Dense) ------------------------------- #
#  create map, which contains all data coming from file created by mathematica code: rho
Mirror_data = pd.read_csv(Riso_HiDens_file, sep='\t', names=['t', 'r', 'rho', 'mass', 'velDis',
                                                             'central-rho', 'central-velDis',
                                                             'names', 'values'])

# --- FIRSTS CLASS: PROPORTION PARAMETER
# create class, which will contain all the necessary data
Mirror_class_1 = rprocF.RprocedureDataMirror(Mirror_data['t'], Mirror_data['r'], Mirror_data['rho'],
                                             Mirror_data['velDis'], Mirror_data['names'], Mirror_data['values'],
                                             proportion=proportion)

# ##################################### PLOTTING: SETTINGS ##################################### #
# --- find values of simulation parameters
Mvir, sigma_m, const_c = RISO_class.return_basic_parameters()
logMvir = np.log10(Mvir)

# --- lines from Rprocedure evolution and time
beforeMerge_step = RISO_class.time_steps - 1  # We have to subtract one!!!
afterMerge_step = Mirror_class_1.time_steps - 1  # We have to subtract one!!!
Rres = 0.01  # [kpc]
r_s = RISO_class.parameters["r_s"]  # [kpc]
rho_s = RISO_class.parameters["rho_s"]
r_1 = RISO_class.parameters["r1"]  # [kpc] at tmerge!
# --- Stable High Dense
t_stableMirror_SU = Mirror_class_1.find_collapse(r_1_at_coll=r_1_at_coll)[1]
t_stableHiDen_SU = proportion * RISO_class.data['time'][-1] - t_stableMirror_SU

print("*************** BASIC SETTINGS ***************")
print("time steps in Rprocedure:", RISO_class.time_steps)
print(f"r_1 at tmerge: {r_1} [kpc]")
print(f"r_s: {r_s} [kpc]")
print(f"rho_s: {rho_s} [M_sun/kpc^3]")
print(f"dark matter concentration: {const_c} [dimensionless]")
print(f"Halo mass: {Mvir} [M_sun]")
print(f"tmerge: {RISO_class.data['time'][-1]} [Gyr]")
print(f"annihilation cross section: {sigma_m} [cm^2/g]")
print("")
print("*************** Collapse: Gravo and Rproc ***************")
data_collapse = Mirror_class_1.find_collapse()
print(f"Collapse in time {data_collapse[1]} [Gyr]")
print(f"Collapse in time {uni.time_tilda(data_collapse[1], rho_s, r_s, sigma_m)} [dimensionless]")
print(f"Collapse in time step {data_collapse[2]}")
print(f"Collapse density {data_collapse[3]} [rho_s]")
print("")
print("*************** MIRROR SETTING ***************")
print(f"stable solution in high dense solution: {t_stableHiDen_SU} [Gyr]")
print(f"stable solution in mirror method: {t_stableMirror_SU} [Gyr]")


# ##################################### SCALE ##################################### #
# we want to have same units as in the `2305.05067`
def time_units(time):
    """
    function to proper scale time.

    where,
        time: evolution time of galaxy [Gyr] (float)
    """
    sigma_m_SI = sigma_m * 10 ** (-4) * 10 ** 3  # [m^2/kg]
    sigma_m_SU = sigma_m_SI * cfg.kpc_SI ** (-2) * cfg.M_solar_SI  # [kpc^2 * M_sun^(-1)]
    return sigma_m_SU * time * rho_s * r_s * np.sqrt(4 * np.pi * cfg.const_G_starUnits * rho_s)

def vrms_units(vel_dis):
    """
    function to proper scale velocity dispersion.

    where,
        vel_dis: velocity dispersion of DM [r_s/Gyr] (float)
    """
    if isinstance(vel_dis, list) or isinstance(vel_dis, np.ndarray):
        vel_dis_arr = np.array([])
        for vel in vel_dis:
            vel_dis_SU = vel * r_s  # [kpc/Gyr]
            vel_dis_hat = vel_dis_SU / (r_s * np.sqrt(4 * np.pi * cfg.const_G_starUnits * rho_s))  # [dimensionless]
            # append
            vel_dis_arr = np.append(vel_dis_arr, vel_dis_hat)

        return vel_dis_arr
    else:
        vel_dis_SU = vel_dis * r_s  # [kpc/Gyr]
        return vel_dis_SU / (r_s * np.sqrt(4 * np.pi * cfg.const_G_starUnits * rho_s))  # [dimensionless]
def vrms_units_NFW(vel_dis):
    """
    function to proper scale velocity dispersion.

    where,
        vel_dis: velocity dispersion of DM [kpc/Gyr] (float)
    """
    if isinstance(vel_dis, list) or isinstance(vel_dis, np.ndarray):
        vel_dis_arr = np.array([])
        for vel in vel_dis:
            vel_dis_hat = vel / (r_s * np.sqrt(4 * np.pi * cfg.const_G_starUnits * rho_s))  # [dimensionless]
            # append
            vel_dis_arr = np.append(vel_dis_arr, vel_dis_hat)

        return vel_dis_arr
    else:
        return vel_dis / (r_s * np.sqrt(4 * np.pi * cfg.const_G_starUnits * rho_s))  # [dimensionless]

# ##################################### PLOTTING: PLOTS ##################################### #
# ------------------ PLOT: CENTRAL DENSITY (Rproc) AND CORE DENSITY (GRAVO) ------------------ #
fig, ax = plt.subplots(figsize=(14.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
# plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
plt.xlabel(r'$\tilde{t} \ \left[dimensionless \right]$', fontsize=18)

plt.title(f'Central density, for: M = 10^{"{:.5f}".format(logMvir)} [M_sun], '
          f'sigma_m = {"{:.1f}".format(sigma_m)} [cm^2/g].', fontsize=18)
ax.tick_params(labelsize=14)

# grid
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# for refined control of log-scale tick marks
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.yaxis.set_major_locator(locmaj)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('both', direction='in', top='on', right='on', length=10,
               width=1, which='major', zorder=301)
ax.tick_params('both', direction='in', top='on', right='on', length=5,
               width=1, which='minor', zorder=301)

# --- DATA
# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (initial: NFW), $\\beta$: {0.3}')

# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_2.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (initial: NFW), $\\beta$: {0.4}.')

# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_3.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (initial: NFW), $\\beta$: {0.5}.')

# --- ISO: LOW DENSE SOLUTION
isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l = RISO_class.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(isoAndNFW_gravo_time_l, rho_s, r_s, sigma_m), isoAndNFW_gravo_rho_l[:len(isoAndNFW_gravo_time_l)],
        label=f'Rprocedure (ISO)')

# --- ISO: HIGH DENSE SOLUTION
mirror_time_l, mirror_rho_l = Mirror_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        "--", color='black', label=f'Rprocedure (Mirror), proportion: {Mirror_class_1.data["proportion"]}')

# --- ISO AND GRAVOTHERMAL
isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l = ISO_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(isoAndNFW_gravo_time_l, rho_s, r_s, sigma_m), isoAndNFW_gravo_rho_l, ":", color='grey',
        label=f'gravothermal (initial: ISO at lowest density), $\\beta$: {0.50}')

# --- CHARACTERISTIC POINTS
# --- MIRROR SWITCH
mirrorSwitch_time = uni.time_tilda(mirror_time_l[-1], rho_s, r_s, sigma_m)
mirrorSwitch_rho = mirror_rho_l[-1]
ax.scatter(mirrorSwitch_time, mirrorSwitch_rho, marker="*", s=70.0, zorder=101,
           label=f'mirror switch',
           facecolor='yellow', edgecolor='black', linewidth=1.0, rasterized=True)
# --- MAXIMUM CORE
maxCore_time = uni.time_tilda(isoAndNFW_gravo_time_l[0], rho_s, r_s, sigma_m)
maxCore_rho = isoAndNFW_gravo_rho_l[0]
ax.scatter(maxCore_time, maxCore_rho, marker="X", s=50.0, zorder=102,
           label=f'maximum core',
           facecolor='seagreen', edgecolor='black', linewidth=1.0, rasterized=True)

# --- LEGEND AND SAVE
ax.legend(fontsize=12)
if plot_save is True:
    plt.savefig('./' + output_directory + '/summaryPlot' + '.png', dpi=300)

    plt.close(fig)
else:
    plt.show()

# ------------------ PLOT: CENTRAL DENSITY (Rproc) AND CORE DENSITY (GRAVO): DIFFERENT UNITS ------------------ #
fig, ax = plt.subplots(figsize=(14.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$\hat{\left(\sigma / m\right)} \hat{t}$', fontsize=18)

plt.title(f'Central density, for: M = 10^{"{:.5f}".format(logMvir)} [M_sun], '
          f'sigma_m = {"{:.1f}".format(sigma_m)} [cm^2/g].', fontsize=18)
ax.tick_params(labelsize=14)

# grid
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# for refined control of log-scale tick marks
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.yaxis.set_major_locator(locmaj)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('both', direction='in', top='on', right='on', length=10,
               width=1, which='major', zorder=301)
ax.tick_params('both', direction='in', top='on', right='on', length=5,
               width=1, which='minor', zorder=301)

# --- DATA
# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_1.return_rho_core_evolution(elements=2)
ax.plot(time_units(gravo_time_l), gravo_rho_l,
        label=f'gravothermal (initial: NFW), $\\beta$: {0.3}')

# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_2.return_rho_core_evolution(elements=2)
ax.plot(time_units(gravo_time_l), gravo_rho_l,
        label=f'gravothermal (initial: NFW), $\\beta$: {0.4}.')

# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_3.return_rho_core_evolution(elements=2)
ax.plot(time_units(gravo_time_l), gravo_rho_l,
        label=f'gravothermal (initial: NFW), $\\beta$: {0.5}.')

# --- ISO: LOW DENSE SOLUTION
isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l = RISO_class.return_rho_core_evolution(elements=2)
ax.plot(time_units(isoAndNFW_gravo_time_l), isoAndNFW_gravo_rho_l[:len(isoAndNFW_gravo_time_l)],
        label=f'Rprocedure (ISO)')

# --- ISO: HIGH DENSE SOLUTION
mirror_time_l, mirror_rho_l = Mirror_class_1.return_rho_core_evolution(elements=2)
ax.plot(time_units(mirror_time_l), mirror_rho_l[:len(mirror_time_l)],
        "--", color='black', label=f'Rprocedure (Mirror), proportion: {Mirror_class_1.data["proportion"]}')

# --- ISO AND GRAVOTHERMAL
isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l = ISO_class_1.return_rho_core_evolution(elements=2)
ax.plot(time_units(isoAndNFW_gravo_time_l), isoAndNFW_gravo_rho_l, ":", color='grey',
        label=f'gravothermal (initial: ISO at lowest density), $\\beta$: {0.50}')

# --- CHARACTERISTIC POINTS
# --- MIRROR SWITCH
mirrorSwitch_time = time_units(mirror_time_l[-1])
mirrorSwitch_rho = mirror_rho_l[-1]
ax.scatter(mirrorSwitch_time, mirrorSwitch_rho, marker="*", s=70.0, zorder=101,
           label=f'mirror switch',
           facecolor='yellow', edgecolor='black', linewidth=1.0, rasterized=True)
# --- MAXIMUM CORE
maxCore_time = time_units(isoAndNFW_gravo_time_l[0])
maxCore_rho = isoAndNFW_gravo_rho_l[0]
ax.scatter(maxCore_time, maxCore_rho, marker="X", s=50.0, zorder=102,
           label=f'maximum core',
           facecolor='seagreen', edgecolor='black', linewidth=1.0, rasterized=True)

# --- LEGEND AND SAVE
ax.legend(fontsize=12)
if plot_save is True:
    plt.savefig('./' + output_directory + '/summaryPlot-different-units' + '.png', dpi=300)

    plt.close(fig)
else:
    plt.show()

# #################################### COMPARISON BETWEEN GRAVO AND RPROC #################################### #
# --- SET UP THE FIGURE WINDOW
fig1 = plt.figure(figsize=(13, 10.0), dpi=80, facecolor='w', edgecolor='k')
fig1.subplots_adjust(left=0.16, right=0.93, bottom=0.12, top=0.91,
                     hspace=0.0, wspace=0.32)
gs = gridspec.GridSpec(3, 2)
# --- SET UP TO NFW
plot_NFW_r = np.logspace(-2.0, 2.0, num=500)  # [r_s]
plot_NFW_logR = np.log10(plot_NFW_r)  # [log r_s]
NFW_r = [plot_NFW_r[i] * r_s for i in range(len(plot_NFW_r))]  # [kpc]


# -------------------------- PLOT: DENSITY PROFILE in RPROC -------------------------- #
ax = fig1.add_subplot(gs[0, 1])
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe profile
plt.ylabel(r"$\hat{\rho}$", fontsize=18)
# plt.xlabel(r"$\hat{r}$", fontsize=18)
plt.title("Isothermal Solutions", fontsize=18)

# --- DATA
# --- ISO: LOWEST CENTRAL DENSITY
RISO_data = RISO_class.return_data_last_moment_forming()
ax.plot(RISO_data['r'], RISO_data['rho'], label='maximum core')
# --- ISO: BEFORE MIRROR SWITCH
RISO_data = RISO_class.return_data_at_fixed_time(beforeMerge_step)
ax.plot(RISO_data['r'], RISO_data['rho'], label='before mirror switch')
# --- ISO: AFTER MIRROR SWITCH
Mirror_data = Mirror_class_1.return_data_at_fixed_time(afterMerge_step)
ax.plot(Mirror_data['r'], Mirror_data['rho'], "--", label='after mirror switch')
# --- ISO: COLLAPSE
Mirror_data = Mirror_class_1.return_data_at_collapse(r_1_at_coll=r_1_at_coll)
ax.plot(Mirror_data['r'], Mirror_data['rho'], "--", label='core collapse')
# --- NFW: CDM-only
NFW_profile = NFWProfile(Mvir, const_c)
NFW_rho = [NFW_profile.rho(x) / rho_s for x in NFW_r]  # [rho_s]
ax.plot(plot_NFW_r, NFW_rho, "grey", label='NFW: CDM-only')

# grid
ax.tick_params(labelsize=14)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# for refined control of log-scale tick marks
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.yaxis.set_major_locator(locmaj)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('both', direction='in', top=0, right=0, length=10,
               width=1, which='major', zorder=301)
ax.tick_params('both', direction='in', top=0, right=0, length=5,
               width=1, which='minor', zorder=301)
plt.xticks(visible=False)  # to hide values under plot

# lines
plt.axvline(x=r_1_at_coll, color='black', linestyle='dashdot',
            label=rf'$\hat{{r}}_{{1}}(t = t_{{coll}}) = {"{:.2f}".format(r_1_at_coll)}$, stable high solution')
plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9)
# label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
# annotation to lines
ax.text(r_1_at_coll, 0.5 * ax.get_ylim()[1], r'$\hat{r}_{1}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)

ax.legend(loc='lower left', fontsize=9)

# -------------------------- PLOT: VELOCITY DISPERSION in RPROC -------------------------- #
ax = fig1.add_subplot(gs[1, 1])
# log scale
ax.set_xscale('log')
# describe
plt.ylabel(r"$\hat{V}_{rms}$ ", fontsize=18)
# plt.xlabel(r"$\hat{r}$", fontsize=18)

# --- DATA
# --- ISO: LOWEST CENTRAL DENSITY
RISO_data = RISO_class.return_data_last_moment_forming()
ax.plot(RISO_data['r'], vrms_units(RISO_data['velDis']), label='maximum core')
# --- ISO: BEFORE MIRROR SWITCH
RISO_data = RISO_class.return_data_at_fixed_time(beforeMerge_step)
ax.plot(RISO_data['r'], vrms_units(RISO_data['velDis']), label='before mirror switch')
# --- ISO: AFTER MIRROR SWITCH
Mirror_data = Mirror_class_1.return_data_at_fixed_time(afterMerge_step)
ax.plot(Mirror_data['r'], vrms_units(Mirror_data['velDis']), "--", label='after mirror switch')
# --- ISO: COLLAPSE
Mirror_data = Mirror_class_1.return_data_at_collapse(r_1_at_coll=r_1_at_coll)
ax.plot(Mirror_data['r'], vrms_units(Mirror_data['velDis']), "--", label='core collapse')
# --- NFW: CDM-only
NFW_sigma = [vrms_units_NFW(NFW_profile.sigma_accurate(x)) for x in NFW_r]
ax.plot(plot_NFW_r, NFW_sigma, "grey", label='NFW: CDM-only')

# grid
ax.tick_params(labelsize=14)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# for refined control of log-scale tick marks
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('x', direction='in', top=0, right=0, length=10,
               width=1, which='major', zorder=301)
ax.tick_params('x', direction='in', top=0, right=0, length=5,
               width=1, which='minor', zorder=301)
ax.tick_params('y', direction='in', top=0, right=0, length=5,
               width=1, which='both', zorder=301)
plt.xticks(visible=False)  # to hide values under plot

# lines
plt.axvline(x=r_1_at_coll, color='black', linestyle='dashdot',
            label=rf'$\hat{{r}}_{{1}}(t = t_{{coll}}) = {"{:.2f}".format(r_1_at_coll)}$, stable high solution')
plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9)
# label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
# annotation to lines
ax.text(r_1_at_coll, 0.5 * ax.get_ylim()[1], r'$\hat{r}_{1}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)

# ax.legend()


# -------------------------- PLOT: DERIVATIVE OF DENSITY in RPROC -------------------------- #
ax = fig1.add_subplot(gs[2, 1])
# log scale
ax.set_xscale('log')
# describe
plt.ylabel(r"$\log\hat{\rho}/\log\hat{r}$ ", fontsize=18)
plt.xlabel(r"$\hat{r}$", fontsize=18)

# --- DATA
# --- ISO: LOWEST CENTRAL DENSITY
RISO_data = RISO_class.return_data_last_moment_forming()
logRho = np.log10(RISO_data['rho'])
logR = np.log10(RISO_data['r'])
dlogRho_dlogR = np.gradient(logRho, logR)
ax.plot(RISO_data['r'], dlogRho_dlogR, label='maximum core')
# --- ISO: BEFORE MIRROR SWITCH
RISO_data = RISO_class.return_data_at_fixed_time(beforeMerge_step)
logRho = np.log10(RISO_data['rho'])
logR = np.log10(RISO_data['r'])
dlogRho_dlogR = np.gradient(logRho, logR)
ax.plot(RISO_data['r'], dlogRho_dlogR, label='before mirror switch')
# --- ISO: AFTER MIRROR SWITCH
Mirror_data = Mirror_class_1.return_data_at_fixed_time(afterMerge_step)
logRho = np.log10(Mirror_data['rho'])
logR = np.log10(Mirror_data['r'])
dlogRho_dlogR = np.gradient(logRho, logR)
ax.plot(Mirror_data['r'], dlogRho_dlogR, "--", label='after mirror switch')
# --- ISO: COLLAPSE
Mirror_data = Mirror_class_1.return_data_at_collapse(r_1_at_coll=r_1_at_coll)
logRho = np.log10(Mirror_data['rho'])
logR = np.log10(Mirror_data['r'])
dlogRho_dlogR = np.gradient(logRho, logR)
ax.plot(Mirror_data['r'], dlogRho_dlogR, "--", label='core collapse')
# --- NFW: CDM-only
NFW_logRho = [np.log10(NFW_profile.rho(x) / rho_s) for x in NFW_r]  # [log rho_s]
NFW_dlogRho_dlogR = np.gradient(NFW_logRho, plot_NFW_logR)  # [d log rho_s / d log r_s]
ax.plot(plot_NFW_r, NFW_dlogRho_dlogR, "grey", label='NFW: CDM-only')

# grid
ax.tick_params(labelsize=14)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# for refined control of log-scale tick marks
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('x', direction='in', top=0, right=0, length=10,
               width=1, which='major', zorder=301)
ax.tick_params('x', direction='in', top=0, right=0, length=5,
               width=1, which='minor', zorder=301)
ax.tick_params('y', direction='in', top=0, right=0, length=5,
               width=1, which='both', zorder=301)

# lines
plt.axvline(x=r_1_at_coll, color='black', linestyle='dashdot',
            label=rf'$\hat{{r}}_{{1}}(t = t_{{coll}}) = {"{:.2f}".format(r_1_at_coll)}$, stable high solution')
plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9)
# label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
# annotation to lines
ax.text(r_1_at_coll, 0.5 * ax.get_ylim()[1], r'$\hat{r}_{1}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)

# ax.legend()


# -------------------------- PLOT: DENSITY PROFILE in GRAVO -------------------------- #
ax = fig1.add_subplot(gs[0, 0])
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe profile
plt.ylabel(r"$\hat{\rho}$", fontsize=18)
# plt.xlabel(r"$\hat{r}$", fontsize=18)
plt.title(rf"Gravothermal Solutions, $\beta = {0.4}$", fontsize=18)

# --- DATA
# --- GRAVO: MAXIMUM CORE
GRAVO_data = NFW_class_2.return_data_last_moment_forming()
ax.plot(GRAVO_data['r'], GRAVO_data['rho'], label='maximum core')
# --- GRAVO: CORE COLLAPSE
time_step_collapse = NFW_class_2.find_collapse(fixed_limit=10**15)[2] - 1
GRAVO_data = NFW_class_2.return_data_at_fixed_time(time_step_collapse)
ax.plot(GRAVO_data['r'], GRAVO_data['rho'], label='core collapse')
# --- NFW: CDM-only
NFW_rho = [NFW_profile.rho(x) / rho_s for x in NFW_r]  # [log rho_s]
ax.plot(plot_NFW_r, NFW_rho, "grey", label='NFW: CDM-only')

# grid
ax.tick_params(labelsize=14)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# for refined control of log-scale tick marks
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.yaxis.set_major_locator(locmaj)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('both', direction='in', top=0, right=0, length=10,
               width=1, which='major', zorder=301)
ax.tick_params('both', direction='in', top=0, right=0, length=5,
               width=1, which='minor', zorder=301)
plt.xticks(visible=False)  # to hide values under plot

# lines
plt.axvline(x=r_1_at_coll, color='black', linestyle='dashdot',
            label=rf'$\hat{{r}}_{{1}}(t = t_{{coll}}) = {"{:.2f}".format(r_1_at_coll)}$, stable high solution')
plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9)
# label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
# annotation to lines
ax.text(r_1_at_coll, 0.5 * ax.get_ylim()[1], r'$\hat{r}_{1}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)

ax.legend(loc='lower left', fontsize=9)

# -------------------------- PLOT: VELOCITY DISPERSION in RPROC -------------------------- #
ax = fig1.add_subplot(gs[1, 0])
# log scale
ax.set_xscale('log')
# describe
plt.ylabel(r"$\hat{V}_{rms}$ ", fontsize=18)
# plt.xlabel(r"$\hat{r}$", fontsize=18)

# --- DATA
# --- GRAVO: MAXIMUM CORE
GRAVO_data = NFW_class_2.return_data_last_moment_forming()
ax.plot(GRAVO_data['r'], vrms_units(GRAVO_data['velDis']), label='maximum core')
# --- GRAVO: CORE COLLAPSE
time_step_collapse = NFW_class_2.find_collapse(fixed_limit=10**15)[2] - 1
GRAVO_data = NFW_class_2.return_data_at_fixed_time(time_step_collapse)
ax.plot(GRAVO_data['r'], vrms_units(GRAVO_data['velDis']), label='core collapse')
# --- NFW: CDM-only
NFW_sigma = [vrms_units_NFW(NFW_profile.sigma_accurate(x)) for x in NFW_r]
ax.plot(plot_NFW_r, NFW_sigma, "grey", label='NFW: CDM-only')

# grid
ax.tick_params(labelsize=14)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# for refined control of log-scale tick marks
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('x', direction='in', top=0, right=0, length=10,
               width=1, which='major', zorder=301)
ax.tick_params('x', direction='in', top=0, right=0, length=5,
               width=1, which='minor', zorder=301)
ax.tick_params('y', direction='in', top=0, right=0, length=5,
               width=1, which='both', zorder=301)
plt.xticks(visible=False)  # to hide values under plot

# lines
plt.axvline(x=r_1_at_coll, color='black', linestyle='dashdot',
            label=rf'$\hat{{r}}_{{1}}(t = t_{{coll}}) = {"{:.2f}".format(r_1_at_coll)}$, stable high solution')
plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9)
# label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
# annotation to lines
ax.text(r_1_at_coll, 0.5 * ax.get_ylim()[1], r'$\hat{r}_{1}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)
# ax.legend()

# -------------------------- PLOT: DERIVATIVE OF DENSITY in GRAVO -------------------------- #
ax = fig1.add_subplot(gs[2, 0])
# log scale
ax.set_xscale('log')
# describe
plt.ylabel(r"$\log\hat{\rho}/\log\hat{r}$ ", fontsize=18)
plt.xlabel(r"$\hat{r}$", fontsize=18)

# --- DATA
# --- GRAVO: MAXIMUM CORE
GRAVO_data = NFW_class_2.return_data_last_moment_forming()
logRho = np.log10(GRAVO_data['rho'])
logR = np.log10(GRAVO_data['r'])
dlogRho_dlogR = np.gradient(logRho, logR)
ax.plot(GRAVO_data['r'], dlogRho_dlogR, label='maximum core')
# --- GRAVO: CORE COLLAPSE
time_step_collapse = NFW_class_2.find_collapse(fixed_limit=10**15)[2] - 1
GRAVO_data = NFW_class_2.return_data_at_fixed_time(time_step_collapse)
logRho = np.log10(GRAVO_data['rho'])
logR = np.log10(GRAVO_data['r'])
dlogRho_dlogR = np.gradient(logRho, logR)
ax.plot(GRAVO_data['r'], dlogRho_dlogR, label='core collapse')
# --- NFW: CDM-only
NFW_logRho = [np.log10(NFW_profile.rho(x) / rho_s) for x in NFW_r]  # [log rho_s]
NFW_dlogRho_dlogR = np.gradient(NFW_logRho, plot_NFW_logR)  # [d log rho_s / d log r_s]
ax.plot(plot_NFW_r, NFW_dlogRho_dlogR, "grey", label='NFW: CDM-only')

# grid
ax.tick_params(labelsize=14)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# for refined control of log-scale tick marks
locmaj = mticker.LogLocator(base=10, numticks=12)
locmin = mticker.LogLocator(base=10.0,
                            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('x', direction='in', top=0, right=0, length=10,
               width=1, which='major', zorder=301)
ax.tick_params('x', direction='in', top=0, right=0, length=5,
               width=1, which='minor', zorder=301)
ax.tick_params('y', direction='in', top=0, right=0, length=5,
               width=1, which='both', zorder=301)

# lines
plt.axvline(x=r_1_at_coll, color='black', linestyle='dashdot',
            label=rf'$\hat{{r}}_{{1}}(t = t_{{coll}}) = {"{:.2f}".format(r_1_at_coll)}$, stable high solution')
plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9)
# label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
# annotation to lines
ax.text(r_1_at_coll, 0.5 * ax.get_ylim()[1], r'$\hat{r}_{1}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)

# ax.legend()

# ---save figure
outfig = './' + output_directory + '/IsoVsGravo_logMvir%.3f_c%.1f_sigmamx%.1f.png'  # %(M_vir,c,sigmamx,tage)
plt.savefig(outfig % (np.log10(Mvir), const_c, sigma_m), dpi=300)
plt.close(fig1)
