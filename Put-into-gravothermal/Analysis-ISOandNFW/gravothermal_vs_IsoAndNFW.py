"""
Compare the results from gravothermal simulation and Rprocedure + gravothermal simulation
(with Isothermal initial profile). The most interesting part is verify that those two
simulation have the same prediction on the core-collapse regime.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
from NFWProfile import NFWProfile, r1  # CDM profile (halo)
import GravothermalData as gravoF
import RprocedureData as rprocF
import auxiliaryFunctions as aux
# --- IMPORT TO DRAW PLOTS
from PlotVelocityDispersion import plot_series_nu_regime_I, plot_series_nu_regime_II, plot_series_nu_regime_II_ISO, \
    plot_series_nu_regime_II_NFW
from PlotDensityProfile import plot_series_rho_regime_I, plot_series_rho_regime_II, plot_series_rho_regime_II_ISO, \
    plot_series_rho_regime_II_NFW

# --- HELPFUL FUNCTIONS
def time_change_units(my_list):
    """in .txt file the units of time is [log10(Gyr)]"""
    return [10 ** x for x in my_list]


# ##################################### SETTINGS ##################################### #
plot_save = True
beta = 1.0  # the value of parameter in the gravothermal simulation: BE WARNING!

# gravothermal
gravo_file = "./Input/Gravothermal/ρσ_sol_M_10.699_t_2200.42_sigma_1.0_beta_0.5.csv"
# gravothermal + Rprocedure
# gravoAndRproc_file = "./Input/RprocedureAndGravothermal/Riso_sol_M_10.699_t_2200.43_sigma_1.0_beta_0.5.csv"
gravoAndRproc_file = "./Input/RprocedureAndGravothermal/trueRiso_sol_M_10.699_t_2051.1_sigma_1.0_beta_1.0.csv"
# after hydrodynamic equilibrium
after_hydro_file = "./Input/hydrodynamicEqulibrium/HydroEqu_M_10.699_sigma_1.0_beta_1.0.csv"
# Rprocedure
Riso_file = "./Input/Rprocedure/Riso_M_10.69897_t_863.105_sigma_1.0_con_11.322.csv"

# ##################################### IMPORT DATA ##################################### #
# ------------------------------- GRAVOTHERMAL ------------------------------- #
#  create map, which contains all data coming from file created by mathematica code
gravo_data = pd.read_csv(gravo_file, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravo_data = gravo_data.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
pureNFW_class = gravoF.GravothermalData(gravo_data['t'], gravo_data['r'], gravo_data['rho'],
                                        gravo_data['velDis'], beta=0.5)

# put the name to differentiate data in class
pureNFW_class.put_the_name("NFW")

# ------------------------------- GRAVOTHERMAL AND R-PROCEDURE ------------------------------- #
# --- main data
#  create map, which contains all data coming from file created by mathematica code: rho
gravoRpro_data = pd.read_csv(gravoAndRproc_file, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravoRpro_data = gravoRpro_data.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
isoAndNFW_gravo_class = gravoF.GravothermalData(gravoRpro_data['t'], gravoRpro_data['r'], gravoRpro_data['rho'],
                                                gravoRpro_data['velDis'], beta=1.0)

# put the name to differentiate data in class
isoAndNFW_gravo_class.put_the_name("Isothermal")

# --- after hydrodynamic equilibrium (first step)
hydro_data = pd.read_csv(after_hydro_file, sep='\t', names=['r', 'rho', 'velDis'])

# ------------------------------- RPROCEDURE: ISOTHERMAL ------------------------------- #
#  create map, which contains all data coming from file created by mathematica code: rho
isoAndNFW_data = pd.read_csv(Riso_file, sep='\t', names=['t', 'r', 'rho', 'mass', 'velDis',
                                                         'central-rho', 'central-velDis',
                                                         'names', 'values'])

# create class, which will contain all the necessary data
isoAndNFW_rproc_class = rprocF.RprocedureData(isoAndNFW_data['t'], isoAndNFW_data['r'], isoAndNFW_data['rho'],
                                              isoAndNFW_data['velDis'], isoAndNFW_data['names'], isoAndNFW_data['values'])

# put the name to differentiate data in class
isoAndNFW_rproc_class.put_the_name("Isothermal")

# ##################################### PLOTTING: SETTINGS ##################################### #
# --- find values of simulation parameters
Mvir, sigma_m, const_c = isoAndNFW_rproc_class.return_basic_parameters()
logMvir = np.log10(Mvir)

isoAndNFW_gravo_class.put_extra_parameters(Mvir, sigma_m)  # put those parameters into class
pureNFW_class.put_extra_parameters(Mvir, sigma_m)  # put those parameters into class

# --- deal with directories
format_log10mass = f'{"{:.5f}".format(logMvir)}'
output_directory = f'Low-Plots-M-{format_log10mass}-sigma-{sigma_m}-beta-{beta}'
aux.make_directory(output_directory)  # make directory

# --- finding moment of starting contracting the core
importantPoints = dict()
# gravothermal: pure NFW
importantPoints['NFW-core'] = dict()
importantPoints['NFW-core']['rho'], \
importantPoints['NFW-core']['time'], \
importantPoints['NFW-core']['time-step'] = pureNFW_class.find_min_rho_core()
# gravothermal: Isothermal
importantPoints['isoAndNFW-core'] = dict()
importantPoints['isoAndNFW-core']['rho'], \
importantPoints['isoAndNFW-core']['time'], \
importantPoints['isoAndNFW-core']['time-step'] = isoAndNFW_rproc_class.find_min_rho_core()

# print("core_rho_min_core", importantPoints['coreNFW-core']['rho'],
#       "coreNFW_time_min_core:", importantPoints['coreNFW-core']['time'])

# --- lines from Rprocedure evolution and time
tmerge_step = isoAndNFW_rproc_class.time_steps - 1  # We have to subtract one!!!
close_tmerge_step = tmerge_step - 1
Rres = 0.01  # [kpc]
r_s = isoAndNFW_rproc_class.parameters["r_s"]  # [kpc]
r_1 = isoAndNFW_rproc_class.parameters["r1"]  # [kpc] at tmerge!

print("time steps in Rprocedure:", isoAndNFW_rproc_class.time_steps)

# --- line from gravothermal evolution and time
tend_step = isoAndNFW_gravo_class.time_steps - 1  # We have to subtract one!!!

# ##################################### PLOTTING: PLOTS ##################################### #
# ------------------ PLOT: CENTRAL DENSITY (Rproc) AND CORE DENSITY (GRAVO) ------------------ #
fig, ax = plt.subplots(figsize=(14.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
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
gravo_time_l, gravo_rho_l = pureNFW_class.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, label='gravothermal (NFW: initial).')

# The evolution of central density for `Isothermal and NFW`
isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l = isoAndNFW_gravo_class.return_rho_core_evolution(elements=2)
ax.plot(isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l, label='gravothermal (Isothermal: initial)')

# The evolution of central density for gravothermal and R-procedure
isoAndNFW_rproc_time_l, isoAndNFW_rproc_rho_l = isoAndNFW_rproc_class.return_rho_core_evolution(elements=2)
ax.plot(isoAndNFW_rproc_time_l, isoAndNFW_rproc_rho_l[0:len(isoAndNFW_rproc_time_l)], label='R-procedure (Isothermal: initial).')

# When we should observe forming the core.
plt.axvline(x=importantPoints['NFW-core']['time'], color='grey', linestyle='dashdot',
            label='Start contracting (gravothermal),' + "\n" +
                  f'in: {"{:.2f}".format(importantPoints["NFW-core"]["time"])} [Gyr].')

plt.axvline(x=importantPoints['isoAndNFW-core']['time'], color='black', linestyle='dashdot',
            label='Start contracting (Isothermal),' + "\n" +
                  f'in: {"{:.2f}".format(importantPoints["isoAndNFW-core"]["time"])} [Gyr].')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/Gravothermal-and-Rprocedure-core-density' + '.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# -------------------------- PLOT: DENSITY PROFILE AT TMERGE -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$r \ \left[r_{s}\right]$', fontsize=18)
plt.title(f'Density profile: Isothermal (R-procedure)', fontsize=18)
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
# profile at tmerge
tmerge_profile = isoAndNFW_rproc_class.return_data_at_fixed_time(tmerge_step)
ax.plot(tmerge_profile['r'], tmerge_profile['rho'], "--",
        label=f'tmerge, at {"{:.2f}".format(tmerge_profile["time"])} [Gyr]')
# profile before the tmerge
before_tmerge_profile = isoAndNFW_rproc_class.return_data_at_fixed_time(close_tmerge_step)
ax.plot(before_tmerge_profile['r'], before_tmerge_profile['rho'], ":",
        label=f'before tmerge,at {"{:.2f}".format(before_tmerge_profile["time"])} [Gyr]')
# profile after the tmerge
after_tmerge_profile = isoAndNFW_gravo_class.return_data_at_fixed_time(tmerge_profile["time"], time_step_bool=False)
ax.plot(after_tmerge_profile['r'], after_tmerge_profile['rho'],
        label=f'after tmerge, at {"{:.2f}".format(after_tmerge_profile["time"])} [Gyr]')
# lines
plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot')
plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted')
plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9)

# plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot',
#             label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
# plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted',
#             label=f'r_1 = {"{:.3f}".format(r_1)} [kpc].')
# plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9,
#             label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
# annotation to lines
ax.text(r_s / r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(r_1 / r_s, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres / r_s, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/Density-profile-at-tmerge' + '.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# -------------------------- PLOT: VELOCITY DISPERSION PROFILE AT TMERGE -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
# describe NFW profile
# plt.ylim(10 ** 1, 7 * 10 ** 1)
plt.ylabel(r"$\nu$ [$r_{s}$ / Gyr]", fontsize=18)
plt.xlabel(r"$r$ [$r_{s}$]", fontsize=18)
plt.title("Dispersion velocity: Isothermal (R-procedure)", fontsize=18)
ax.tick_params(labelsize=14)

# --- DATA
# profile at tmerge
ax.plot(tmerge_profile['r'], tmerge_profile['velDis'], "--",
        label=f'tmerge, at {"{:.2f}".format(tmerge_profile["time"])} [Gyr]')
# profile before the tmerge
ax.plot(before_tmerge_profile['r'], before_tmerge_profile['velDis'], ":",
        label=f'before tmerge,at {"{:.2f}".format(before_tmerge_profile["time"])} [Gyr]')
# profile after the tmerge
ax.plot(after_tmerge_profile['r'], after_tmerge_profile['velDis'],
        label=f'after tmerge, at {"{:.2f}".format(after_tmerge_profile["time"])} [Gyr]')

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

# lines
plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot')
plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted')
plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9)

# plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot',
#             label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
# plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted',
#             label=f'r_1 = {"{:.3f}".format(r_1)} [kpc].')
# plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9,
#             label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
# annotation to lines
ax.text(r_s / r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(r_1 / r_s, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres / r_s, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)
# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/Velocity-dispersion-profile-at-tmerge' + '.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# ##################################### PLOTTING: HYDRODYNAMIC ##################################### #
# compare the density and velocity dispersion before and after hydrodynamic step
# -------------------------- PLOT: DENSITY PROFILE AT TMERGE -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$r \ \left[r_{s}\right]$', fontsize=18)
plt.title(f'Density profile: Isothermal (R-procedure)', fontsize=18)
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
# profile before hydrodynamic equilibrium
tmerge_profile = isoAndNFW_rproc_class.return_data_at_fixed_time(tmerge_step)
ax.plot(tmerge_profile['r'], tmerge_profile['rho'], "--",
        label=f'before hydrodynamic equl')
# profile after hydrodynamic equilibrium
ax.plot(hydro_data['r'], hydro_data['rho'],
        label=f'after hydrodynamic equl')

# lines
plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot')
plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted')
plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9)

# annotation to lines
ax.text(r_s / r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(r_1 / r_s, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres / r_s, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/Hydro-rho' + '.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# -------------------------- PLOT: VELOCITY DISPERSION -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
# describe NFW profile
# plt.ylim(10 ** 1, 7 * 10 ** 1)
plt.ylabel(r"$\nu$ [$r_{s}$ / Gyr]", fontsize=18)
plt.xlabel(r"$r$ [$r_{s}$]", fontsize=18)
plt.title("Dispersion velocity and hydrodynamic equilibrium", fontsize=18)
ax.tick_params(labelsize=14)

# --- DATA
# profile before hydrodynamic equilibrium
before_hyrdo = isoAndNFW_rproc_class.return_data_at_fixed_time(tmerge_step)
ax.plot(tmerge_profile['r'], tmerge_profile['velDis'], "--",
        label=f'before hydrodynamic equl')
# profile after hydrodynamic equilibrium
ax.plot(hydro_data['r'], hydro_data['velDis'],
        label=f'after hydrodynamic equl')

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

# lines
plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot')
plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted')
plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9)

# annotation to lines
ax.text(r_s / r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(r_1 / r_s, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
        ha='right', va='top', transform=ax.transData, rotation=90)
ax.text(Rres / r_s, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
        ha='left', va='top', transform=ax.transData, rotation=90)
# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/Hydro-Nu' + '.png', dpi=300)
    plt.close(fig)
else:
    plt.show()


# ##################################### PLOTTING: AT FIXED TIME ##################################### #
# Here a dozens of plots which describes the velocity dispersion and density
# So at fixed time we will compare how look profile for `pure NFW` gravothermal
# simulation and `Isothermal` - in one evolution region we will have data coming
# from R-procedure and from gravothermal simulation.

# --- finding proper r_1
NFW_profile_class = NFWProfile(Mvir, const_c)  # NFW profile
r_1 = r1(NFW_profile_class, sigmamx=sigma_m, tage=0.01)  # r1 [kpc]

# --- we consider data in regime: core formation (when Rprocedure is valid)
start = 0
stop = tmerge_step
num = int((start + tmerge_step) / 5)
rproc_step_plot = np.linspace(start, stop, num=num, endpoint=True, retstep=False, dtype=int)

# --- we consider data in regime: core collapse (when Rprocedure fails)
start = 0
stop = tend_step
num = 40
gravo_step_plot = np.linspace(start, stop, num=num, endpoint=True, retstep=False, dtype=int)

# --- we consider data in regime: core collapse (when Rprocedure fails)
start = 0
num = 40
stop = start + num - 1
gravo_step_plot_begEvo = np.linspace(start, stop, num=num, endpoint=True, retstep=False, dtype=int)


# print("gravo_step_plot:", gravo_step_plot)
# print("coreNFW_gravo_class.number_radi:", coreNFW_gravo_class.number_radi)

# -------------------------- PLOT: DENSITY PROFILE AT FIXED TIME -------------------------- #
# --- setting
xmin = 7 * 10 ** (-4)
xmax = 2 * 10 ** 2
ymin = 5 * 10 ** (-7)
ymax = 3 * 10 ** 2

# --- Regime I
plot_series_rho_regime_I(isoAndNFW_rproc_class, pureNFW_class, rproc_step_plot,
                         path_dir=output_directory + '/Density-regime-I',
                         xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# --- Regime II
plot_series_rho_regime_II(isoAndNFW_gravo_class, pureNFW_class, gravo_step_plot,
                          path_dir=output_directory + '/Density-regime-II',
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# -------------------------- PLOT: VELOCITY DISPERSION AT FIXED TIME -------------------------- #
# --- setting
xmin = 7 * 10 ** (-4)
xmax = 2 * 10 ** 2
ymin = 1.8
ymax = 7.9

# --- Regime I
plot_series_nu_regime_I(isoAndNFW_rproc_class, pureNFW_class, rproc_step_plot,
                        path_dir=output_directory + '/Nu-regime-I',
                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# --- Regime II: NFW and Isothermal
plot_series_nu_regime_II(isoAndNFW_gravo_class, pureNFW_class, gravo_step_plot,
                         path_dir=output_directory + '/Nu-regime-II',
                         xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# --- Regime II: just Isothermal
plot_series_nu_regime_II_ISO(isoAndNFW_gravo_class, gravo_step_plot,
                             path_dir=output_directory + '/Nu-regime-II-ISO',
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# --- Regime II: on the beginning (Isothermal)
plot_series_nu_regime_II_ISO(isoAndNFW_gravo_class, gravo_step_plot_begEvo,
                             path_dir=output_directory + '/Nu-regime-II-ISO-beg',
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# --- Regime II: on the beginning (Gravothermal)
plot_series_nu_regime_II_NFW(isoAndNFW_gravo_class, pureNFW_class, gravo_step_plot_begEvo,
                             path_dir=output_directory + '/Nu-regime-II-NFW-beg',
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
