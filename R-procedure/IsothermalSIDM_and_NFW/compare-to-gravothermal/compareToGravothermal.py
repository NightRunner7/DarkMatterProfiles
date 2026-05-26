"""
Compare the results from gravothermal simulation and Isothermal simulation (R1-procedure).
The one thing has to be made:

    1) accuracy of data coming from Low dense is set to `r_low=-2.0`. So it means
    the smallest radi is: `r_s * 10**-2.0`. For such data, unfortunately there is
    a small deviation when it comes to central velocity dispersion (coming from minimization
    procedure) and core velocity dispersion  (coming from saved data). But when it comes to
    central density and core density there is no such a thing. In generally we shouldn't care
    about it, because this is the problem with data, which we stored, if someone really need
    good accuracy for low time in `Isothermal simulation` just change `r_low=-4.0`. And we
    do not care, because we want to use this data in geothermal evolution (which is implemented
    in mathematica file, which requires `r_low=-2.0`, differently the simulation really get
    slow down).

    2) accuracy of data coming from High dense is set to `r_low=-4.0`. In this case we actually
    do care about accuracy, because the differences appear in the central density (coming from
    minimization procedure) and core density  (coming from saved data). Also, we need such data
    in the mirror method, where in high dense solution we reverse the time - beginning of simulation
    become the collapse.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import RprocedureData as rprocF
import units as uni
from config import DATA_GRAVOTHERMAL_DIR, DATA_ISOTHERMAL_NFW


# ##################################### SETTINGS ##################################### #
plot_save = True

# --- select gravothermal directory
# gravothermal
DATA_GRAVOTHERMAL_RHO_DIR = os.path.join(DATA_GRAVOTHERMAL_DIR, 'data-t-10-Gyr', 'rho_M_5x10^10.0')
DATA_GRAVOTHERMAL_SIGMA_DIR = os.path.join(DATA_GRAVOTHERMAL_DIR, 'data-t-10-Gyr', 'sigma_M_5x10^10.0')
# isothermal
DATA_ISOTHERMAL_NFW_LoDen_DIR = os.path.join(DATA_ISOTHERMAL_NFW, 'LowDenSolution_rlow_-2.0')
DATA_ISOTHERMAL_NFW_HiDen_DIR = os.path.join(DATA_ISOTHERMAL_NFW, 'HighDenSolution_rlow_-4.0')

# --- select files:
gravothermal_rho_file = 'ρsol_M_10.699_t_10._sigma_100..txt'
gravothermal_sigma_file = 'σsol_M_10.699_t_10._sigma_100..txt'
isothermal_loDen_file = 'Riso_M_10.69897_t_7.399_sigma_100.0_con_11.322.csv'
isothermal_HiDen_file = 'RisoHiDens_M_10.69897_t_7.399_sigma_m_100.0_con_11.322.csv'

# ##################################### SET DATA TO CLASS ##################################### #
# ------------------------------- GRAVOTHERMAL ------------------------------- #
gravoEvolution = gravoF.create_gravothermalData_from_file(gravothermal_rho_file, DATA_GRAVOTHERMAL_RHO_DIR,
                                                          beta=0.75,
                                                          one_file=False,
                                                          veldis_file_name=gravothermal_sigma_file,
                                                          localization_of_veldis_file=DATA_GRAVOTHERMAL_SIGMA_DIR)
# put the name to differentiate data in class (what initial profile is)
gravoEvolution.put_the_name("NFW")

# ------------------------------- ISOTHERMAL (low dense) ------------------------------- #
isoEvolution_lowDen = rprocF.create_RprocedureData_from_file(isothermal_loDen_file, DATA_ISOTHERMAL_NFW_LoDen_DIR)
# put the name to differentiate data in class
isoEvolution_lowDen.put_the_name("IsothermalAndNFW")

# ------------------------------- ISOTHERMAL (high dense) ------------------------------- #
isoEvolution_higDen = rprocF.create_RprocedureData_from_file(isothermal_HiDen_file, DATA_ISOTHERMAL_NFW_HiDen_DIR)
# put the name to differentiate data in class
isoEvolution_higDen.put_the_name("IsothermalAndNFW")

# ##################################### PLOTTING: CONTROL AND PRINT ##################################### #
# --- find values of simulation parameters
IsoParameters = isoEvolution_lowDen.return_parameters()
Mvir = IsoParameters["Mvir"]
logMvir = np.log10(IsoParameters["Mvir"])
const_c = IsoParameters["const_c"]
sigma_m = IsoParameters["sigma_m"]
# --- lines from Rprocedure evolution and time
# tmerge_step = isoEvolution.time_steps - 1  # We have to subtract one!!!
tmerge_step = isoEvolution_lowDen.time_steps  # We have to subtract one!!!

close_tmerge_step = tmerge_step - 1
Rres = 0.01  # [kpc]
r_s = IsoParameters["r_s"]  # [kpc]
rho_s = IsoParameters["rho_s"]
r_1 = IsoParameters["r1"]  # [kpc] at tmerge!
tmerge_Rproc = IsoParameters["time-end"]

print("*************** BASIC SETTINGS ***************")
print("time steps in Rprocedure:", isoEvolution_lowDen.time_steps)
print(f"r_1 at tmerge: {r_1} [kpc]")
print(f"r_s: {r_s} [kpc]")
print(f"rho_s: {rho_s} [M_sun/kpc^3]")
print(f"dark matter concentration: {const_c} [dimensionless]")
print(f"Halo mass: {Mvir} [M_sun]")
print(f"tmerge: {isoEvolution_lowDen.data['time'][-1]} [Gyr]")
print(f"annihilation cross section: {sigma_m} [cm^2/g]")

print("*************** Collapse: Gravo ***************")
data_collapse = gravoEvolution.find_collapse(fixed_limit=8*10**5)
print(f"beta = {gravoEvolution.beta}, collapse in time {data_collapse[1]} [Gyr]")
print(f"beta = {gravoEvolution.beta}, collapse in time {uni.time_tilda(data_collapse[1], rho_s, r_s, sigma_m)} "
      f"[dimensionless]")

print("*************** Starting core contraction: Gravo ***************")
min_rho_core, time_min_core, time_step_min_core = gravoEvolution.find_min_rho_core(elements=2)
print("min_rho_core:", min_rho_core)
print("time_min_core:", time_min_core)
print("time_step_min_core:", time_step_min_core)

# ------------------ PLOT: CENTRAL DENSITY (ISO) AND CORE DENSITY (GRAVO) -------------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
plt.title(r'Central density', fontsize=18)
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
gravo_time_l, gravo_rho_l = gravoEvolution.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, label=f'Gravothermal $\\beta$: {gravoEvolution.beta}, core density.')
# The evolution of central density for R-procedure simulation
iso_lowDen_central_time_l, iso_lowDen_central_rho_l = isoEvolution_lowDen.return_central_rho()
ax.plot(iso_lowDen_central_time_l, iso_lowDen_central_rho_l, label='Isothermal (low dense), central density.')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_higDen.return_central_rho()
ax.plot(iso_HigDen_central_time_l, iso_HigDen_central_rho_l, label='Isothermal (high dense), central density.')

# When we should observe forming the core.
plt.axvline(x=time_min_core, color='grey', linestyle='dashdot',
            label='core contraction in gravothermal,' + "\n" +
                  f'occurred in: {"{:.2f}".format(time_min_core)} [Gyr].')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + 'Two-Simulation-core-density' + '.png', dpi=300)
else:
    plt.show()

# ------------------ PLOT: CENTRAL DENSITY (ISO LOW) AND CORE DENSITY (ISO LOW) -------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
# plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
plt.title(r'Central density', fontsize=18)
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
# The evolution of central density for R-procedure simulation
iso_lowDen_central_time_l, iso_lowDen_central_rho_l = isoEvolution_lowDen.return_central_rho()
ax.plot(iso_lowDen_central_time_l, iso_lowDen_central_rho_l, label='Isothermal (low dense), central density.')
iso_lowDen_core_time_l, iso_lowDen_core_rho_l = isoEvolution_lowDen.return_rho_core_evolution(elements=1)
ax.plot(iso_lowDen_core_time_l, iso_lowDen_core_rho_l, "--", label='Isothermal (low dense), core density.')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + 'Isothermal-lowDen-core-density' + '.png', dpi=300)
else:
    plt.show()

# ------------------ PLOT: CENTRAL DENSITY (ISO HIGH) AND CORE DENSITY (ISO HIGH) ------------------------------------ #
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
# plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
plt.title(r'Central density', fontsize=18)
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
# The evolution of central density for R-procedure simulation
iso_higDen_central_time_l, iso_higDen_central_rho_l = isoEvolution_higDen.return_central_rho()
ax.plot(iso_higDen_central_time_l, iso_higDen_central_rho_l, label='Isothermal (high dense), central density.')
iso_higDen_core_time_l, iso_higDen_core_rho_l = isoEvolution_higDen.return_rho_core_evolution(elements=1)
ax.plot(iso_higDen_core_time_l, iso_higDen_core_rho_l, "--", label='Isothermal (high dense), core density.')

# When we should observe forming the core.
plt.axvline(x=time_min_core, color='grey', linestyle='dashdot',
            label='core contraction in gravothermal,' + "\n" +
                  f'occurred in: {"{:.2f}".format(time_min_core)} [Gyr].')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + 'Isothermal-higDen-core-density' + '.png', dpi=300)
else:
    plt.show()

# ------------------ PLOT: CENTRAL VelDis (Rproc) AND CORE VelDis (GRAVO) -------------------------------------------- #
# VelDis: velocity dispersion
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# log scale
ax.set_xscale('log')
# ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 10 ** 1)
plt.ylabel(r'$\nu_{0} \ \left[r_{s} \ Gyr^{-1} \right]$', fontsize=18)
plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
plt.title(r'Central velocity dispersion', fontsize=18)
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
# locmaj = mticker.LogLocator(base=10, numticks=12)
# locmin = mticker.LogLocator(base=10.0,
#                             subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
#                             numticks=12)
# ax.yaxis.set_major_locator(locmaj)
# ax.yaxis.set_minor_locator(locmin)
# ax.yaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('both', direction='in', top='on', right='on', length=10,
               width=1, which='major', zorder=301)
ax.tick_params('both', direction='in', top='on', right='on', length=5,
               width=1, which='minor', zorder=301)

# --- DATA
# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_veldis_l = gravoEvolution.return_veldis_core_evolution(elements=1)
ax.plot(gravo_time_l, gravo_veldis_l, label=f'Gravothermal $\\beta$: {gravoEvolution.beta}, core veldis.')
# The evolution of central density for R-procedure simulation
iso_lowDen_central_time_l, iso_lowDen_central_veldis_l = isoEvolution_lowDen.return_central_veldis()
ax.plot(iso_lowDen_central_time_l, iso_lowDen_central_veldis_l, label='Isothermal (low dense), central veldis.')
iso_HigDen_central_time_l, iso_HigDen_central_veldis_l = isoEvolution_higDen.return_central_veldis()
ax.plot(iso_HigDen_central_time_l, iso_HigDen_central_veldis_l, label='Isothermal (high dense), central veldis.')


# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + 'Two-Simulation-core-velocity-dispersion' + '.png', dpi=300)
else:
    plt.show()

# ------------------ PLOT: CENTRAL VelDis (Rproc) AND CORE VelDis (Rproc) -------------------------------------------- #
# VelDis: velocity dispersion
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# log scale
ax.set_xscale('log')
# ax.set_yscale('log')
# describe plot
plt.ylim(3 * 10 ** 0, 8 * 10 ** 0)
plt.ylabel(r'$\nu_{0} \ \left[r_{s} \ Gyr^{-1} \right]$', fontsize=18)
plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
plt.title(r'Central velocity dispersion', fontsize=18)
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
# locmaj = mticker.LogLocator(base=10, numticks=12)
# locmin = mticker.LogLocator(base=10.0,
#                             subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
#                             numticks=12)
# ax.yaxis.set_major_locator(locmaj)
# ax.yaxis.set_minor_locator(locmin)
# ax.yaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('both', direction='in', top='on', right='on', length=10,
               width=1, which='major', zorder=301)
ax.tick_params('both', direction='in', top='on', right='on', length=5,
               width=1, which='minor', zorder=301)

# --- DATA
# The evolution of central velocity dispersion for R-procedure simulation
iso_lowDen_central_time_l, iso_lowDen_central_veldis_l = isoEvolution_lowDen.return_central_veldis()
ax.plot(iso_lowDen_central_time_l, iso_lowDen_central_veldis_l, label='Isothermal (low dense), central veldis.')
iso_HigDen_central_time_l, iso_HigDen_central_veldis_l = isoEvolution_higDen.return_central_veldis()
ax.plot(iso_HigDen_central_time_l, iso_HigDen_central_veldis_l, label='Isothermal (high dense), central veldis.')
# The evolution of core velocity dispersion for R-procedure simulation
iso_lowDen_core_time_l, iso_lowDen_core_veldis_l = isoEvolution_lowDen.return_veldis_core_evolution(elements=1)
ax.plot(iso_lowDen_core_time_l, iso_lowDen_core_veldis_l, '--', label='Isothermal (low dense), core veldis.')
iso_HigDen_core_time_l, iso_HigDen_core_veldis_l = isoEvolution_higDen.return_veldis_core_evolution(elements=1)
ax.plot(iso_HigDen_core_time_l, iso_HigDen_core_veldis_l, ':', label='Isothermal (high dense), core veldis.')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + 'Isothermal-core-velocity-dispersion' + '.png', dpi=300)
else:
    plt.show()


# ------------------ PLOT: DENSITY AT TMERGE (Rproc) AND CORE DENSITY (GRAVO) ---------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
# plt.ylim(10 ** (0), 2 * 10 ** (3))
plt.ylabel(r'$\rho(r) \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$r \ \left[r_{s}\right]$', fontsize=18)
plt.title(f'Density at tmerge: {"{:.2f}".format(tmerge_Rproc)} [Gyr]', fontsize=18)
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
# Density at tmerge: gravothermal
Rproc_tmerge = gravoEvolution.return_data_at_fixed_time(tmerge_Rproc, time_step_bool=False)
ax.plot(Rproc_tmerge['r'], Rproc_tmerge['rho'], label=f'Gravothermal $\\beta$: {gravoEvolution.beta}')
# Density at tmerge: R-procedure: low dense case
Rproc_LoDen_data = isoEvolution_lowDen.return_data_at_fixed_time(close_tmerge_step)
ax.plot(Rproc_LoDen_data['r'], Rproc_LoDen_data['rho'], label='Isothermal (low dense)')
Rproc_HiDen_data = isoEvolution_higDen.return_data_at_fixed_time(close_tmerge_step)
ax.plot(Rproc_LoDen_data['r'], Rproc_LoDen_data['rho'], "--", label='Isothermal (high dense)')

# lines
plt.axhline(y=iso_lowDen_central_rho_l[close_tmerge_step], color='black', linestyle='--',
            label=f'central density (low dense): '
                  f'{"{:.2E}".format(iso_lowDen_central_rho_l[close_tmerge_step])} [rho_s].')
plt.axhline(y=iso_HigDen_central_rho_l[close_tmerge_step], color='grey', linestyle=':',
            label=f'central density (high dense): '
                  f'{"{:.2E}".format(iso_HigDen_central_rho_l[close_tmerge_step])} [rho_s].')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + 'Two-Simulation-density-at-tmerge' + '.png', dpi=300)
else:
    plt.show()

# ------------------ PLOT: CENTRAL VelDis (Rproc) AND CORE VelDis (GRAVO) ------------------ #
# VelDis: velocity dispersion
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# log scale
ax.set_xscale('log')
# ax.set_yscale('log')
# describe plot
# plt.ylim(10 ** 0, 10 ** 1)
plt.ylabel(r'$\nu_{0} \ \left[r_{s} \ Gyr^{-1} \right]$', fontsize=18)
plt.xlabel(r'$r \ \left[r_{s}\right]$', fontsize=18)
plt.title(f'Velocity dispersion at tmerge: {"{:.2f}".format(tmerge_Rproc)} [Gyr]', fontsize=18)
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
# locmaj = mticker.LogLocator(base=10, numticks=12)
# locmin = mticker.LogLocator(base=10.0,
#                             subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
#                             numticks=12)
# ax.yaxis.set_major_locator(locmaj)
# ax.yaxis.set_minor_locator(locmin)
# ax.yaxis.set_minor_formatter(mticker.NullFormatter())
# tick length
ax.tick_params('both', direction='in', top='on', right='on', length=10,
               width=1, which='major', zorder=301)
ax.tick_params('both', direction='in', top='on', right='on', length=5,
               width=1, which='minor', zorder=301)

# --- DATA
# Velocity dispersion at tmerge: gravothermal
ax.plot(Rproc_tmerge['r'], Rproc_tmerge['velDis'], label=f'Gravothermal $\\beta$: {gravoEvolution.beta}')
# Velocity dispersion at tmerge: R-procedure: low dense case
ax.plot(Rproc_LoDen_data['r'], Rproc_LoDen_data['velDis'], label='Isothermal (low dense)')
ax.plot(Rproc_HiDen_data['r'], Rproc_HiDen_data['velDis'], "--", label='Isothermal (high dense)')

# lines
plt.axhline(y=iso_lowDen_central_veldis_l[close_tmerge_step], color='black', linestyle='--',
            label=f'central velDis (low dense): '
                  f'{"{:.2E}".format(iso_lowDen_central_veldis_l[close_tmerge_step])} [r_s / Gyr].')
plt.axhline(y=iso_HigDen_central_veldis_l[close_tmerge_step], color='grey', linestyle=':',
            label=f'central velDis (high dense): '
                  f'{"{:.2E}".format(iso_HigDen_central_veldis_l[close_tmerge_step])} [r_s / Gyr].')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + 'Two-Simulation-velocity-dispersion-at-tmerge' + '.png', dpi=300)
else:
    plt.show()
