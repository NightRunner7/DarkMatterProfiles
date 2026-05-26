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
import GravothermalData as gravoF
import RprocedureData as rprocF
import units as uni

# ##################################### SETTINGS ##################################### #
plot_save = True
output_directory = ""

# gravothermal
gravo_file_1 = "./Input/Gravothermal/ρσ_sol_M_10.699_t_2200.42_sigma_1.0_beta_0.5.csv"
gravo_file_2 = "./Input/Gravothermal/ρσ_sol_M_10.699_t_2200.42_sigma_1.0_beta_0.6.csv"
gravo_file_3 = "./Input/Gravothermal/ρσ_sol_M_10.699_t_2200.43_sigma_1.0_beta_0.75.csv"
gravo_file_4 = "./Input/Gravothermal/ρσ_sol_M_10.699_t_2051.1_sigma_1.0_beta_0.3385.csv"


# gravothermal + Rprocedure
gravoAndRproc_file_0 = "./Input/RprocedureAndGravothermal/Riso_minDen_sol_M_10.699_t_2200.43_sigma_1.0_beta_0.5.csv"
gravoAndRproc_file_1 = "./Input/RprocedureAndGravothermal/Riso_vol1_sol_M_10.699_t_2200.43_sigma_1.0_beta_0.5.csv"
gravoAndRproc_file_2 = "./Input/RprocedureAndGravothermal/Riso_vol2_sol_M_10.699_t_2138.01_sigma_1.0_beta_0.5.csv"
gravoAndRproc_file_3 = "./Input/RprocedureAndGravothermal/trueRiso_sol_M_10.699_t_2051.1_sigma_1.0_beta_1.0.csv"

select_file = gravoAndRproc_file_3  # gravoAndRproc_file_0, gravoAndRproc_file_1, gravoAndRproc_file_2

# Rprocedure
# Riso_file = "./Input/Rprocedure/Riso_M_10.69897_t_855.204_sigma_1.0_con_11.322.csv"
# Riso_file = "./Input/Rprocedure/Riso_M_10.69897_t_880.172_sigma_1.0_con_11.322.csv"
Riso_file = "./Input/Rprocedure/Riso_M_10.69897_t_863.105_sigma_1.0_con_11.322.csv"

# Rprocedure High dense (to mirror method)
# Riso_HiDens_file = "./Input/Rprocedure/RisoHiDens_M_10.69897_t_855.204_sigma_1.0_con_11.322.csv"
Riso_HiDens_file = "./Input/Rprocedure/RisoHiDens_M_10.69897_t_863.105_sigma_1.0_con_11.322.csv"

# ##################################### IMPORT DATA ##################################### #
# ------------------------------- GRAVOTHERMAL ------------------------------- #
# --- FILE 1
#  create map, which contains all data coming from file created by mathematica code
gravo_data_1 = pd.read_csv(gravo_file_1, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravo_data_1 = gravo_data_1.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
NFW_class_1 = gravoF.GravothermalData(gravo_data_1['t'], gravo_data_1['r'], gravo_data_1['rho'],
                                      gravo_data_1['velDis'], beta=0.5)

# put the name to differentiate data in class
NFW_class_1.put_the_name("NFW")

# --- FILE 2
#  create map, which contains all data coming from file created by mathematica code
gravo_data_2 = pd.read_csv(gravo_file_2, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravo_data_2 = gravo_data_2.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
NFW_class_2 = gravoF.GravothermalData(gravo_data_2['t'], gravo_data_2['r'], gravo_data_2['rho'],
                                      gravo_data_2['velDis'], beta=0.6)

# put the name to differentiate data in class
NFW_class_2.put_the_name("NFW")

# --- FILE 3
#  create map, which contains all data coming from file created by mathematica code
gravo_data_3 = pd.read_csv(gravo_file_3, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravo_data_3 = gravo_data_3.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
NFW_class_3 = gravoF.GravothermalData(gravo_data_3['t'], gravo_data_3['r'], gravo_data_3['rho'],
                                      gravo_data_3['velDis'], beta=0.75)

# put the name to differentiate data in class
NFW_class_3.put_the_name("NFW")

# --- FILE 4
#  create map, which contains all data coming from file created by mathematica code
gravo_data_4 = pd.read_csv(gravo_file_4, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravo_data_4 = gravo_data_4.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
NFW_class_4 = gravoF.GravothermalData(gravo_data_4['t'], gravo_data_4['r'], gravo_data_4['rho'],
                                      gravo_data_4['velDis'], beta=0.3385)

# put the name to differentiate data in class
NFW_class_4.put_the_name("NFW")

# ------------------------------- GRAVOTHERMAL AND R-PROCEDURE ------------------------------- #
# --- FILE 1
#  create map, which contains all data coming from file created by mathematica code: rho
gravoRpro_data_1 = pd.read_csv(select_file, sep='\t', names=['t', 'r', 'rho', 'velDis'])
gravoRpro_data_1 = gravoRpro_data_1.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

# create class, which will contain all the necessary data
ISO_class_1 = gravoF.GravothermalData(gravoRpro_data_1['t'], gravoRpro_data_1['r'], gravoRpro_data_1['rho'],
                                      gravoRpro_data_1['velDis'], beta=0.5)

# put the name to differentiate data in class
ISO_class_1.put_the_name("Isothermal")

# ------------------------------- RPROCEDURE: ISO + NFW ------------------------------- #
#  create map, which contains all data coming from file created by mathematica code: rho
isoAndNFW_data = pd.read_csv(Riso_file, sep='\t', names=['t', 'r', 'rho', 'mass', 'velDis',
                                                         'central-rho', 'central-velDis',
                                                         'names', 'values'])

# create class, which will contain all the necessary data
RISO_class = rprocF.RprocedureData(isoAndNFW_data['t'], isoAndNFW_data['r'], isoAndNFW_data['rho'],
                                   isoAndNFW_data['velDis'], isoAndNFW_data['names'], isoAndNFW_data['values'])

# put the name to differentiate data in class
RISO_class.put_the_name("Isothermal")

# ------------------------------- RPROCEDURE: ISO + NFW (High Dense) ------------------------------- #
#  create map, which contains all data coming from file created by mathematica code: rho
Mirror_data = pd.read_csv(Riso_HiDens_file, sep='\t', names=['t', 'r', 'rho', 'mass', 'velDis',
                                                             'central-rho', 'central-velDis',
                                                             'names', 'values'])

# --- FIRSTS CLASS: PROPORTION PARAMETER
# create class, which will contain all the necessary data
Mirror_class_1 = rprocF.RprocedureDataMirror(Mirror_data['t'], Mirror_data['r'], Mirror_data['rho'],
                                             Mirror_data['velDis'], Mirror_data['names'], Mirror_data['values'],
                                             proportion=2.0)

Mirror_class_1.put_the_name("Isothermal-High-dense")  # put the name to differentiate data in class

# --- SECOND CLASS: PROPORTION PARAMETER
# create class, which will contain all the necessary data
Mirror_class_2 = rprocF.RprocedureDataMirror(Mirror_data['t'], Mirror_data['r'], Mirror_data['rho'],
                                             Mirror_data['velDis'], Mirror_data['names'], Mirror_data['values'],
                                             proportion=1.32)

Mirror_class_2.put_the_name("Isothermal-High-dense")  # put the name to differentiate data in class


# --- THIRD CLASS: PROPORTION PARAMETER
# create class, which will contain all the necessary data
Mirror_class_3 = rprocF.RprocedureDataMirror(Mirror_data['t'], Mirror_data['r'], Mirror_data['rho'],
                                             Mirror_data['velDis'], Mirror_data['names'], Mirror_data['values'],
                                             proportion=1.7)

Mirror_class_3.put_the_name("Isothermal-High-dense")  # put the name to differentiate data in class


# --- FOURTH CLASS: PROPORTION PARAMETER
# create class, which will contain all the necessary data
Mirror_class_4 = rprocF.RprocedureDataMirror(Mirror_data['t'], Mirror_data['r'], Mirror_data['rho'],
                                             Mirror_data['velDis'], Mirror_data['names'], Mirror_data['values'],
                                             proportion=1.5)

Mirror_class_4.put_the_name("Isothermal-High-dense")  # put the name to differentiate data in class

# --- FIFTH CLASS: PROPORTION PARAMETER
# create class, which will contain all the necessary data
Mirror_class_5 = rprocF.RprocedureDataMirror(Mirror_data['t'], Mirror_data['r'], Mirror_data['rho'],
                                             Mirror_data['velDis'], Mirror_data['names'], Mirror_data['values'],
                                             proportion=2.4)

Mirror_class_5.put_the_name("Isothermal-High-dense")  # put the name to differentiate data in class



# ##################################### PLOTTING: SETTINGS ##################################### #
# --- find values of simulation parameters
Mvir, sigma_m, const_c = RISO_class.return_basic_parameters()
logMvir = np.log10(Mvir)

# --- lines from Rprocedure evolution and time
tmerge_step = RISO_class.time_steps - 1  # We have to subtract one!!!
close_tmerge_step = tmerge_step - 1
Rres = 0.01  # [kpc]
r_s = RISO_class.parameters["r_s"]  # [kpc]
rho_s = RISO_class.parameters["rho_s"]
r_1 = RISO_class.parameters["r1"]  # [kpc] at tmerge!

print("*************** BASIC SETTINGS ***************")
print("time steps in Rprocedure:", RISO_class.time_steps)
print(f"r_1 at tmerge: {r_1} [kpc]")
print(f"r_s: {r_s} [kpc]")
print(f"rho_s: {rho_s} [M_sun/kpc^3]")
print(f"dark matter concentration: {const_c} [dimensionless]")
print(f"Halo mass: {Mvir} [M_sun]")
print(f"tmerge: {RISO_class.data['time'][-1]} [Gyr]")

print(f"annihilation cross section: {sigma_m} [cm^2/g]")

# ##################################### COLLAPSE ##################################### #
print("")
print("*************** Colapse: Gravo and Rproc ***************")
data_collapse = ISO_class_1.find_collapse(fixed_limit=10**1)
print(f"Collapse in time {data_collapse[1]} [Gyr]")
print(f"Collapse in time {uni.time_tilda(data_collapse[1], rho_s, r_s, sigma_m)} [dimensionless]")

print("*************** Colapse: Gravo ***************")
data_collapse = NFW_class_1.find_collapse(fixed_limit=10**23)
print(f"beta = {0.5}, collapse in time {data_collapse[1]} [Gyr]")
print(f"beta = {0.5}, collapse in time {uni.time_tilda(data_collapse[1], rho_s, r_s, sigma_m)} [dimensionless]")

data_collapse = NFW_class_2.find_collapse(fixed_limit=10**23)
print(f"beta = {0.6}, collapse in time {data_collapse[1]} [Gyr]")
print(f"beta = {0.5}, collapse in time {uni.time_tilda(data_collapse[1], rho_s, r_s, sigma_m)} [dimensionless]")

data_collapse = NFW_class_3.find_collapse(fixed_limit=10**23)
print(f"beta = {0.75}, collapse in time {data_collapse[1]} [Gyr]")
print(f"beta = {0.5}, collapse in time {uni.time_tilda(data_collapse[1], rho_s, r_s, sigma_m)} [dimensionless]")

data_collapse = NFW_class_4.find_collapse(fixed_limit=10**23)
print(f"beta = {0.3385}, collapse in time {data_collapse[1]} [Gyr]")
print(f"beta = {0.5}, collapse in time {uni.time_tilda(data_collapse[1], rho_s, r_s, sigma_m)} [dimensionless]")

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
        label=f'gravothermal (NFW: initial), $\\beta$: {0.5}')

# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_2.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (NFW: initial), $\\beta$: {0.6}.')

# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_3.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (NFW: initial), $\\beta$: {0.75}')

# The evolution of core density for gravothermal simulation
gravo_time_l, gravo_rho_l = NFW_class_4.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (NFW: initial), $\\beta$: {0.34}')

# --- ISO

# The evolution of central density for `Isothermal and NFW`
# isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l = ISO_class_1.return_rho_core_evolution(elements=2)
# ax.plot(uni.cal_time_tilda(isoAndNFW_gravo_time_l, rho_s, r_s, sigma_m), isoAndNFW_gravo_rho_l, "--", color='black',
#         label=f'gravothermal (initial: Isothermal at merge), $\\beta$: {0.5}')

# The evolution of central density for `Isothermal and NFW`
isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l = ISO_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(isoAndNFW_gravo_time_l, rho_s, r_s, sigma_m), isoAndNFW_gravo_rho_l, ":", color='grey',
        label=f'gravothermal (initial: Isothermal at lowest density), $\\beta$: {1}')

# The evolution of central density for `Isothermal and NFW`
isoAndNFW_gravo_time_l, isoAndNFW_gravo_rho_l = RISO_class.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(isoAndNFW_gravo_time_l, rho_s, r_s, sigma_m), isoAndNFW_gravo_rho_l[:len(isoAndNFW_gravo_time_l)],
        label=f'Rprocedure (Isothermal)')


# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    # plt.savefig('./' + output_directory + '/CheckEvolution-min-density' + '.png', dpi=300)
    # plt.savefig('./' + output_directory + '/CheckEvolution-merge' + '.png', dpi=300)
    plt.savefig('./' + output_directory + '/CheckEvolution-true' + '.png', dpi=300)

    plt.close(fig)
else:
    plt.show()



# ------------------ PLOT: CENTRAL DENSITY MIRROR METHOD: PROP: 2.0 ------------------ #
fig, ax = plt.subplots(figsize=(14.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
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
# --- GRAVO
gravo_time_l, gravo_rho_l = NFW_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (NFW: initial), $\\beta$: {0.5}')

gravo_time_l, gravo_rho_l = NFW_class_4.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (NFW: initial), $\\beta$: {0.34}')

# --- ISO: low dense solution
isoAndNFW_time_l, isoAndNFW_rho_l = RISO_class.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(isoAndNFW_time_l, rho_s, r_s, sigma_m), isoAndNFW_rho_l[:len(isoAndNFW_time_l)],
        label=f'Rprocedure (Isothermal)')

# --- ISO: high dense solution (MIRROR METHOD)
mirror_time_l, mirror_rho_l = Mirror_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        "--", color='black', label=f'Rprocedure (Mirror), proportion: {Mirror_class_1.data["proportion"]}')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/CheckEvolution-mirror-method-prop-2.0' + '.png', dpi=300)

    plt.close(fig)
else:
    plt.show()


# ------------------ PLOT: CENTRAL DENSITY MIRROR METHOD: PROP: 1.34 ------------------ #
fig, ax = plt.subplots(figsize=(14.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
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
# --- GRAVO
gravo_time_l, gravo_rho_l = NFW_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (NFW: initial), $\\beta$: {0.5}')

gravo_time_l, gravo_rho_l = NFW_class_2.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
        label=f'gravothermal (NFW: initial), $\\beta$: {0.6}')

# --- ISO: low dense solution
isoAndNFW_time_l, isoAndNFW_rho_l = RISO_class.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(isoAndNFW_time_l, rho_s, r_s, sigma_m), isoAndNFW_rho_l[:len(isoAndNFW_time_l)],
        label=f'Rprocedure (Isothermal)')

# --- ISO: high dense solution (MIRROR METHOD)
mirror_time_l, mirror_rho_l = Mirror_class_2.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        "--", color='black', label=f'Rprocedure (Mirror), proportion: {Mirror_class_2.data["proportion"]}')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/CheckEvolution-mirror-method-prop-1.34' + '.png', dpi=300)

    plt.close(fig)
else:
    plt.show()


# ------------------ PLOT: CENTRAL DENSITY MIRROR METHOD: COMPARISON ------------------ #
fig, ax = plt.subplots(figsize=(14.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
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
# --- ISO: low dense solution
isoAndNFW_time_l, isoAndNFW_rho_l = RISO_class.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(isoAndNFW_time_l, rho_s, r_s, sigma_m), isoAndNFW_rho_l[:len(isoAndNFW_time_l)],
        label=f'Rprocedure (Isothermal)')

# --- ISO: high dense solution (MIRROR METHOD)
mirror_time_l, mirror_rho_l = Mirror_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        label=f'Rprocedure (Mirror), proportion: {Mirror_class_1.data["proportion"]}')

mirror_time_l, mirror_rho_l = Mirror_class_3.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        label=f'Rprocedure (Mirror), proportion: {Mirror_class_3.data["proportion"]}')

mirror_time_l, mirror_rho_l = Mirror_class_4.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        label=f'Rprocedure (Mirror), proportion: {Mirror_class_4.data["proportion"]}')

mirror_time_l, mirror_rho_l = Mirror_class_5.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        label=f'Rprocedure (Mirror), proportion: {Mirror_class_5.data["proportion"]}')


# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/CheckEvolution-mirror-method-comparison' + '.png', dpi=300)

    plt.close(fig)
else:
    plt.show()

# ------------------ PLOT: CENTRAL DENSITY MIRROR METHOD: COMPARISON 2 ------------------ #
fig, ax = plt.subplots(figsize=(14.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.xlim(1 * 10 ** 1, 9 * 10 ** 2)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
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
# --- ISO: low dense solution
isoAndNFW_time_l, isoAndNFW_rho_l = RISO_class.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(isoAndNFW_time_l, rho_s, r_s, sigma_m), isoAndNFW_rho_l[:len(isoAndNFW_time_l)],
        label=f'Rprocedure (Isothermal)')

# --- ISO: high dense solution (MIRROR METHOD)
Mirror_class_1.change_proportion_value(proportion=1.32)
mirror_time_l, mirror_rho_l = Mirror_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        label=f'Rprocedure (Mirror), proportion: {Mirror_class_1.data["proportion"]}')

Mirror_class_1.change_proportion_value(proportion=1.42)
mirror_time_l, mirror_rho_l = Mirror_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        label=f'Rprocedure (Mirror), proportion: {Mirror_class_1.data["proportion"]}')

Mirror_class_1.change_proportion_value(proportion=1.52)
mirror_time_l, mirror_rho_l = Mirror_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        label=f'Rprocedure (Mirror), proportion: {Mirror_class_1.data["proportion"]}')

Mirror_class_1.change_proportion_value(proportion=1.62)
mirror_time_l, mirror_rho_l = Mirror_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(mirror_time_l, rho_s, r_s, sigma_m), mirror_rho_l[:len(mirror_time_l)],
        label=f'Rprocedure (Mirror), proportion: {Mirror_class_1.data["proportion"]}')

# --- GRAVO
gravo_time_l, gravo_rho_l = NFW_class_1.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l, "-.", color="black",
        label=f'gravothermal (NFW: initial), $\\beta$: {0.5}')

gravo_time_l, gravo_rho_l = NFW_class_2.return_rho_core_evolution(elements=2)
ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l, "--", color="grey",
        label=f'gravothermal (NFW: initial), $\\beta$: {0.6}')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/CheckEvolution-mirror-method-comparison-2' + '.png', dpi=300)

    plt.close(fig)
else:
    plt.show()
