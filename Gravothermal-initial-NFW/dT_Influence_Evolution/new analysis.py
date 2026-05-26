"""
Show some plots, which will test proper working of gravothermal simulation set up.
Probably the good test is present the evolution of core density (central density) in time.
And so on (maybe some particular behaviour in the core-collapse regime)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import auxiliaryFunctions as aux
import units as uni

# ------------------------------- USEFUL FUNCTIONS ------------------------------- #
def create_NFW_class(file_name1, file_name2, _beta):
    """create object, which contains all necessary information about gravothermal simulation"""
    #  create map, which contains all data coming from file created by mathematica code
    gravo_data1 = pd.read_csv(file_name1, sep='\t', names=['t', 'r', 'rho'])
    gravo_data2 = pd.read_csv(file_name2, sep='\t', names=['t', 'r', 'velDis'])


    gravo_data1 = gravo_data1.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r
    gravo_data2 = gravo_data2.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r

    # create class, which will contain all the necessary data
    nfw_class = gravoF.GravothermalData(gravo_data1['t'], gravo_data1['r'], gravo_data1['rho'],
                                        gravo_data2['velDis'], beta=_beta)
    # put the name to differentiate data in class
    nfw_class.put_the_name("NFW")
    return nfw_class

# ##################################### SETTINGS ##################################### #
# ------------------------------------- USER SET ------------------------------------- #
plot_save = True
output_directory = ""

# --- gravothermal
gravo_file_1 = "ρsol_M_10.1_t_10.0_sigma_0.008_beta_0.75.txt"

fileList = [gravo_file_1]
lineSettings = ['solid', 'dashed', 'dotted', 'solid', 'dotted', 'dashed']
fileNum = len(fileList)
# splitting into parts the file name
split_file_name_1 = fileList[0].split("_")

# --- set values of variables
param = dict()
param["logMvir"] = float(split_file_name_1[2])
param["mvir"] = 10 ** param["logMvir"]  # viral mass, [M_sun]
param["time_evolution"] = float(split_file_name_1[4])
param["sigma_m"] = float(split_file_name_1[6])
param["z"] = 0  # [dimensionless]
# ------------------------------------- AUTOMATIC ------------------------------------- #
param["beta"] = fileNum * [0.]
param["dT"] = fileNum * [0.]
for i in range(0, fileNum):
    split_file_name = fileList[i].split("_")
    param["beta"][i] = float(split_file_name[8])
    param["dT"][i] = float(split_file_name[10])

# --- calculate some constants
param["rho_s"] = aux.cal_rho_s(param["mvir"], param["z"])
param["r_s"] = aux.cal_r_s(param["mvir"], param["z"])

# ##################################### IMPORT DATA ##################################### #
# ------------------------------- GRAVOTHERMAL ------------------------------- #
NFW_class_list = []
for i in range(0, fileNum):
    # --- FILE i
    NFW_class = create_NFW_class("ρsol_M_10.1_t_10.0_sigma_0.008_beta_0.75.txt", "σsol_M_10.1.txt", 0.75)
    NFW_class_list.append(NFW_class)

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
plt.xlabel(r'$\hat{t} \ \left[\frac{4}{\sqrt{\pi}} \hat{\sigma}_m \hat{\nu} \hat{\rho} \right]$', fontsize=18)

plt.title(f'Central density, for: M = 10^{"{:.5f}".format(param["logMvir"])} [M_sun], '
          f'sigma_m = {"{:.1f}".format(param["sigma_m"])} [cm^2/g].', fontsize=18)
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
for i in range(0, fileNum):
    # The evolution of core density for gravothermal simulation
    gravo_time_l, gravo_rho_l = NFW_class_list[i].return_rho_core_evolution(elements=2)
    ax.plot(uni.time_tilda(gravo_time_l, param["rho_s"], param["r_s"], param["sigma_m"]), gravo_rho_l,
            linestyle=lineSettings[i],
            label=f'gravothermal (NFW: initial), $\\beta$: {param["beta"][i]}, $\\delta T$: {param["dT"][i]}')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/CheckEvolution-density' + '.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# ------------------ PLOT: CENTRAL DENSITY (Rproc) AND CORE DENSITY (GRAVO) ------------------ #
fig, ax = plt.subplots(figsize=(14.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.xlim(10 ** 2, 2 * 10 ** 3)
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
# plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
plt.xlabel(r'$\hat{t} \ \left[\frac{4}{\sqrt{\pi}} \hat{\sigma}_m \hat{\nu} \hat{\rho} \right]$', fontsize=18)

plt.title(f'Central density, for: M = 10^{"{:.5f}".format(param["logMvir"])} [M_sun], '
          f'sigma_m = {"{:.1f}".format(param["sigma_m"])} [cm^2/g].', fontsize=18)
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
for i in range(0, fileNum):
    # The evolution of core density for gravothermal simulation
    gravo_time_l, gravo_rho_l = NFW_class_list[i].return_rho_core_evolution(elements=2)
    ax.plot(uni.time_tilda(gravo_time_l, param["rho_s"], param["r_s"], param["sigma_m"]), gravo_rho_l,
            linestyle=lineSettings[i],
            label=f'gravothermal (NFW: initial), $\\beta$: {param["beta"][i]}, $\\delta T$: {param["dT"][i]}')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_directory + '/CheckEvolution-density-zoom' + '.png', dpi=300)
    plt.close(fig)
else:
    plt.show()
