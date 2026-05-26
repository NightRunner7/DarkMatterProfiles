"""
!!!!!!!!!!!!!!!!!!!! BUGS: WORKING PROGRES !!!!!!!!!!!!!!!!!!!!


In this file we're interested in getting a data at time, when the Rprocedure fails to describe evolution
of dark matter in galaxy. When and why this procedure failed at some time was mentioned in the
`IsothermalSIDMModel.py`. Shortly speaking at some time the unphysical solution become the physical one.
Thus, we evolve our system to that point and save the data, which describes the dark matter at that time
-> we will use that data as initial condition in the gravothemal simulation thanks to which we get the
results farther in the future.

Important remark (first): in the ending step in the evolution we somehow combine the `Isothermal` and `NFW` profile
of dark matter. To better understanding see files: `CoreNFWProfile.py` and `NFWProfile.py`. Moreover, the ending
step we get from fitting to the `core NFW` profile. We do so, because previously the profile of velocity dispersion
have discontinuity (discontinuity of the derivative at one point), which is numerical obstacle during gravothermal
simulation -> in that approach, we do not have sth like this.

Important remark (second): to get some plots from evolution you can use that code. But also it exits another one,
where we save whole data during evolution.
"""
# ############# IMPORTING ############# #
import os
import sys
import numpy as np
import csv  # save file
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# ------------ SETS PATH ------------ #
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
makePlotsDir = currentdir + '\\make-plots'
# ------------ FILES PARENTDIR ------------ #
sys.path.insert(0, parentdir)
# ------------ FILES makePlotsDir ------------ #
sys.path.insert(0, makePlotsDir)
from plotsWholeComparison import plotFullComparison_vol2  # or plotFullComparison
from plotEvolutionRhoISO import plotEvolutionRhoISO
# ------------ FILES CURRENTDIR ------------ #
sys.path.insert(0, currentdir)
from NFWProfile import NFWProfile, r1  # CDM profile (halo)
from IsothermalSIDMModel import IsoEvolution  # ISO profile (halo)
import config as cfg
import auxiliaryFunctions as aux
import units as uni
# from velDisp_class import VelDisp_data
from IsoAndHalo import ISO_and_NFW  # ISO + NFW profile (halo)
from CoreNFWProfile import find_fitting_parm, CoreNFWProfile


# ############# SETS OF SIMULATION: USER SETS! ############# #
makePlots = True  # if is set to: `True` then you get plots describes evolution of `ISO`.
rel_err_margin = 1.0 + 1e-2  # see `calRelErr_mergin` in `auxiliaryFunctions.py`
sigma_m = 1.0  # [cm^2/g] annihilation cross-section
sigma_m_SI = sigma_m * 10**(-4) * 10**3  # [m^2/kg]
M_vir = 5*10**10.0  # [M_sun] Viral mass
# const_c = 15.0  # [dimensionless] concentration of DM
const_c = aux.DMc_gravo(M_vir, 0, cfg.const_h)  # [dimensionless] concentration of DM
print('Dark matter concentration:', const_c)

# tage_grid = np.logspace(-5 * 0.1, 2.5, 50)  # [Gyr] # for c = 15.2, sigma_m = 1.0, M_vir = 10**11.04
# tage_grid = np.logspace(-1.0, 2.4, 49)  #[Gyr] # for c = 15, sigma_m = 5.0, M_vir = 10**9.32 # [M_sun]

# ###################### CREATING NFW CLASS ###################### #
"""
The initial profile of DM (we starting our evolution from that)
"""
NFW_profile = NFWProfile(M_vir, const_c)
r_s = NFW_profile.r_s
rounded_r_s = cfg.rounded_number(r_s, 3)

rho_s = NFW_profile.rho_s
rounded_rho_s = cfg.rounded_number(rho_s, 3)
print(f"rho_s = {rounded_rho_s} [M_solar / kpc^3], r_s = {rounded_r_s} [kpc]")

# ###################### .txt FILE: USER SETS! ###################### #
nums_r = 500  # for how many radius we want data in .txt  file
r_tilda_start = 0.01  # [dimensionless]
r_tilda_end = 100.0  # [dimensionless]
r_start = uni.convert_r_tilda(r_tilda_start, r_s)  # [kpc]
r_end = uni.convert_r_tilda(r_tilda_end, r_s)  # [kpc]
# list which contains radius
r = np.logspace(np.log10(r_start), np.log10(r_end), nums_r)  # (array) [kpc]

# Evolution time of galaxy
nums_time = 200  # how many we want to have time steps: to find merging
time_tilda_start = 6*10**(-1)  # [dimensionless]
time_tilda_end = 540.0  # [dimensionless]

time_start = uni.convert_time_tilda(time_tilda_start, rho_s, r_s, sigma_m)  # [Gyr]
time_end = uni.convert_time_tilda(time_tilda_end, rho_s, r_s, sigma_m)  # [Gyr]
# list which contains time
tage_grid = np.logspace(np.log10(time_start), np.log10(time_end), nums_time)  # (array) [kpc]
# tage_grid = np.linspace(time_start, time_end, nums_time)  # (array) [kpc]

# ---------------- DO I HAVE PLOTS? WHERE I STORED THEM ---------------- #
tage_to_plot = []
if makePlots is True:
    tage_tilda_to_plot = [3.0, 20.0, 90.0, 170.0, 330.0, 400.0, 425.0, 455.0]  # [dimensionless]
    for i in range(0, len(tage_tilda_to_plot)):
        tage_plot = uni.convert_time_tilda(tage_tilda_to_plot[i], rho_s, r_s, sigma_m)  # [Gyr]
        # appending
        tage_to_plot.append(tage_plot)
        tage_grid = np.append(tage_grid, tage_plot)
    tage_grid.sort()
    tage_to_plot.sort()
# print(tage_grid)
output_plots_path = './PlotsCore_Mvir%.2f_c%.1f_sigma_m%.1f' % (np.log10(M_vir), const_c, sigma_m)
try:
    # Create target Directory
    os.mkdir(output_plots_path)
    print(">>>>>>>>Directory ", output_plots_path,  " Created ")
except FileExistsError:
    print(">>>>>>>>Directory ", output_plots_path,  " already exists")
# --------------------------
diff_tim = len(tage_grid)  # how many times we set

# ---------------- AUTOMATIC SETTINGS ---------------- #
rho0_LoDens_tilda_l = []  # [dimensionless]
rho0_HiDens_tilda_l = []  # [dimensionless]
time_tilda_l = []  # [dimensionless]

params_at_tmerge = {'Mvir': M_vir, 'c_const': const_c, 'cross_section': sigma_m}
params_at_tmerge['r1'] = 0.0  # [kpc], value of r1 when merging occurs
params_at_tmerge['t'] = 0.0  # merging time in [Gyr]
params_at_tmerge['t_tilda'] = 0.0  # merging time in [dimensionless]
params_at_tmerge['rho0_LoDens'] = 0.0  # fitted central density (low dense) in [M_sun/kpc^3]
params_at_tmerge['rho0_LoDens_tilda'] = 0.0  # fitted central density (low dense) in [dimensionless]
params_at_tmerge['sigma0_LoDens'] = 0.0  # fitted central velocity dispersion (low dense) in [kpc/Gyr]
params_at_tmerge['sigma0_LoDens_tilda'] = 0.0  # fitted central velDis (low dense) in [dimensionless]
params_at_tmerge['rho0_HiDens'] = 0.0  # fitted central density (high dense) in [M_sun/kpc^3]
params_at_tmerge['rho0_HiDens_tilda'] = 0.0  # fitted central density (high dense) in [dimensionless]
params_at_tmerge['sigma0_HiDens'] = 0.0  # fitted central velocity dispersion (high dense) in [kpc/Gyr]
params_at_tmerge['sigma0_HiDens_tilda'] = 0.0  # fitted central velDis (high dense) in [dimensionless]

merged_appear = False  # flag to find the first time when physical and unphysical have comparable values

# ###################### CREATING ISO_CDM CLASS ###################### #
r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage_grid[0])
ISO_CDM_class = IsoEvolution(NFW_profile, r_1)

# ###################### LOOP OVER TIME ###################### #
for time in range(0, diff_tim):
    print("time:", time)
    # ###################### BASIC VARIABLES ###################### #
    tage = tage_grid[time]  # [Gyr] selected time
    sim_parms = [M_vir, const_c, sigma_m, tage]  # simulation parameters

    # -------------- FINDING R1 OF OUR PROFILE --------------#
    r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage)

    # ###################### ISOTHERMAL ###################### #
    ISO_CDM_class.new_evolution_step(r_1)  # one step in evolution find central density value

    # ---------------------- SAVE NECESSARY DATA ---------------------- #
    # find the density values
    rhodm0_LoDens = ISO_CDM_class.retrun_rho0_LoDen()  # [M_sun/kpc^3]
    rhodm0_HiDens = ISO_CDM_class.return_rho0_HiDen()  # [M_sun/kpc^3]
    # calculate the tilda value
    rho0_LoDens_tilda = uni.rho_tilda(rhodm0_LoDens, rho_s)  # [dimensionless]
    rho0_HiDens_tilda = uni.rho_tilda(rhodm0_HiDens, rho_s)  # [dimensionless]
    # append to list
    rho0_LoDens_tilda_l.append(rho0_LoDens_tilda)  # [dimensionless]
    rho0_HiDens_tilda_l.append(rho0_HiDens_tilda)  # [dimensionless]

    # calculate time tilda time
    time_tilda = uni.time_tilda(tage, rho_s, r_s, sigma_m)  # [dimensionless]
    # append to list
    time_tilda_l.append(time_tilda)  # [dimensionless]

    # ---------------------- SAVING FILES - IF YOU WANT ---------------------- #
    # second way: third plots in one place. Without dispersion velocity.
    if makePlots is True:
        for i in range(0, len(tage_to_plot)):
            time_plot = tage_to_plot[i]
            if time_plot == tage:
                SIDM_LoDens, SIDM_HiDens = ISO_CDM_class.get_ISO_data_evolution()
                plotFullComparison_vol2(NFW_profile,
                                        SIDM_LoDens,
                                        SIDM_HiDens,
                                        r_1,
                                        sim_parms,
                                        output_plots_path)

    # ---------------------- SEARCHING TMERGE: USER SETS! ---------------------- #
    Is_merging = aux.calRelErr_mergin(rho0_HiDens_tilda, rho0_LoDens_tilda, rel_err_margin)

    if (Is_merging is True) and merged_appear is False:
        # we find margin two densities
        merged_appear = True
        # --- values of parameters
        params_at_tmerge['r1'] = r_1  # [kpc]
        params_at_tmerge['t'] = tage  # [Gyr]
        params_at_tmerge['t_tilda'] = uni.time_tilda(tage, rho_s, r_s, sigma_m)  # [dimensionless]
        params_at_tmerge['rho0_LoDens'] = rhodm0_LoDens  # [M_sun/kpc^3]
        params_at_tmerge['rho0_LoDens_tilda'] = rho0_LoDens_tilda  # [dimensionless]
        params_at_tmerge['sigma0_LoDens'] = ISO_CDM_class.return_sigma0_LoDen()  # [kpc/Gyr]
        sigma0_LoDens_tilda = uni.nu_tilda(ISO_CDM_class.return_sigma0_LoDen(), r_s, rho_s)
        params_at_tmerge['sigma0_LoDens_tilda'] = sigma0_LoDens_tilda  # [dimensionless]
        params_at_tmerge['rho0_HiDens'] = rhodm0_HiDens  # [M_sun/kpc^3]
        params_at_tmerge['rho0_HiDens_tilda'] = rho0_HiDens_tilda  # [dimensionless]
        params_at_tmerge['sigma0_HiDens'] = ISO_CDM_class.return_sigma0_HiDen()  # [kpc/Gyr]
        sigma0_HiDens_tilda = uni.nu_tilda(ISO_CDM_class.return_sigma0_HiDen(), r_s, rho_s)
        params_at_tmerge['sigma0_HiDens_tilda'] = sigma0_HiDens_tilda  # [dimensionless]

        # --- print some important one
        print('r1 value during merging:', r_1, '[kpc]')
        print('time value during merging:', tage, '[Gyr]')

        # --- Find parameter `r_c`: model `core NFW`
        # ISO and NFW profile: get data (low dense)
        SIDM_LoDens = ISO_CDM_class.get_ISO_data_evolution()[0]
        ISO_data = dict()
        ISO_data["r"] = SIDM_LoDens[4]  # radius [kpc]
        ISO_data["rho"] = SIDM_LoDens[2]  # density [M_sun/kpc^3]
        ISO_data["mass"] = SIDM_LoDens[5]  # enclosed mass [M_sun]

        # Create `ISO and NFW` profile
        ISO_and_NFW_class = ISO_and_NFW([ISO_data["r"],
                                         ISO_data["rho"],
                                         ISO_data["mass"]],
                                        params_at_tmerge)

        # using fitting function find `r_c`
        r_c = find_fitting_parm(ISO_and_NFW_class, r)[0]

        # creating a class contains `core NFW`
        coreNFW_class = CoreNFWProfile(M_vir, const_c, r_c)

        # Here we stored data from `core NFW`
        coreNFW_data = dict()
        coreNFW_data["r"] = r  # radius [kpc]
        coreNFW_data["rho"] = coreNFW_class.rho(r)  # density [M_sun/kpc^3]
        coreNFW_data["mass"] = coreNFW_class.M(r)  # enclosed mass [M_sun]
        coreNFW_data["velDis"] = coreNFW_class.sigma_accurate(r)  # velocity dispersion in [kpc/Gyr]

# ###################### AFTER LOOP: DRAW ###################### #
# Time evolution of central density in physical and unphysical case
plotEvolutionRhoISO(params_at_tmerge,
                    time_tilda_l,
                    rho0_LoDens_tilda_l,
                    rho0_HiDens_tilda_l,
                    savePlot=True,
                    path_to_directory=output_plots_path)

# ---------------------- COLLECT DATA FROM: NFW (initial) ---------------------- #
NFW_data = dict()
NFW_data['r'] = r
NFW_data['rho'] = NFW_profile.rho(r)  # density [M_sun/kpc^3]
NFW_data['mass'] = NFW_profile.Mass(r)
NFW_data['velDis'] = NFW_profile.sigma_accurate(r)

# ---------------------- COLLECT AND SAVE AS .txt FILE ---------------------- #
core_NFW_txt = dict()
core_NFW_txt['t'] = np.array(nums_r * [np.log10(params_at_tmerge['t'])])  # [log10(Gyr)] list, right units
core_NFW_txt['r'] = np.array(uni.r_tilda(r, r_s))  # [dimensionless] list, right units
core_NFW_txt['rho'] = np.array(uni.rho_tilda(coreNFW_data["rho"], rho_s))  # [dimensionless] list, right units
core_NFW_txt['mass'] = np.array(uni.mass_tilde(coreNFW_data["mass"], rho_s, r_s))  # [dimensionless] list, right units
core_NFW_txt['velDis'] = np.array(uni.nu_txt(coreNFW_data["velDis"], r_s))  # [r_s/Gyr] list, right units

# all data
data_m = []
for i in range(0, nums_r):
    data_m.append([core_NFW_txt['t'][i],
                   core_NFW_txt['r'][i],
                   core_NFW_txt['rho'][i],
                   core_NFW_txt['mass'][i],
                   core_NFW_txt['velDis'][i]])
# add parameters
keys_list = list(params_at_tmerge.keys())
for i in range(0, len(params_at_tmerge)):
    # append string: name of parameter
    str_par = keys_list[i]
    data_m[i].append(str_par)
    # read value of parameter
    val_par = params_at_tmerge[str_par]
    data_m[i].append(val_par)

# --- WRITE AND SAVE .txt FILE
format_t = f'{"{:.3f}".format(params_at_tmerge["t"])}'
format_c = f'{"{:.3f}".format(const_c)}'
format_log10mass = f'{"{:.5f}".format(np.log10(M_vir))}'

file_name = f'RprocedureCore_M_{format_log10mass}_t_{format_t}_sigma_m_{sigma_m}_con_{format_c}.csv'
# now we have to write the path to this file, where we want to save it
file_path = 'data-at-tmerge' + '/' + file_name

with open(file_path, mode="w", newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(data_m)


# ###################### PLOTTING ###################### #
# data
r_til = uni.r_tilda(r, r_s)
velDisp_til = uni.nu_tilda(coreNFW_data['velDis'], r_s, rho_s)
velDisp_til_NFW = uni.nu_tilda(NFW_data['velDis'], r_s, rho_s)


# ----- Plotting: DISPERSION VELOCITY TILDA
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
# ax.set_yscale('log')
# description
plt.ylabel(r"$\tilde{\nu}$ [dimensionless]", fontsize=18)
plt.xlabel(r"$\tilde{r}$ [dimensionless]", fontsize=18)
ax.tick_params(labelsize=14)
# data
ax.plot(r_til, velDisp_til, label="Dispersion velocity: core NFW")
ax.plot(r_til, velDisp_til_NFW, label="Dispersion velocity: NFW (initial)")

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

# lines
plt.axvline(x=1.0, color='black', linestyle='dashdot',
            label=f'r_s = {"{:.3f}".format(r_s/r_s)} [dimensionless].')
plt.axvline(x=params_at_tmerge["r1"]/r_s, color='grey', linestyle='dotted',
            label=f'r_1 = {"{:.3f}".format(params_at_tmerge["r1"]/r_s)} [dimensionless].')

ax.legend()
# plt.show()
plt.savefig(output_plots_path + '/Velocity_dispersion_tilda.png', dpi=300)
plt.close(fig)

# ----- Plotting: DISPERSION VELOCITY
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
# ax.set_yscale('log')
plt.ylabel(r"$\nu$ [kpc / Gyr]", fontsize=18)
plt.xlabel(r"$r$ [kpc]", fontsize=18)
ax.tick_params(labelsize=14)
# data
ax.plot(r, coreNFW_data['velDis'], label="Dispersion velocity: core NFW")
ax.plot(r, NFW_data['velDis'], label="Dispersion velocity: NFW (initial)")

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

# lines
plt.axvline(x=r_s, color='black', linestyle='dashdot',
            label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
plt.axvline(x=params_at_tmerge["r1"], color='grey', linestyle='dotted',
            label=f'r_1 = {"{:.3f}".format(params_at_tmerge["r1"])} [kpc].')

ax.legend()
# plt.show()
plt.savefig(output_plots_path + '/Velocity_dispersion.png', dpi=300)
plt.close(fig)

# ----- Plotting: ENCLOSED MASS
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel(r"$M(<r)$ [M_sun]", fontsize=18)
plt.xlabel(r"$r$ [kpc]", fontsize=18)
ax.tick_params(labelsize=14)
# data
ax.plot(r, coreNFW_data["mass"], label="Enclosed mass: core NFW")
ax.plot(r, NFW_data['mass'], label="Enclosed mass: NFW (initial)")

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

# lines
plt.axvline(x=r_s, color='black', linestyle='dashdot',
            label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
plt.axvline(x=params_at_tmerge["r1"], color='grey', linestyle='dotted',
            label=f'r_1 = {"{:.3f}".format(params_at_tmerge["r1"])} [kpc].')

ax.legend()
# plt.show()
plt.savefig(output_plots_path + '/Enclosed_mass.png', dpi=300)
plt.close(fig)

# ----- Plotting: DENSITY
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel(r"$\rho \, \, [M_{\odot} \, kpc^{-3}]$", fontsize=18)
plt.xlabel(r"$r$ [kpc]", fontsize=18)
ax.tick_params(labelsize=14)
# data
ax.plot(r, coreNFW_data["rho"], label="Density: core NFW")
ax.plot(r, NFW_data['rho'], label="Density: NFW (initial)")

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

# lines
plt.axvline(x=r_s, color='black', linestyle='dashdot',
            label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
plt.axvline(x=params_at_tmerge["r1"], color='grey', linestyle='dotted',
            label=f'r_1 = {"{:.3f}".format(params_at_tmerge["r1"])} [kpc].')

ax.legend()
# plt.show()
plt.savefig(output_plots_path + '/Density_comparison.png', dpi=300)
plt.close(fig)
