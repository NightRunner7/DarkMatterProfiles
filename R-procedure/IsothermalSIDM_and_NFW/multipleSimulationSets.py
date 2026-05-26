"""
This file is used to create some plots to get the feeling of our results. In general,
I would say it is the best place to somehow test and try to get the intuition of `Isothermal`
evolution, which is implemented in `IsothermalSIDMModel.py`.

During some check I find out:
    1) time_tilda_start = t/t0: refers to dimensionless time. We can start from the arbitrary
    time, but in turns out: `time_tilda_start = 6*10**(-1)` is a good choice.
    2) rel_err_mergin: refers to accuracy of distinguishable `LoDen` and `HiDen` solution of R1-procedure.
    At now, we're not having the best guess, how should be fixed.
    3) nums_r: how many radi points we want in e.g. density profile (`rho(r)`) in each time step of Isothermal
    evolution. Usually all of such profiles will be stored and then saved in `.csv` file. But to save data
    in such a way, please see `createData.py`. Typically: `nums_r = 400`.
    4) r_low: `r_s * 10^r_low` is the smallest radi, which you want to consider / save in `.csv` file.
    Typically: `r_low=-2.0`.
    5) r_up: `r_s * 10^r_low` is the biggest radi, which you want to consider / save in `.csv` file.
     Typically: `r_up=2.0`.
    6) cff_to_Rres: `cff * r_s` is a spatial resolution. Typically: `cff=0.001`. The lower spatial resolution
    is the lower time of starting simulation we can take.

One on the most important check, which we did in this code was: Did for any radi bin, we put into Isothermal
evolution could reproduce central density (`rho0`) and central velocity dispersion (`sigma0`) - after naive
applying the Isothermal method of finding density and velocity dispersion profile? It seems not.

To differentiate:
-> central value: comes from minimization procedure during evolution of Istohermal SIDM model.
-> core vale: comes post-factum (after we already knew `rho0` and `sigma0`) for fixed radi bin,
which we choose.

It turns out that:

    1) r_ToCore_elements: how many radius we take into account to calculate core (lowest radi).
    Typically: `r_ToCore_elements = 1`.
    2) r_bin_ToCore: how many radi points we consider. Those points are in range: [r_s * 10 ** r_low_ToCore, r1].
    Typically: `r_bin_ToCore = 500`. Usually won't lead to problems, if we change it.
    3) r_low_ToCore: `r_s * 10^r_low` is the smallest radi, which we consider to find e.g. `core density`.
    It turns out, that changing order of this, will lead to insufficient result, if is not properly defined:

        r_low_ToCore = log10(cff) - 1.0,

    such definition lead us to proper results (rho core reproduces central density). Same relation can be
    found in `IsothermalSIDMModel.py` file in `IsoEvolution`.
"""
# ############# IMPORTING ############# #
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from timeit import default_timer as timer
# ------------ FROM FILES ------------ #
import config as cfg  # constants
import auxiliaryFunctions as aux  # helpful functions
import units as uni
from NFWProfile import NFWProfile, r1
# --- Isothermal and NFW halo
from IsothermalSIDMModel import IsoEvolution
from plotsWholeComparison import plotFullComparison, plotFullComparison_vol2
# --- Isothermal with baryons and NFW halo
# from IsothermalSIDMModelWithBaryons import IsoEvolutionWithBaryons
# from plotsWholeComparisonBaryons import plotFullComparison, plotFullComparison_vol2


start = timer()
# ############# SETS OF SIMULATION: USER SETS! ############# #
# --- cosmological setting
sigma_m = 5.0  # [cm^2/g] annihilation cross-section
sigma_m_SI = sigma_m * 10**(-4) * 10**3  # [m^2/kg]
M_vir = 1*10**9.89  # [M_sun] Viral mass
const_c = 15.8  # [dimensionless] concentration of DM
# const_c = aux.DMc_gravo(M_vir, 0, cfg.const_h)  # [dimensionless] concentration of DM
print('Dark matter concentration:', const_c)

# --- flags and find merging
makePlots = True  # if is set to: `True` then you get plots describes evolution of `ISO`.
savePlots = True  # if is set to: `True` then you save plots in established directory.
rel_err_mergin = 1.0 + 20e-2  # see `calRelErr_mergin` in `auxiliaryFunctions.py`, 1.0 + 40e-2
rel_err_mergin_vol2 = 1.0 + 5e-2  # in Isothermal Evolution this should be lower than rel_err_mergin, 1.0 + 5e-2

# --- Space resolution
num_r_to_IsoEvo = 500  # 500, how many radius points during Isothermal evolution
cff_to_Rres = 0.001  # typically: 0.001
r_ToCore_elements = 1  # how many radius we take into account to calculate core
r_bin_ToCore = 400  # number of bins
r_low_ToCore = -4.0  # the lower value of radi in 10^x [kpc], typically: -4.0
initSetUp = False  # we enforce to produce data, which are not correlate to settings inside the definition of class

# --- Evolution time of galaxy
# dimensionless time
nums_time_1 = 200  # 100, 300, 50 how many we want to have time steps: to find merging
nums_time_2 = 200  # 200, 100, 0
time_tilda_start = 6*10**(-1)  # [dimensionless]
time_tilda_end_1 = 300  # 300, 400, 500 [dimensionless]
time_tilda_end_2 = 500  # 500 [dimensionless]

# --- places, where we want to have snapshots of density / mass / velocity dispersion profiles.
tage_tilda_to_plot = [3.0, 20.0, 40.0, 50.0, 100.0, 170.0, 330.9, 380.9, 400.9, 425.9, 450.9]  # [dimensionless]

# --- output plots path
output_plots_path = './SimulationSets_Rres%.1f_rLow%.1f_rBin%.1f_rIsoEvo%.1f_Mvir%.2f_c%.1f_sigma_m%.1f' % \
                    (np.log10(cff_to_Rres), r_low_ToCore, r_bin_ToCore, num_r_to_IsoEvo, np.log10(M_vir), const_c, sigma_m)
if makePlots is True or savePlots is True:
    try:
        # Create target Directory
        os.mkdir(output_plots_path)
        print(">>>>>>>> Directory ", output_plots_path,  " Created ")
    except FileExistsError:
        print(">>>>>>>> Directory ", output_plots_path,  " already exists")

# ################################### CREATING NFW CLASS ################################### #
"""
The initial profile of DM (we starting our evolution from that)
"""
NFW_profile = NFWProfile(M_vir, const_c)
r_s = NFW_profile.r_s
rounded_r_s = cfg.rounded_number(r_s, 3)

rho_s = NFW_profile.rho_s
rounded_rho_s = cfg.rounded_number(rho_s, 3)
print(f"rho_s = {rounded_rho_s} [M_solar / kpc^3], r_s = {rounded_r_s} [kpc]")

# ################################### TIME LIST ################################### #
# dimensional time
time_start = uni.convert_time_tilda(time_tilda_start, rho_s, r_s, sigma_m)  # [Gyr]
time_end_1 = uni.convert_time_tilda(time_tilda_end_1, rho_s, r_s, sigma_m)  # [Gyr]
time_end_2 = uni.convert_time_tilda(time_tilda_end_2, rho_s, r_s, sigma_m)  # [Gyr]

# list which contains time
tage_grid_log = np.logspace(np.log10(time_start), np.log10(time_end_1), nums_time_1, endpoint=False)  # (array) [kpc]
tage_grid_lin = np.linspace(time_end_1, time_end_2, nums_time_2)  # (array) [kpc]
tage_grid = [*tage_grid_log, *tage_grid_lin]
tage_grid.sort()

# tage_grid = np.logspace(-5 * 0.1, 2.5, 50)  # [Gyr] # for c = 15.8, sigma_m = 1.0, M_vir = 10**11.00
# tage_grid = np.logspace(-1.0, 2.4, 49)  #[Gyr] # for c = 15, sigma_m = 5.0, M_vir = 10**9.89 # [M_sun]

# ################################### MAKE PLOTS AND LISTS ################################### #
# -------------------------- DO I HAVE PLOTS? WHERE I STORED THEM -------------------------- #
tage_to_plot = []
if makePlots is True:
    for i in range(0, len(tage_tilda_to_plot)):
        tage_plot = uni.convert_time_tilda(tage_tilda_to_plot[i], rho_s, r_s, sigma_m)  # [Gyr]
        # appending
        tage_to_plot.append(tage_plot)
        tage_grid = np.append(tage_grid, tage_plot)
    tage_grid.sort()
    tage_to_plot.sort()
# path to output directory
# --------------------------
diff_tim = len(tage_grid)  # how many times we set

# -------------------------- AUTOMATIC SETTINGS -------------------------- #
rho0_LoDens_tilda_l = []  # [dimensionless]
rho0_HiDens_tilda_l = []  # [dimensionless]
time_tilda_l = []  # [dimensionless]
rhoCore_LoDens_tilda_l = []
rhoCore_HiDens_tilda_l = []
parameters_to_plot = []

r1merge = 0.0  # [kpc], value of r1 when merging occurs
tmerge = 0.0  # merging time in [Gyr]
tmerge_tilda = 0.0  # [dimensionless]
rho0_merge = 0.0  # [M_sun/kpc^3]
rho0_tilda_merge = 0.0  # [dimensionless]

merged_appear = False  # flag to find the first time when physical and unphysical have comparable values

# ###################### CREATING ISO_CDM CLASS ###################### #
r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage_grid[0])
IsoEvolution_class = IsoEvolution(NFW_profile, r_1, cff=cff_to_Rres,
                                  nr=num_r_to_IsoEvo, rel_err_mergin=rel_err_mergin_vol2)
# IsoEvolution_class = IsoEvolutionWithBaryons(NFW_profile, r_1, cff=cff_to_Rres,
#                                              nr=num_r_to_IsoEvo, rel_err_mergin=rel_err_mergin_vol2)

# ###################### LOOP OVER TIME ###################### #
for time in range(0, diff_tim):
    print("time:", time)
    # -------------------------- BASIC VARIABLES -------------------------- #
    tage = tage_grid[time]  # [Gyr] selected time
    sim_parms = [M_vir, const_c, sigma_m, tage]  # simulation parameters

    # -------------------------- FINDING R1 OF OUR PROFILE --------------------------#
    r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage)

    # -------------------------- ISOTHERMAL -------------------------- #
    IsoEvolution_class.new_evolution_step(r_1)  # one step in evolution find central density value
    # IsoEvolution_class.new_evolution_step_naive(r_1)  # one step in evolution find central density value


    # -------------------------- SAVING FILES - IF YOU WANT -------------------------- #
    # first way: fourth plots in one place.
    # for time in tage_to_plot:
    #     if tage == time:
    #          plotFullComparison(NFW_profile, SIDM_LoDens, SIDM_HiDens, r_1, sim_parms)

    # second way: third plots in one place. Without dispersion velocity.
    if makePlots is True:
        for i in range(0, len(tage_to_plot)):
            time_plot = tage_to_plot[i]
            if time_plot == tage:
                SIDM_LoDens, SIDM_HiDens = IsoEvolution_class.get_ISO_data_evolution()
                plotFullComparison_vol2(NFW_profile,
                                        SIDM_LoDens,
                                        SIDM_HiDens,
                                        r_1,
                                        sim_parms,
                                        output_plots_path)

    # -------------------------- GET RHO_TILDA AND T_TILDA -------------------------- #
    # find the density values
    rho0_LoDens = IsoEvolution_class.retrun_rho0_LoDen()  # [M_sun/kpc^3]
    rho0_HiDens = IsoEvolution_class.return_rho0_HiDen()  # [M_sun/kpc^3]
    # calculate the tilda value
    rho0_LoDens_tilda = uni.rho_tilda(rho0_LoDens, rho_s)
    rho0_HiDens_tilda = uni.rho_tilda(rho0_HiDens, rho_s)
    # append to list
    rho0_LoDens_tilda_l.append(rho0_LoDens_tilda)
    rho0_HiDens_tilda_l.append(rho0_HiDens_tilda)

    # calculate time tilda time
    time_tilda = uni.time_tilda(tage, rho_s, r_s, sigma_m)
    # append to list
    time_tilda_l.append(time_tilda)
    # -------------------------- GET RHO CORE -------------------------- #
    # find core density
    rho_core_Low = IsoEvolution_class.return_rho_core_LoDen(elements=r_ToCore_elements, nr=r_bin_ToCore,
                                                            r_low=r_low_ToCore,
                                                            initSetUp=False)

    rho_core_Hi = IsoEvolution_class.return_rho_core_HiDen(elements=r_ToCore_elements, nr=r_bin_ToCore,
                                                           r_low=r_low_ToCore,
                                                           initSetUp=False)
    # calculate the tilda value
    rho_core_Low_tilda = uni.rho_tilda(rho_core_Low, rho_s)
    rho_core_Hi_tilda = uni.rho_tilda(rho_core_Hi, rho_s)
    # append to list
    rhoCore_LoDens_tilda_l.append(rho_core_Low_tilda)
    rhoCore_HiDens_tilda_l.append(rho_core_Hi_tilda)

    # ------------------------------ SEARCHING FOR TMERGED TIME ------------------------------ #
    Is_merging = aux.calRelErr_mergin(rho0_HiDens_tilda, rho0_LoDens_tilda, rel_err_mergin)

    if (Is_merging is True) and (merged_appear is False):
        # set merging values
        tmerge_tilda = time_tilda
        tmerge = tage
        rho0_tilda_merge = rho0_LoDens_tilda
        merged_appear = True

# ###################### AFTET THE LOOPING: PLOT RHO EVOLUTION ###################### #
print("rho0_LoDens_tilda_l:", rho0_LoDens_tilda_l)
print("rho0_HiDens_tilda_l", rho0_HiDens_tilda_l)
print("time_tilda_l:", time_tilda_l)
print(f"tmerged: {tmerge} [Gyr]")
print(f"tmerged: {tmerge_tilda} [dimensionless]")


fig, ax = plt.subplots(figsize=(11.0, 7.0))
# solutions: low and high density
ax.plot(time_tilda_l[:nums_time_1], rho0_LoDens_tilda_l[:nums_time_1],
        color="blue", label="SIDM: low density", zorder=1)
ax.plot(time_tilda_l[nums_time_1:], rho0_LoDens_tilda_l[nums_time_1:],
        color="blue", zorder=1)

ax.plot(time_tilda_l[:nums_time_1], rho0_HiDens_tilda_l[:nums_time_1],
        color="orange", label="SIDM: high density", zorder=2)
ax.plot(time_tilda_l[nums_time_1:], rho0_HiDens_tilda_l[nums_time_1:],
        color="orange", zorder=2)
# merging
if tmerge_tilda > 0.0:
    rounded_tmarged = cfg.rounded_number(tmerge_tilda, 5)
else:
    rounded_tmarged = 0.0
if tmerge > 0.0:
    rounded_tmarged_Gyr = cfg.rounded_number(tmerge, 5)
else:
    rounded_tmarged_Gyr = 0.0
ax.scatter(tmerge_tilda, rho0_tilda_merge, marker='o', s=50.0, zorder=3,
           label=f'merging two solutions at: {rounded_tmarged} ,' + '\n' +
                 f'which refers to {rounded_tmarged_Gyr} [Gyr].',
           facecolor='grey', edgecolor='black', linewidth=1.0, rasterized=True)

# --- log-linear scale
ax.set_xscale('log')
ax.set_yscale('log')
# scale
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.xlim(10 ** (-1), 8 * 10 ** 2)
# plt.xlim(10 ** (-1), 2 * 10 ** 3)
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
# describe NFW profile
plt.ylabel(r"$\hat{\rho_{0}}$ $\left[ \rho_s \right]$", fontsize=18)
plt.xlabel(r"$\hat{t}$ $\left[\frac{4}{\sqrt{\pi}} \hat{\sigma_m} \hat{\nu} \hat{\rho} \right]$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
# title
plt.title(r'Isothermal solution. $M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}$' \
        % (np.log10(M_vir), const_c, sigma_m), fontsize=14)
# ---save figure
if savePlots is True:
    plt.savefig(output_plots_path + f'/Central-density-timeBins-{diff_tim}-'
                                    f'tmerge-{"{:.3f}".format(tmerge)}-relM-{rel_err_mergin}.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# -------------------------- PLOT: JUST EVOLUTION OF CENTRAL DENSITY: ZOOM -------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# --- DATA, solutions: low and high density
ax.plot(time_tilda_l[:nums_time_1], rho0_LoDens_tilda_l[:nums_time_1],
        color="blue", label="SIDM: low density", zorder=1)
ax.plot(time_tilda_l[nums_time_1:], rho0_LoDens_tilda_l[nums_time_1:],
        color="blue", zorder=1)

ax.plot(time_tilda_l[:nums_time_1], rho0_HiDens_tilda_l[:nums_time_1],
        color="orange", label="SIDM: high density", zorder=2)
ax.plot(time_tilda_l[nums_time_1:], rho0_HiDens_tilda_l[nums_time_1:],
        color="orange", zorder=2)
# --- LINES
plt.axvline(x=tmerge_tilda, color='black', linestyle='dashdot',
            label=rf'$t_{{merge}}$ = {"{:.3f}".format(tmerge_tilda)} [dimensionless].')
# log scale
ax.set_xscale('symlog', linthresh=400)
ax.set_yscale('log')
plt.ylim(10 ** 0, 6 * 10 ** 0)
plt.xlim(1 * 10 ** 1, 6 * 10 ** 2)
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
# describe NFW profile
plt.ylabel(r"$\tilde{\rho_{0}}$", fontsize=18)
plt.xlabel(r"$\tilde{t}$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
# title
plt.title(r'Isothermal solution. $M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}$' \
        % (np.log10(M_vir), const_c, sigma_m), fontsize=14)
# ---save figure
if savePlots is True:
    plt.savefig(output_plots_path + f'/Central-density-zoom-timeBins-{diff_tim}-'
                                    f'tmerge-{"{:.3f}".format(tmerge)}-relM-{rel_err_mergin}.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# -------------------------- PLOT: JUST EVOLUTION OF CENTRAL DENSITY (LOW DENSE) -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$t$ [Gyr]', fontsize=18)
plt.title(r'Isothermal (high dense). $M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}, r_{low}=%.1f$' \
          % (np.log10(M_vir), const_c, sigma_m, r_low_ToCore), fontsize=13)
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
ax.plot(tage_grid, rho0_LoDens_tilda_l, label="central density", zorder=1)
ax.plot(tage_grid, rhoCore_LoDens_tilda_l, "--", label="core density", zorder=2)

# --- LINES
plt.axvline(x=tmerge, color='black', linestyle='dashdot',
            label=rf'$t_{{merge}}$ = {"{:.3f}".format(tmerge)} [Gyr].')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if savePlots is True:
    plt.savefig(output_plots_path + f'/Central-density-low-dense-timeBins-{diff_tim}-'
                                    f'tmerge-{"{:.3f}".format(tmerge)}-relM-{rel_err_mergin}.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# -------------------------- PLOT: JUST EVOLUTION OF CENTRAL DENSITY (HIGH DENSE) -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$t$ [Gyr]', fontsize=18)
plt.title(r'Isothermal (low dense). $M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}, r_{low}=%.1f$' \
          % (np.log10(M_vir), const_c, sigma_m, r_low_ToCore), fontsize=13)
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
ax.plot(tage_grid, rho0_HiDens_tilda_l, label="central density", zorder=1)
ax.plot(tage_grid, rhoCore_HiDens_tilda_l, "--", label="core density", zorder=2)

# --- LINES
plt.axvline(x=tmerge, color='black', linestyle='dashdot',
            label=rf'$t_{{merge}}$ = {"{:.3f}".format(tmerge)} [Gyr].')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if savePlots is True:
    plt.savefig(output_plots_path + f'/Central-density-high-dense-timeBins-{diff_tim}-'
                                    f'tmerge-{"{:.3f}".format(tmerge)}-relM-{rel_err_mergin}.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# -------------------------- PLOT: JUST EVOLUTION OF CENTRAL DENSITY (HIGH DENSE) vol 2 -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$t$ [dimensionless]', fontsize=18)
plt.title(r'Isothermal (low dense). $M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}, r_{low}=%.1f$' \
          % (np.log10(M_vir), const_c, sigma_m, r_low_ToCore), fontsize=13)
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
ax.plot(time_tilda_l, rho0_HiDens_tilda_l, label="central density", zorder=1)
ax.plot(time_tilda_l, rhoCore_HiDens_tilda_l, "--", label="core density", zorder=2)

# --- LINES
plt.axvline(x=tmerge_tilda, color='black', linestyle='dashdot',
            label=rf'$t_{{merge}}$ = {"{:.3f}".format(tmerge_tilda)} [dimensionless].')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if savePlots is True:
    plt.savefig(output_plots_path + f'/Central-density-high-dense-timeBins-{diff_tim}-'
                                    f'tmerge-{"{:.3f}".format(tmerge)}-relM-{rel_err_mergin}.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

end = timer()
print("script working:", end - start, "s")
