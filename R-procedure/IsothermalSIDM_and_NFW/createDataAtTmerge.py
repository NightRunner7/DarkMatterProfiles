"""
In this file we're interested in getting a data at time, when the Rprocedure fails to describe evolution
of dark matter in galaxy. When and why this procedure failed at some time was mentioned in the
`IsothermalSIDMModel.py`. Shortly speaking at some time the unphysical solution become the physical one.
Thus, we evolve our system to that point and save the data, which describes the dark matter at that time
-> we will use that data as initial condition in the gravothermal simulation thanks to which we get the
results farther in the future.

Important remark: in the ending step in the evolution we somehow combine the `Isothermal` and `NFW` profile
of dark matter. To better understanding see files: `IsoAndHalo.py` and `NFWProfile.py`.

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
"""
# ################################### IMPORTING ################################### #
import os
import numpy as np
import csv  # save file
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from timeit import default_timer as timer
# ------------ FROM FILES ------------ #
import config as cfg
import auxiliaryFunctions as aux
import units as uni
from NFWProfile import NFWProfile, r1  # CDM profile (halo)
from IsothermalSIDMModel import IsoEvolution  # ISO profile (halo)
from plotsWholeComparison import plotFullComparison_vol2  # or plotFullComparison
from plotEvolutionRhoISO import plotEvolutionRhoISO

start = timer()
# ############# SETS OF SIMULATION: USER SETS! ############# #
# --- cosmological setting
sigma_m = 1.0  # [cm^2/g] annihilation cross-section
sigma_m_SI = sigma_m * 10**(-4) * 10**3  # [m^2/kg]
M_vir = 1*10**12.0  # [M_sun] Viral mass
# const_c = 15.0  # [dimensionless] concentration of DM
const_c = aux.DMc_gravo(M_vir, 0, cfg.const_h)  # [dimensionless] concentration of DM
print('Dark matter concentration:', const_c)

# --- flags and find merging
makePlots = True  # if is set to: `True` then you get plots describes evolution of `ISO`.
LowDense = True  # differentiate Low dense or High dense: which data save.
saveData = False  # if you want to save data, set to `True`
rel_err_mergin = 1.0 + 20e-2  # see `calRelErr_mergin` in `auxiliaryFunctions.py`, 1.0 + 20e-2, 1.0 + 40e-2
rel_err_mergin_vol2 = 1.0 + 5e-2  # in Isothermal Evolution this should be lower than rel_err_mergin, 1.0 + 5e-2

# --- Space resolution
num_r_to_IsoEvo = 500  # 500, how many radius points during Isothermal evolution
cff_to_Rres = 0.001  # typically: 0.001
if LowDense is True:
    """
    If we want put Isothermal profile as initial in gravothermal simulation we shouldn't take care
    that our output data cannot reproduce central velocity dispersion for low time. Because
    we're interested that profile at tmerge is correct (for that setting this is true). To differentiate
    the difference go to `compare-to-gravothermal` directory.
    """
    r_low = -2.0  # 10**r_low [dimensionless]: lowest radi
    nums_r = 400  # 400, for how many radius we want data in .txt  file, for each time step
else:
    """
    For high dense solution we actually do care about accuracy of core density (in general we
    have to reproduce central density by using data - core density).We actually care, because of
    the `Mirror method` were we reverse the time - so the beginning of the high dense became
    a results around collapse.
    """
    r_low = -4.0  # 10**r_low [dimensionless]: lowest radi
    nums_r = 600  # 600, for how many radius we want data in .txt  file, for each time step
r_up = 2.0  # 10**r_up [dimensionless]: biggest radi. Typically: `r_up=2.0 `

# --- Evolution time of galaxy
# dimensionless time
nums_time_1 = 200  # 200, 300, 50 how many we want to have time steps: to find merging
nums_time_2 = 166  # 200, 100, 0
time_tilda_start = 6*10**(-1)  # [dimensionless]
time_tilda_end_1 = 300.0  # 300, 500 [dimensionless]
time_tilda_end_2 = 500.0  # 500, 500 [dimensionless]

# --- places, where we want to have snapshots of density / mass / velocity dispersion profiles.
tage_tilda_to_plot = [3.0, 20.0, 100.0, 170.0, 330.0, 380, 400.0, 425.0, 450.0]  # [dimensionless]

# --- output plots path
output_plots_path = './Plots_Mvir%.2f_c%.1f_sigma_m%.1f' % (np.log10(M_vir), const_c, sigma_m)
try:
    # Create target Directory
    os.mkdir(output_plots_path)
    print(">>>>>>>>Directory ", output_plots_path,  " Created ")
except FileExistsError:
    print(">>>>>>>>Directory ", output_plots_path,  " already exists")
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

# tage_grid = np.logspace(-5 * 0.1, 2.5, 50)  # [Gyr] # for c = 15.0, sigma_m = 1.0, M_vir = 10**11.00
# tage_grid = np.logspace(-1.0, 2.4, 49)  #[Gyr] # for c = 15, sigma_m = 5.0, M_vir = 10**9.89 # [M_sun]

# ################################### MAKE PLOTS AND LISTS ################################### #
# ---------------- DO I HAVE PLOTS? WHERE I STORED THEM ---------------- #
tage_to_plot = []
if makePlots is True:
    for i in range(0, len(tage_tilda_to_plot)):
        tage_plot = uni.convert_time_tilda(tage_tilda_to_plot[i], rho_s, r_s, sigma_m)  # [Gyr]
        # appending
        tage_to_plot.append(tage_plot)
        tage_grid = np.append(tage_grid, tage_plot)
    tage_grid.sort()
    tage_to_plot.sort()
# print(tage_grid)
# --------------------------
diff_tim = len(tage_grid)  # how many times we set

# ---------------- AUTOMATIC SETTINGS ---------------- #
rho0_LoDens_tilda_l = []  # [dimensionless]
rho0_HiDens_tilda_l = []  # [dimensionless]
time_tilda_l = []  # [dimensionless]
time_l = []  # [Gyr]

merged_appear = False  # flag to find the first time when physical and unphysical have comparable values

# ################################### CREATING ISO_CDM CLASS ################################### #
r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage_grid[0])
IsoEvolution_class = IsoEvolution(NFW_profile, r_1, cff=cff_to_Rres, nr=num_r_to_IsoEvo,
                                  rel_err_mergin=rel_err_mergin_vol2)

# ################################### LOOP OVER TIME ################################### #
for time in range(0, diff_tim):
    print("time:", time)
    # ###################### BASIC VARIABLES ###################### #
    tage = tage_grid[time]  # [Gyr] selected time
    sim_parms = [M_vir, const_c, sigma_m, tage]  # simulation parameters

    # -------------- FINDING R1 OF OUR PROFILE --------------#
    r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage)

    # ###################### ISOTHERMAL ###################### #
    IsoEvolution_class.new_evolution_step(r_1)  # one step in evolution find central density value

    # ---------------------- SAVE NECESSARY DATA ---------------------- #
    # find the density values
    rhodm0_LoDens = IsoEvolution_class.retrun_rho0_LoDen()  # [M_sun/kpc^3]
    rhodm0_HiDens = IsoEvolution_class.return_rho0_HiDen()  # [M_sun/kpc^3]
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
    time_l.append(tage)  # [Gyr]

    # ---------------------- SAVING FILES - IF YOU WANT ---------------------- #
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

    # ---------------------- SEARCHING TMERGE AND STORED DATA ---------------------- #
    Is_merging = aux.calRelErr_mergin(rho0_HiDens_tilda, rho0_LoDens_tilda, rel_err_mergin)

    if (Is_merging is True) and merged_appear is False:
        # --- we find margin two densities
        merged_appear = True

        # --- update data at merging
        IsoEvolution_class.update_at_merging(tage, sigma_m,
                                             nr=nums_r, r_low=r_low, r_up=r_up,
                                             lowDens=LowDense)
        # --- print some important one
        print('r1 value during merging:', r_1, '[kpc]')
        print('time value during merging:', tage, '[Gyr]')

    if (Is_merging is False) and merged_appear is False:
        # save proper data in each step: after finding the merging point we are not interested in saving
        # --- update vale of central data
        IsoEvolution_class.update_central_data(lowDens=LowDense)


# ###################### AFTER LOOP: DRAW ###################### #
params_at_tmerge = IsoEvolution_class.return_params_at_tmerge()
# Time evolution of central density in physical and unphysical case
plotEvolutionRhoISO(params_at_tmerge,
                    time_tilda_l,
                    rho0_LoDens_tilda_l,
                    rho0_HiDens_tilda_l,
                    savePlot=True,
                    path_to_directory=output_plots_path)

# ###################### AFTER LOOP: COLLECT DATA ###################### #
IsoAndNFW_data = IsoEvolution_class.return_IsoAndNFW_data()
# ---------------------- COLLECT AND SAVE AS .txt FILE ---------------------- #
IS0_and_NFW_txt = dict()
IS0_and_NFW_txt['t'] = np.array(nums_r * [np.log10(params_at_tmerge['t'])])  # [log10(Gyr)] list, txt units
IS0_and_NFW_txt['r'] = np.array(uni.r_tilda(IsoAndNFW_data['r'], r_s))  # [dimensionless] list, txt units
IS0_and_NFW_txt['rho'] = np.array(uni.rho_tilda(IsoAndNFW_data['rho'], rho_s))  # [dimensionless] list, txt units
IS0_and_NFW_txt['mass'] = np.array(uni.mass_tilde(IsoAndNFW_data['mass'], rho_s, r_s))  # [dimensionless] list, txt units
IS0_and_NFW_txt['velDis'] = np.array(uni.nu_txt(IsoAndNFW_data['velDis'], r_s))  # [r_s/Gyr] list, txt units

# all data
data_m = []
for i in range(0, nums_r):
    data_m.append([IS0_and_NFW_txt['t'][i],
                   IS0_and_NFW_txt['r'][i],
                   IS0_and_NFW_txt['rho'][i],
                   IS0_and_NFW_txt['mass'][i],
                   IS0_and_NFW_txt['velDis'][i]])
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
if saveData is True:
    format_t = f'{"{:.3f}".format(params_at_tmerge["t"])}'
    format_c = f'{"{:.3f}".format(const_c)}'
    format_log10mass = f'{"{:.5f}".format(np.log10(M_vir))}'

    if LowDense is True:
        file_name = f'RisoAtTmerge_M_{format_log10mass}_t_{format_t}_sigma_m_{sigma_m}_con_{format_c}.csv'
    else:
        file_name = f'RisoAtTmerge_HiDens_M_{format_log10mass}_t_{format_t}_sigma_m_{sigma_m}_con_{format_c}.csv'
    # now we have to write the path to this file, where we want to save it
    file_path = 'data-at-tmerge' + '/' + file_name

    with open(file_path, mode="w", newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(data_m)

# ###################### PLOTTING ###################### #
# data
r_til = uni.r_tilda(IsoAndNFW_data["r"], r_s)
velDisp_til = uni.nu_tilda(IsoAndNFW_data['velDis'], r_s, rho_s)

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
ax.plot(r_til, velDisp_til, label="Dispersion velocity ISO + NFW")

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
plt.savefig(output_plots_path + '/Velocity-dispersion-tilda.png', dpi=300)
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
ax.plot(IsoAndNFW_data['r'], IsoAndNFW_data['velDis'], label="Dispersion velocity ISO + NFW")

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
plt.savefig(output_plots_path + '/Velocity-dispersion.png', dpi=300)
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
ax.plot(IsoAndNFW_data['r'], IsoAndNFW_data['mass'], label="Enclosed mass: ISO + NFW")
ax.plot(IsoAndNFW_data['r'], NFW_profile.Mass(IsoAndNFW_data['r']), label="Enclosed mass: NFW (initial)")

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
plt.savefig(output_plots_path + '/Enclosed-mass.png', dpi=300)
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
ax.plot(IsoAndNFW_data['r'], IsoAndNFW_data['rho'], label="Density: ISO + NFW")
ax.plot(IsoAndNFW_data['r'], NFW_profile.rho(IsoAndNFW_data['r']), label="Density: NFW (initial)")

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
plt.savefig(output_plots_path + '/Density-comparison.png', dpi=300)
plt.close(fig)

# -------------------------- PLOT: JUST EVOLUTION OF CENTRAL DENSITY -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylabel(r'$\rho_{0} \ \left[\rho_{s}\right]$', fontsize=18)
plt.xlabel(r'$t$ [Gyr]', fontsize=18)
plt.title(f'Density evolution: Isothermal (R-procedure)', fontsize=18)
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
ax.plot(tage_grid[0:len(IsoAndNFW_data['central-rho'])], IsoAndNFW_data['central-rho'],
        label=f'central density')

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
plt.savefig(output_plots_path + '/Central-density-check.png', dpi=300)
plt.close(fig)

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
plt.axvline(x=uni.time_tilda(params_at_tmerge['t'], rho_s, r_s, sigma_m), color='black', linestyle='dashdot',
            label=rf'$t_{{merge}}$ = '
                  rf'{"{:.3f}".format(uni.time_tilda(params_at_tmerge["t"], rho_s, r_s, sigma_m))} [dimless].')
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
ax.legend(fontsize=14)
plt.savefig(output_plots_path + '/Central-density-zoom-check.png', dpi=300)
plt.close(fig)

# -------------------------- PLOT: VELOCITY DISPERSION PROFILE AT TMERGE -------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
# log scale
ax.set_xscale('log')
# describe NFW profile
# plt.ylim(10 ** 1, 7 * 10 ** 1)
plt.ylabel(r"$\nu$ [$r_{s}$ / Gyr]", fontsize=18)
plt.xlabel(r"$t$ [Gyr]", fontsize=18)
plt.title("Dispersion velocity: Isothermal (R-procedure)", fontsize=18)
ax.tick_params(labelsize=14)

# --- DATA
ax.plot(tage_grid[0:len(IsoAndNFW_data['central-rho'])], IsoAndNFW_data['central-velDis'],
        label=f'central velocity dispersion')

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

ax.legend(fontsize=14)
plt.savefig(output_plots_path + '/Central-velocity-dispersion-check.png', dpi=300)
plt.close(fig)

end = timer()
print("script working:", end - start, "s")
