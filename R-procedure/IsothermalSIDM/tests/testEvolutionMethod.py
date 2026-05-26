# ############# IMPORTING ############# #
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pylab as pyl
from timeit import default_timer as timer
# ------------ FROM FILES ------------ #
import config as cfg  # constants
import auxiliaryFunctions as aux  # helpful functions
import units as uni
from NFWProfile import NFWProfile, r1
# --- Isothermal and NFW halo
from IsothermalSIDMModel import IsoEvolution, deltaSqare
# --- Isothermal with baryons and NFW halo
# from IsothermalSIDMModelWithBaryons import IsoEvolutionWithBaryons

# --- Helpfully function
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

start = timer()
# ############# SETS OF SIMULATION: USER SETS! ############# #
# --- cosmological setting
sigma_m = 5.0  # [cm^2/g] annihilation cross-section
sigma_m_SI = sigma_m * 10**(-4) * 10**3  # [m^2/kg]
M_vir = 1*10**9.89  # [M_sun] Viral mass
# const_c = 15.8  # [dimensionless] concentration of DM
const_c = aux.DMc_gravo(M_vir, 0, cfg.const_h)  # [dimensionless] concentration of DM
print('Dark matter concentration:', const_c)

# --- flags and find merging
makePlots = False  # if is set to: `True` then you get plots describes evolution of `ISO`.
savePlots = True  # if is set to: `True` then you save plots in established directory.
rel_err_mergin = 1.0 + 20e-2  # see `calRelErr_mergin` in `auxiliaryFunctions.py`, 1.0 + 40e-2
rel_err_mergin_vol2 = 1.0 + 5e-2  # in Isothermal Evolution this should be lower than rel_err_mergin, 1.0 + 5e-2

# --- Space resolution
num_r_to_IsoEvo = 504  # 500, how many radius points during Isothermal evolution
cff_to_Rres = 0.001  # typically: 0.001

# --- Evolution time of galaxy
# dimensionless time
# nums_time_1 = 200  # 100, 300, 50 how many we want to have time steps: to find merging
# nums_time_2 = 60  # 200, 100, 0
# nums_time_3 = 100  # 200, 100, 0
# nums_time_4 = 35  # 200, 100, 0
# time_tilda_start = 6*10**(-1)  # [dimensionless]
# time_tilda_end_1 = 300  # 300, 400, 500 [dimensionless]
# time_tilda_end_2 = 360  # 360 [dimensionless]
# time_tilda_end_3 = 380  # 380 [dimensionless]
# time_tilda_end_4 = 450  # 450 [dimensionless]

# Jiangs model
nums_time_1 = 400
time_tilda_start = 6*10**(-1)  # [dimensionless]
time_tilda_end_1 = 376  # 300, 400, 500 [dimensionless]

# --- output plots path
output_plots_path = './SimulationSets_Rres%.1f_rIsoEvo%.1f_Mvir%.2f_c%.1f_sigma_m%.1f' % \
                    (np.log10(cff_to_Rres), num_r_to_IsoEvo, np.log10(M_vir), const_c, sigma_m)
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
# # dimensional time
# time_start = uni.convert_time_tilda(time_tilda_start, rho_s, r_s, sigma_m)  # [Gyr]
# time_end_1 = uni.convert_time_tilda(time_tilda_end_1, rho_s, r_s, sigma_m)  # [Gyr]
# time_end_2 = uni.convert_time_tilda(time_tilda_end_2, rho_s, r_s, sigma_m)  # [Gyr]
# time_end_3 = uni.convert_time_tilda(time_tilda_end_3, rho_s, r_s, sigma_m)  # [Gyr]
# time_end_4 = uni.convert_time_tilda(time_tilda_end_4, rho_s, r_s, sigma_m)  # [Gyr]
#
# # list which contains time
# tage_grid_log = np.logspace(np.log10(time_start), np.log10(time_end_1), nums_time_1, endpoint=False)  # (array) [kpc]
# tage_grid_lin_1 = np.linspace(time_end_1, time_end_2, nums_time_2, endpoint=False)  # (array) [kpc]
# tage_grid_lin_2 = np.linspace(time_end_2, time_end_3, nums_time_2, endpoint=False)  # (array) [kpc]
# tage_grid_lin_3 = np.linspace(time_end_3, time_end_4, nums_time_2)  # (array) [kpc]
#
# tage_grid = [*tage_grid_log, *tage_grid_lin_1, *tage_grid_lin_2, *tage_grid_lin_3]
# tage_grid.sort()

# Jings model
time_start = uni.convert_time_tilda(time_tilda_start, rho_s, r_s, sigma_m)  # [Gyr]
time_end_1 = uni.convert_time_tilda(time_tilda_end_1, rho_s, r_s, sigma_m)  # [Gyr]
tage_grid = np.logspace(np.log10(time_start), np.log10(time_end_1), nums_time_1, endpoint=False)  # (array) [kpc]


# tage_grid = np.logspace(-5 * 0.1, 2.5, 50)  # [Gyr] # for c = 15.8, sigma_m = 1.0, M_vir = 10**11.00
# tage_grid = np.logspace(-1.0, 2.4, 49)  #[Gyr] # for c = 15, sigma_m = 5.0, M_vir = 10**9.89 # [M_sun]

diff_tim = len(tage_grid)  # how many times we set

# ################################### LISTS ################################### #
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

    print("time_tilda:", time_tilda, "r1:", r_1)

    # ------------------------------ SEARCHING FOR TMERGED TIME ------------------------------ #
    Is_merging = aux.calRelErr_mergin(rho0_HiDens_tilda, rho0_LoDens_tilda, rel_err_mergin)

    if (Is_merging is True) and (merged_appear is False):
        # set merging values
        tmerge_tilda = time_tilda
        tmerge = tage
        rho0_tilda_merge = rho0_LoDens_tilda
        merged_appear = True

    # ##################### MAKING CONTOUR PLOTS ############################################## #
    if makePlots and (329.0 < time_tilda < 391.0):
        # take or set necessary data: values
        rhodm0_LoDens = IsoEvolution_class.retrun_rho0_LoDen()
        rhodm0_HiDens = IsoEvolution_class.return_rho0_HiDen()
        sigma0_LoDens = IsoEvolution_class.return_sigma0_LoDen()
        sigma0_HiDens = IsoEvolution_class.return_sigma0_HiDen()
        Rres = cfg.find_Rres(NFW_profile, cff=cff_to_Rres)
        r_low = np.log10(cff_to_Rres) - 1.0

        print("-----------------------------------------------------------")
        print("time tilda:", time_tilda)
        print(f"rho0 Low dense: {rhodm0_LoDens:.4e} [M_sun/kpc^3].")
        print(f"rho0 High dense: {rhodm0_HiDens:.4e} [M_sun/kpc^3].")
        print(f"sigma0 Low dense: {sigma0_LoDens:.4e} [kpc/Gyr].")
        print(f"sigma0 High dense: {sigma0_HiDens:.4e} [kpc/Gyr].")

        # take or set necessary data: lists
        r = np.logspace(np.log10(r_s * 10 ** r_low), np.log10(r_1), num_r_to_IsoEvo)

        # ------------------------ PLOT CONTROL ------------------------ #
        # lw = 2.5
        size = 50.
        edgewidth = 0.
        outfig1 = output_plots_path + '/IsothermalAndCMD_Mvir%.2f_c%.1f_sigmamx%.1f_That%.4f.png'  # %(M_vir,c,sigmamx,tage)

        # ------------------------ SETTING POSSIBLE VALUES OF PARAMETERS ------------------------ #
        # --- density
        # beg_power = float(np.log10(NFW_profile.rho(r_1)))  # where we have to start
        # rho_dm0 = np.logspace(beg_power, 14., 100)  # instabilities occurring here
        # rho_dm0 = np.logspace(beg_power, 13., 100)

        rho_dm0 = np.logspace(7., 9., 200)

        # --- velocity dispersion
        sigma_beg = 0.5 * NFW_profile.sigma_accurate(r_1) * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
        sigma_end = 2 * NFW_profile.sigma_accurate(r_1) * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s
        sigma_0 = np.logspace(np.log10(sigma_beg), np.log10(sigma_end), 200)

        # ------------------------ NFW PROFILE ------------------------ #
        rhoCDM1 = NFW_profile.rho(r_1)
        MCDM1 = NFW_profile.Mass(r_1)
        rhoCDMRres = NFW_profile.rho(Rres)
        nu_at_r1 = NFW_profile.sigma_accurate(r_1)  # [kpc/Gyr]

        # Jinag algorithm
        # LoDen_upperBound = np.sqrt(7e-1 * rhoCDM1 * rhoCDMRres)

        # our algorithm
        rhoLoDen_pre = IsoEvolution_class.rho0_pre
        rhoHiDen_pre = IsoEvolution_class.rho0_HiDen_pre
        LoDen_upperBound = np.sqrt(rhoLoDen_pre * rhoHiDen_pre)
        # ------------------------ CALCULATE DELTA ------------------------ #
        def cal_delta(rho_dm0, sigma_0):
            """
            calculate the deltaSqare, which was appeared in file IsothermalSIDMModel.py
            """
            log_rho_dm0 = np.log10(rho_dm0)
            log_sigma_0 = np.log10(sigma_0)
            p_list = [log_rho_dm0, log_sigma_0]
            p = np.array(p_list)

            return deltaSqare(p, r_s, rhoCDM1, MCDM1, r)


        # ------------------------ PREPARE AXIX IN CONTOUR PLOT ------------------------ #
        x = sigma_0  # sigma_0 = np.logspace(np.log10(30), np.log10(118), 500)
        y = rho_dm0  # rho_dm0 = np.logspace(7., 14., 500)
        X, Y = pyl.meshgrid(x, y)  # grid of point

        # creating the Z's values (log delta)
        Z = []
        for yy in y:
            zy = []
            for xx in x:
                # calculate log delta
                log_delta = np.log10(cal_delta(yy, xx)) * 1 / 2
                zy.append(log_delta)

            Z.append(zy)

        # ------------------------ PLOTTING ------------------------#
        fig, ax = plt.subplots(figsize=(15.0, 12.0))

        # Z_to_show = np.arange(min(min(Z)),max(max(Z)),.1) #Adjust the .001 to get finer gradient
        # Z_to_show = [-2., -1.5, -1., -0.5, 0.]

        CS = ax.contourf(X, Y, Z, levels=50, cmap='nipy_spectral')

        # axis
        plt.ylim(9 * 10 ** 6, 2 * 10 ** 9)
        plt.xlim(0.9 * 0.5 * nu_at_r1, 1.4 * 2.0 * nu_at_r1)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Apply the log scale and grid settings
        apply_log_scale_grid(ax)

        # ax2.set_title('$\log \delta$', fontsize=18 )
        ax.set_xlabel(r"$v_{0}$ [km/s]", fontsize=18)
        ax.set_ylabel(r"$rho_{0} \, \, [M_{\odot} \, kpc^{-3}]$", fontsize=18)

        # scatter

        ax.scatter(sigma0_LoDens, rhodm0_LoDens, marker="*", s=size,
                   facecolor='k', edgecolor='k', linewidth=edgewidth, rasterized=True)
        ax.scatter(sigma0_HiDens, rhodm0_HiDens, marker="*", s=size,
                   facecolor='r', edgecolor='r', linewidth=edgewidth, rasterized=True)
        # lines
        plt.axvline(x=0.5 * nu_at_r1, color='black', linestyle='--',
                    label=f'0.5 * Velocity dispersion (nu) at r1: {"{:.0f}".format(0.5 * nu_at_r1)} [km/s].')
        plt.axvline(x=2 * nu_at_r1, color='black', linestyle='--',
                    label=f'2 * Velocity dispersion (nu) at r1: {"{:.0f}".format(2 * nu_at_r1)} [km/s].')
        plt.axhline(y=LoDen_upperBound, color='black', linestyle='--',
                    label=f'LoDen_upperBound.')

        # annotation to lines
        ax.text(0.5 * nu_at_r1, 2. * ax.get_ylim()[0], r'$0.5\, v(r_{1})$', color='k', fontsize=16,
                ha='right', va='bottom', transform=ax.transData, rotation=90)
        ax.text(2 * nu_at_r1, 2. * ax.get_ylim()[0], r'$2\, v(r_{1})}$', color='k', fontsize=16,
                ha='left', va='bottom', transform=ax.transData, rotation=90)

        # Make a colorbar for the ContourSet returned by the contourf call.
        cbar = fig.colorbar(CS, ticks=mticker.MaxNLocator(6))
        cbar.ax.set_ylabel(r'$\log \delta$', fontsize=18)
        plt.title(f"evolution time: {time_tilda}")
        # ---save figure
        plt.savefig(outfig1 % (M_vir, const_c, sigma_m, time_tilda), dpi=300)
        # fig1.canvas.manager.window.raise_()
        # plt.get_current_fig_manager().window.setGeometry(50,50,1200,1200)

        # ---we have to close figure.
        plt.close(fig)


# ###################### AFTET THE LOOPING: PLOT RHO EVOLUTION ###################### #
print("r1 at tmerge in Iso class:", IsoEvolution_class.IsoAndNFW_data["r1-check-merge"])
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

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

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
    plt.savefig(output_plots_path + f'/Central-density.png', dpi=300)
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

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

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
    plt.savefig(output_plots_path + f'/Central-density-zoom-timeBins.png', dpi=300)
    plt.close(fig)
else:
    plt.show()

# -------------------------- PLOT: EVOLUTION OF log delta -------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 6.0))
# --- DATA, solutions: low and high density
logDelta_LoDen_arr, logDelta_HiDen_arr = IsoEvolution_class.return_logDelta_evolution()
ax.plot(time_tilda_l, logDelta_LoDen_arr,
        color="blue", label="SIDM: low density", zorder=1)
ax.plot(time_tilda_l, logDelta_HiDen_arr,
        color="orange", label="SIDM: high density", zorder=2)
# --- LINES
plt.axvline(x=tmerge_tilda, color='black', linestyle='dashdot',
            label=rf'$t_{{merge}}$ = {"{:.3f}".format(tmerge_tilda)} [dimensionless].')
# log scale
ax.set_xscale('log')

# describe axis
plt.ylabel(r"$\log{\delta}$", fontsize=18)
plt.xlabel(r"$\tilde{t}$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
# title
plt.title(r'Minimization. $M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}$' \
        % (np.log10(M_vir), const_c, sigma_m), fontsize=14)
# ---save figure
if savePlots is True:
    plt.savefig(output_plots_path + f'/log-delta-comparison.png', dpi=300)
    plt.close(fig)
else:
    plt.show()


end = timer()
print("script working:", end - start, "s")
