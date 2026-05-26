import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pylab as pyl
# import files
import os
import sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
# take files from parent dir
sys.path.insert(0, parentdir)
import config as cfg
from IsothermalSIDMModel import deltaSqare

def plotContourBasic(NFW_profile, SIDM_LoDens, SIDM_HiDens, r_1):
    """
    Draw the plot, which presents how looks the value of log10(delta) for fixed values of parameters
    (rho_dm0, sigma_0). Delta denotes relative error in stitching the isothermal-core profile
    to the outer CDM-like profile.

    where

        NFW_profile: class, which contains all methods necessary to deal with profile CDM-only.
        SIDM_LoDens: list contains all necessary data, which is corresponding to physical Isothermal
        solution.
        SIDM_LoDens: list contains all necessary data, which is corresponding to unphysical Isothermal
        solution.
        val_r1: the distance where we're switching formula from Isothermal to NFW. [kpc]
    """
    # ------------------------ PLOT CONTROL ------------------------ #
    lw = 2.5
    size = 50.
    edgewidth = 0.

    r_FullRange = np.logspace(-3, 3, 200)  # [kpc] for plotting the full profile

    # ------------------------ SETTING POSSIBLE VALUES OF PARAMETERS ------------------------ #
    beg_power = float(np.log10(NFW_profile.rho(r_1)))  # where we have to start

    rho_dm0 = np.logspace(beg_power, 14., 100)
    sigma_beg = 0.5 * NFW_profile.sigma_accurate(r_1) * cfg.kpc_SI / cfg.Gyr * 10**(-3) # [km/s]
    sigma_end = 2 * NFW_profile.sigma_accurate(r_1) * cfg.kpc_SI / cfg.Gyr * 10**(-3) # [km/s]
    sigma_0 = np.logspace(np.log10(sigma_beg), np.log10(sigma_end), 100)

    # ------------------------ NFW PROFILE ------------------------ #
    rhoCDM1 = NFW_profile.rho(r_1)
    MCDM1 = NFW_profile.Mass(r_1)
    r_s = NFW_profile.r_s
    # the radiuses of using isothermal profile
    r = np.logspace(-3., np.log10(r_1), 500)

    # ------------------------ PRINTING IMPORTANT VALUES ------------------------ #
    # printing important values - which we will present on plot
    print(f"CDM-olny, rho value at r1: {'{:.2E}'.format(rhoCDM1)} [M_sun / kpc^3] --> rho(r_1) ")
    rhoCDMRres = NFW_profile.rho(0.01)
    print(f"CDM-olny, rho value at spatial resolution (0.01 kpc): {'{:.2E}'.format(rhoCDMRres)} [M_sun / kpc^3]. --> rho(r_res)")

    # velocity dispersion
    nu_at_r1 = NFW_profile.sigma_accurate(r_1) #[kpc/Gyr]
    nu_at_r1_SI =nu_at_r1 * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
    print(f"Velocity dispersion (nu) at r1: {'{:.1f}'.format(nu_at_r1_SI)} [km/s] --> v(r_1)")

    # circular velocity
    Vcirc_at_r1 = NFW_profile.Vcirc(r_1)  # [kpc/Gyr]
    Vcirc_at_r1_SI = Vcirc_at_r1 * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
    # print(f"Circular velocity at r1: {'{:.1f}'.format(Vcirc_at_r1_SI)} [km/s]")

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
    origin = 'lower'

    fig, ax2 = plt.subplots(constrained_layout=True, figsize=(9.0, 6.0))

    # Z_to_show = np.arange(min(min(Z)),max(max(Z)),.1) #Adjust the .001 to get finer gradient
    # Z_to_show = [-2., -1.5, -1., -0.5, 0.]

    CS = ax2.contourf(X, Y, Z, levels=50, cmap='nipy_spectral')

    # axis
    plt.ylim(10 ** 5, 2 * 10 ** 14)
    plt.xlim(10 ** 1, 2 * 10 ** 2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=14)

    # grid
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.4)
    # for refined control of log-scale tick marks
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax2.xaxis.set_major_locator(locmaj)
    ax2.xaxis.set_minor_locator(locmin)
    ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax2.yaxis.set_major_locator(locmaj)
    ax2.yaxis.set_minor_locator(locmin)
    ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
    # tick length
    ax2.tick_params('both', direction='in', top='on', right='on', length=10,
                   width=1, which='major', zorder=301)
    ax2.tick_params('both', direction='in', top='on', right='on', length=5,
                   width=1, which='minor', zorder=301)

    # ax2.set_title('$\log \delta$', fontsize=18 )
    ax2.set_xlabel(r"$v_{0}$ [km/s]", fontsize=18)
    ax2.set_ylabel(r"$rho_{0} \, \, [M_{\odot} \, kpc^{-3}]$", fontsize=18)

    # scatter
    sigma0_LoDens = SIDM_LoDens[1]
    rhodm0_LoDens = SIDM_LoDens[0]
    sigma0_HiDens = SIDM_HiDens[1]
    rhodm0_HiDens = SIDM_HiDens[0]

    ax2.scatter(sigma0_LoDens, rhodm0_LoDens, marker='*', s=size,
                facecolor='k', edgecolor='k', linewidth=edgewidth, rasterized=True)
    ax2.scatter(sigma0_HiDens, rhodm0_HiDens, marker='*', s=size,
                facecolor='r', edgecolor='r', linewidth=edgewidth, rasterized=True)
    # lines
    plt.axvline(x=0.5 * nu_at_r1, color='black', linestyle='--',
                label=f'0.5 * Velocity dispersion (nu) at r1: {"{:.0f}".format(0.5 * nu_at_r1)} [km/s].')
    plt.axvline(x=2 * nu_at_r1, color='black', linestyle='--',
                label=f'2 * Velocity dispersion (nu) at r1: {"{:.0f}".format(2 * nu_at_r1)} [km/s].')
    plt.axhline(y=rhoCDM1, color='black', linestyle='--',
                label=f'rho value at r1: {"{:.2E}".format(rhoCDM1)} [M_sun / kpc^3].')
    plt.axhline(y=rhoCDMRres, color='black', linestyle='--',
                label=f'rho value at spatial resolution (0.01 kpc): {"{:.2E}".format(rhoCDMRres)} [M_sun / kpc^3].')
    # annotation to lines
    ax2.text(0.5 * nu_at_r1, 2. * ax2.get_ylim()[0], r'$0.5\, v(r_{1})$', color='k', fontsize=16,
             ha='right', va='bottom', transform=ax2.transData, rotation=90)
    ax2.text(2 * nu_at_r1, 2. * ax2.get_ylim()[0], r'$2\, v(r_{1})}$', color='k', fontsize=16,
             ha='left', va='bottom', transform=ax2.transData, rotation=90)

    ax2.text(2. * ax2.get_xlim()[0], rhoCDMRres, r'$\rho(r_{res})$', color='k', fontsize=16,
             ha='right', va='bottom', transform=ax2.transData, rotation=0)
    ax2.text(2. * ax2.get_xlim()[0], rhoCDM1, r'$\rho(r_\mathrm{1})$', color='k', fontsize=16,
             ha='right', va='bottom', transform=ax2.transData, rotation=0)

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(CS, ticks=mticker.MaxNLocator(6))
    cbar.ax.set_ylabel('$\log \delta$', fontsize=18)

    # ax2.legend(loc='lower left')
