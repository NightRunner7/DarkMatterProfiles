import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
# take files from parent dir
sys.path.insert(0, parentdir)
import config as cfg

def plotVelocityDisparsion(NFW_profile, SIDM_LoDens, SIDM_HiDens, val_r1):
    """
    Draw the plot, which presents how looks like the CDM-only velocity dispersion, Isothermal velocity dispersion
    for low and high density (we will have two solution for Isothermal and first is physical and second
    is unphysical).

    where

        NFW_profile: class, which contains all methods necessary to deal with profile CDM-only.
        SIDM_LoDens: list contains all necessary data, which is corresponding to physical Isothermal
        solution.
        SIDM_LoDens: list contains all necessary data, which is corresponding to unphysical Isothermal
        solution.
        val_r1: the distance where we're switching formula from Isothermal to NFW. [kpc]

    -----------
    One important remark: this plot shows only how look like the velocity dispersion, so will
    apper the discontinuity. To shows the more proper plots go further.
    """
    plot_r = np.logspace(-3.0, 2.0, num=300)
    # ------------------------------ NFW PROFILE ------------------------------ #
    # NFW profile: dispersion velocity
    Vdis_NFW = NFW_profile.sigma_accurate(plot_r)  # [kpc/Gyr]

    Vdis_NFW_km_s = []
    for i in range(0, len(plot_r)):
        Vdis_SI = Vdis_NFW[i] * cfg.kpc_SI / cfg.Gyr  # [m/s]
        # append
        Vdis_NFW_km_s.append(Vdis_SI * 10**(-3))  # [km/s]

    # ------------------------------ ISOTHERMAL ------------------------------ #
    # Isothermal - low dense: dispersion velocity
    r_LoDens = SIDM_LoDens[4]  # [kpc]
    sigma0_LoDens = SIDM_LoDens[1]  # [kpc/Gyr]
    sigma0_LoDens_km_s = sigma0_LoDens * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma0_LoDens_list = [sigma0_LoDens_km_s] * len(r_LoDens)
    # Isothermal - high dense dense: dispersion velocity
    r_HiDens = SIDM_HiDens[4]  # [kpc]
    sigma0_HiDens = SIDM_HiDens[1]  # [kpc/Gyr]
    sigma0_HiDens_km_s = sigma0_HiDens * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma0_HiDens_list = [sigma0_HiDens_km_s] * len(r_HiDens)

    # ------------------------------ LINES ------------------------------ #
    r_s = NFW_profile.r_s  # [kpc]
    Rres = 0.01  # [kpc]
    r_1 = val_r1  # [kpc]

    # ------------------------------ PLOTTING ------------------------------ #
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    # log scale
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # describe NFW profile
    plt.ylim(10 ** 1, 7 * 10 ** 1)
    plt.ylabel(r"$\nu$ [km / s]", fontsize=18)
    plt.xlabel(r"$r$ [kpc]", fontsize=18)
    plt.title("Dispersion velocity", fontsize=18)
    ax.tick_params(labelsize=14)
    # Data
    ax.plot(plot_r, Vdis_NFW_km_s, '--', label="NFW: CDM-only")
    ax.plot(r_LoDens, sigma0_LoDens_list, label="SIDM: low density")
    ax.plot(r_HiDens, sigma0_HiDens_list, label="SIDM: high density")

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
    # ax.yaxis.set_major_locator(locmaj)
    # ax.yaxis.set_minor_locator(locmin)
    # ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # # tick length
    # ax.tick_params('both', direction='in', top='on', right='on', length=10,
    #                width=1, which='major', zorder=301)
    # ax.tick_params('both', direction='in', top='on', right='on', length=5,
    #                width=1, which='minor', zorder=301)

    # lines
    plt.axvline(x=r_s, color='black', linestyle='dashdot',
                label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
    plt.axvline(x=r_1, color='grey', linestyle='dotted',
                label=f'r_1 = {"{:.3f}".format(r_1)} [kpc].')
    plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9,
                label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
    # annotation to lines
    ax.text(r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
            ha='right', va='top', transform=ax.transData, rotation=90)
    ax.text(r_1, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
            ha='right', va='top', transform=ax.transData, rotation=90)
    ax.text(Rres, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
            ha='left', va='top', transform=ax.transData, rotation=90)

    ax.legend()
    plt.show()
