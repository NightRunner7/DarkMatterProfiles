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

def plotCircularVelocity(NFW_profile, SIDM_LoDens, SIDM_HiDens, val_r1):
    """
    Draw the plot, which presents how looks like the CDM-only circular velocity, Isothermal circular velocity
    for low and high density (we will have two solution for Isothermal and first is physical and second
    is unphysical).

    where

        NFW_profile: class, which contains all methods necessary to deal with profile CDM-only.
        SIDM_LoDens: list contains all necessary data, which is corresponding to physical Isothermal
        solution.
        SIDM_LoDens: list contains all necessary data, which is corresponding to unphysical Isothermal
        solution.
        val_r1: the distance where we're switching formula from Isothermal to NFW. [kpc]
    """
    plot_r = np.logspace(-3.0, 2.0, num=300)  # [log10(kpc)]
    # ------------------------------ NFW PROFILE ------------------------------ #
    # NFW profile: circular velocity
    Vcirc_NFW = NFW_profile.Vcirc(plot_r)  # [kpc/Gyr]

    Vcirc_NFW_SI = []
    for i in range(0, len(plot_r)):
        Vcirc_SI = Vcirc_NFW[i] * cfg.kpc_SI / cfg.Gyr  # [m/s]
        # append
        Vcirc_NFW_SI.append(Vcirc_SI * 10**(-3))  # [km/s]

    # ------------------------------ ISOTHERMAL ------------------------------ #
    # Isothermal - low dense: circular velocity
    r_LoDens = SIDM_LoDens[4]  # [kpc]
    Vcirc_LoDens_starUnit = SIDM_LoDens[3]  # kpc/Gyr
    Vcirc_LoDens = Vcirc_LoDens_starUnit * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    # Isothermal - high dense: circular velocity
    r_HiDens = SIDM_HiDens[4]
    Vcirc_HiDens_starUnit = SIDM_HiDens[3]
    Vcirc_HiDens = Vcirc_HiDens_starUnit * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]

    # ------------------------------ LINES ------------------------------ #
    r_s = NFW_profile.r_s  # [kpc]
    Rres = 0.01  # [kpc]
    r_1 = val_r1  # [kpc]

    # ------------------------------ PLOTTING ------------------------------ #
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # describe
    plt.ylim(10 ** 0, 5 * 10 ** 2)
    plt.ylabel(r"$V_{circular}$ [km / s]", fontsize=18)
    plt.xlabel(r"$r$ [kpc]", fontsize=18)
    ax.tick_params(labelsize=14)
    # data
    ax.plot(plot_r, Vcirc_NFW_SI, '--',label="NFW: CDM-only")
    ax.plot(r_LoDens, Vcirc_LoDens,label="SIDM: low density")
    ax.plot(r_HiDens, Vcirc_HiDens,label="SIDM: high density")

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
