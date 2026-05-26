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

def plotEvolutionRhoISO(parameters, time_tilda, rho0_LoDens_tilda, rho0_HiDens_tilda,
                        savePlot=True, path_to_directory="."):
    """
    This function is created a plot, which presents how look like the central density
    (the core density) in time. We can see that using the Isothermal simulation, firstly
    the core density is decreasing and at some point starts to rising. This is how behave
    the `true central density` (because the unphysical core density should decrease for
    whole evolution).

    At some point the physical value meets with the unphysical - in that time the isothermal
    simulation fails to predict how behave the evolution of core density of DM. Here we're
    making plot, which shows us how this evolution look like and where the merging between
    two results (physical and unphysical) meet.

    -------------
    where:
        parameters: list contains all necessary parameters, which we want top have on plot.
            parameters['Mvir']: viral mass [M_sun/kpc^3]
            parameters['c_const']: concentration of DM [dimensionless]
            parameters['cross_section']: annihilation cross-section [cm^2/g]
            parameters['t_tilda']: time when we observed merging [dimensionless]
            parameters['t']: time when we observed merging [Gyr]
            parameters['rho0_LoDens_tilda']: the value of density when we observed merging [dimensionless]
        time_tilda: list contains points in time for which we find values of central density
            in the physical (low dense) and unphysical (high dense) cases.
            (array) [dimensionless]
        rho0_LoDens_tilda: central density (core density) of DM in physical case (low dense).
            (array) [dimensionless]
        rho0_LoDens_tilda: central density (core density) of DM in unphysical case (high dense).
            (array) [dimensionless]

        path_to_directory: a path to directory were you want to store your plot.
            (string)
    -------------
    One important remark: some arrays / parameters we can take as arguments, have very
    specific units (for e.g. [density] = [dimensionless]). This is reduced units. To more
    details see section: `DIMENSIONLESS UNITS` in file `auxiliaryFunctions.py` .
    """
    # ------------------------- PARAMETERS ------------------------- #
    M_vir = parameters['Mvir']
    const_c = parameters['c_const']
    sigma_m = parameters['cross_section']
    tmerged = parameters['t_tilda']
    tmerged_Gyr = parameters['t']
    rho0_tilda_tmerge = parameters['rho0_LoDens_tilda']
    # ------------------------- PLOT ------------------------- #
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    # solutions: low and high density
    ax.plot(time_tilda, rho0_LoDens_tilda, label="SIDM: low density", zorder=1)
    ax.plot(time_tilda, rho0_HiDens_tilda, label="SIDM: high density", zorder=2)
    # merging
    rounded_tmerged = cfg.rounded_number(tmerged, 5)
    rounded_tmerged_Gyr = cfg.rounded_number(tmerged_Gyr, 5)
    ax.scatter(tmerged, rho0_tilda_tmerge, marker='o', s=50.0, zorder=3,
               label=f'merging two solutions at: {rounded_tmerged} ,' + '\n' + f'which refers to {rounded_tmerged_Gyr} [Gyr].',
               facecolor='grey', edgecolor='black', linewidth=1.0, rasterized=True)

    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(10 ** 0, 2 * 10 ** 3)
    plt.xlim(10 ** (-1), 2 * 10 ** 3)
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
    if savePlot is True:
        # file name
        format_sigma_m = f'{"{:.1f}".format(sigma_m)}'
        format_c = f'{"{:.3f}".format(const_c)}'
        format_log10mass = f'{"{:.5f}".format(np.log10(M_vir))}'
        file_name = f'IsothermalInTime_Mvir{format_log10mass}_c{format_c}_sigmamx{format_sigma_m}.png'
        # localization
        file_path = path_to_directory + '/' + file_name

        plt.savefig(file_path, dpi=300)
        # ---we have to close figure.
        plt.close(fig)
    else:
        plt.show()
