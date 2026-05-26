"""
Here we have code to draw the velocity dispersion at fixed time.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
from NFWProfile import NFWProfile, r1  # CDM profile (halo)
import auxiliaryFunctions as aux

# --------------------------------------- SERIES OF PLOTS --------------------------------------- #
def plot_series_nu_regime_I(isothermal_class, gravothermal_class, steps_list, path_dir="",
                            xmin=7 * 10 ** (-4), xmax=2 * 10 ** 2, ymin=1.8, ymax=7.9):
    """
    Draw the plot, which show how look the velocity dispersion profile for series of time.
    Regime I, corresponding to the situation when the R-procedure simulation is valid. So,
    part of data should be obtained from that simulation.

    :param isothermal_class: object, which stored accurate data. More details you can find
        at `RprocedureData.py`. Those data is corresponding to the Isothermal profile
        during Rprocedure simulation.
        (object)
    :param gravothermal_class: object, which stored accurate data. More details you can find
        at `GravothermalData.py`. Those data is corresponding to the Gravothermal simulation.
        (object)
    :param steps_list: list of time steps (snapshots) of the velocity dispersion, which we want to have.
    :param path_dir: path to the main directory, where we want stored those plots.
    :param xmin: the minimum value of x-axis [r_s] (float)
    :param xmax: the maximum value of x-axis [r_s] (float)
    :param ymin: the minimum value of y-axis [r_s/Gyr] (float)
    :param ymax: the maximum value of y-axis [r_s/Gyr] (float)
    ------
    Comparison between Isothermal and Gravothermal data.
    """
    # --- create director to stored plots
    aux.clear_and_make_directory(path_dir)  # make directory
    # --- take names from class
    iso_name = isothermal_class.name
    gravo_name = gravothermal_class.name

    for i in range(0, len(steps_list)):
        # --- get proper data
        step = steps_list[i]
        # profile from Rprocedure
        iso_profile = isothermal_class.return_data_at_fixed_time(step, time_step_bool=True)
        # profile from gravothermal (NFW)
        Gravo_profile = gravothermal_class.return_data_at_fixed_time(iso_profile["time"], time_step_bool=False)

        # --- find propre r1
        Mvir, sigma_m, const_c = isothermal_class.return_basic_parameters()
        NFW_profile_class = NFWProfile(Mvir, const_c)  # NFW profile
        r_1 = r1(NFW_profile_class, sigmamx=sigma_m, tage=iso_profile["time"])  # r1 [kpc]

        # --- Different lines
        Rres = 0.01  # [kpc]
        r_s = isothermal_class.parameters["r_s"]  # [kpc]

        # --- plotting
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        # log scale
        ax.set_xscale('log')
        # describe NFW profile
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)
        plt.ylabel(r"$\nu$ [$r_{s}$ / Gyr]", fontsize=18)
        plt.xlabel(r"$r$ [$r_{s}$]", fontsize=18)
        plt.title(f"Velocity dispersion: {iso_name} (R-procedure) vs {gravo_name} (gravothermal)", fontsize=16)
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

        # --- DATA
        # profile from Rprocedure (Isothermal)
        ax.plot(iso_profile['r'], iso_profile['velDis'],
                label=f'{iso_name} (Rprocedure), at {"{:.2f}".format(iso_profile["time"])} [Gyr]')
        # profile from gravothermal (NFW)
        ax.plot(Gravo_profile['r'], Gravo_profile['velDis'],
                label=f'{gravo_name} (gravothermal), at {"{:.2f}".format(Gravo_profile["time"])} [Gyr]')

        # lines
        plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot')
        plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted')
        plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9)

        # annotation to lines
        ax.text(r_s / r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
                ha='right', va='top', transform=ax.transData, rotation=90)
        ax.text(r_1 / r_s, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
                ha='right', va='top', transform=ax.transData, rotation=90)
        ax.text(Rres / r_s, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
                ha='left', va='top', transform=ax.transData, rotation=90)
        # --- LEGEND AND SAVE
        ax.legend(fontsize=14)
        plt.savefig('./' + path_dir + f'/Velocity-dispersion-at-step-{step:03d}.png', dpi=300)
        plt.close(fig)


def plot_series_nu_regime_II(isothermal_class, gravothermal_class, steps_list, path_dir="",
                             xmin=7 * 10 ** (-4), xmax=2 * 10 ** 2, ymin=1.8, ymax=7.9):
    """
    Draw the plot, which show how look the velocity dispersion profile for series of time.
    Regime II, corresponding to the situation when the R-procedure simulation is failed.
    So, all data is coming from gravothermal simulation.

    :param isothermal_class: object, which stored accurate data. More details you can find
        at `GravothermalData.py`. Those data is corresponding to the Isothermal profile
        during gravothermal simulation.
        (object)
    :param gravothermal_class: object, which stored accurate data. More details you can find
        at `GravothermalData.py`. Those data is corresponding to the Gravothermal simulation.
        (object)
    :param steps_list: list of time steps (snapshots) of the velocity dispersion, which we want to have.
    :param path_dir: path to the main directory, where we want stored those plots.
    :param xmin: the minimum value of x-axis [r_s] (float)
    :param xmax: the maximum value of x-axis [r_s] (float)
    :param ymin: the minimum value of y-axis [r_s/Gyr] (float)
    :param ymax: the maximum value of y-axis [r_s/Gyr] (float)
    ------
    Comparison between Isothermal and Gravothermal data.
    """
    # --- create director to stored plots
    aux.clear_and_make_directory(path_dir)  # make directory
    # --- take names from class
    iso_name = isothermal_class.name
    gravo_name = gravothermal_class.name

    for i in range(0, len(steps_list)):
        # --- get proper data
        step = steps_list[i]
        # profile from Rprocedure
        iso_profile = isothermal_class.return_data_at_fixed_time(step, time_step_bool=True)
        # profile from gravothermal (NFW)
        Gravo_profile = gravothermal_class.return_data_at_fixed_time(iso_profile["time"], time_step_bool=False)

        # --- find propre r1
        Mvir, sigma_m, const_c = isothermal_class.return_basic_parameters()
        NFW_profile_class = NFWProfile(Mvir, const_c)  # NFW profile
        r_1 = r1(NFW_profile_class, sigmamx=sigma_m, tage=iso_profile["time"])  # r1 [kpc]

        # --- Different lines
        Rres = 0.01  # [kpc]
        r_s = isothermal_class.parameters["r_s"]  # [kpc]

        # --- plotting
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        # log scale
        ax.set_xscale('log')
        # describe NFW profile
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)
        plt.ylabel(r"$\nu$ [$r_{s}$ / Gyr]", fontsize=18)
        plt.xlabel(r"$r$ [$r_{s}$]", fontsize=18)
        plt.title(f"Velocity dispersion: {iso_name} (gravothermal) vs {gravo_name} (gravothermal)", fontsize=16)
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

        # --- DATA
        # profile from Rprocedure (Isothermal)
        ax.plot(iso_profile['r'], iso_profile['velDis'],
                label=f'{iso_name} (gravothermal), at {"{:.2f}".format(iso_profile["time"])} [Gyr]')
        # profile from gravothermal (NFW)
        ax.plot(Gravo_profile['r'], Gravo_profile['velDis'],
                label=f'{gravo_name} (gravothermal), at {"{:.2f}".format(Gravo_profile["time"])} [Gyr]')

        # lines
        plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot')
        plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted')
        plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9)

        # annotation to lines
        ax.text(r_s / r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
                ha='right', va='top', transform=ax.transData, rotation=90)
        ax.text(r_1 / r_s, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
                ha='right', va='top', transform=ax.transData, rotation=90)
        ax.text(Rres / r_s, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
                ha='left', va='top', transform=ax.transData, rotation=90)
        # --- LEGEND AND SAVE
        ax.legend(fontsize=14)
        plt.savefig('./' + path_dir + f'/Velocity-dispersion-at-step-{step:03d}.png', dpi=300)
        plt.close(fig)


def plot_series_nu_regime_II_ISO(isothermal_class, steps_list, path_dir="",
                                 xmin=7 * 10 ** (-4), xmax=2 * 10 ** 2, ymin=1.8, ymax=7.9):
    """
    Draw the plot, which show how look the velocity dispersion profile for series of time.
    Regime II, corresponding to the situation when the R-procedure simulation is failed.
    So, all data is coming from gravothermal simulation.

    :param isothermal_class: object, which stored accurate data. More details you can find
        at `GravothermalData.py`. Those data is corresponding to the Isothermal profile
        during gravothermal simulation.
        (object)
    :param steps_list: list of time steps (snapshots) of the velocity dispersion, which we want to have.
    :param path_dir: path to the main directory, where we want stored those plots.
    :param xmin: the minimum value of x-axis [r_s] (float)
    :param xmax: the maximum value of x-axis [r_s] (float)
    :param ymin: the minimum value of y-axis [r_s/Gyr] (float)
    :param ymax: the maximum value of y-axis [r_s/Gyr] (float)
    ------
    Just data from Isothermal.
    """
    # --- create director to stored plots
    aux.clear_and_make_directory(path_dir)  # make directory
    # --- take names from class
    iso_name = isothermal_class.name

    for i in range(0, len(steps_list)):
        # --- get proper data
        step = steps_list[i]
        # profile from Rprocedure
        iso_profile = isothermal_class.return_data_at_fixed_time(step, time_step_bool=True)

        # --- find propre r1
        Mvir, sigma_m, const_c = isothermal_class.return_basic_parameters()
        NFW_profile_class = NFWProfile(Mvir, const_c)  # NFW profile
        r_1 = r1(NFW_profile_class, sigmamx=sigma_m, tage=iso_profile["time"])  # r1 [kpc]

        # --- Different lines
        Rres = 0.01  # [kpc]
        r_s = isothermal_class.parameters["r_s"]  # [kpc]

        # --- plotting
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        # log scale
        ax.set_xscale('log')
        # describe NFW profile
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)
        plt.ylabel(r"$\nu$ [$r_{s}$ / Gyr]", fontsize=18)
        plt.xlabel(r"$r$ [$r_{s}$]", fontsize=18)
        plt.title(f"Velocity dispersion: {iso_name} (gravothermal)", fontsize=16)
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

        # --- DATA
        # profile from Rprocedure (Isothermal)
        ax.plot(iso_profile['r'], iso_profile['velDis'],
                label=f'{iso_name} (gravothermal), at {"{:.2f}".format(iso_profile["time"])} [Gyr]')

        # lines
        plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot')
        plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted')
        plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9)

        # annotation to lines
        ax.text(r_s / r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
                ha='right', va='top', transform=ax.transData, rotation=90)
        ax.text(r_1 / r_s, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
                ha='right', va='top', transform=ax.transData, rotation=90)
        ax.text(Rres / r_s, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
                ha='left', va='top', transform=ax.transData, rotation=90)
        # --- LEGEND AND SAVE
        ax.legend(fontsize=14)
        plt.savefig('./' + path_dir + f'/Velocity-dispersion-at-step-{step:03d}.png', dpi=300)
        plt.close(fig)


def plot_series_nu_regime_II_NFW(isothermal_class, gravothermal_class, steps_list, path_dir="",
                                 xmin=7 * 10 ** (-4), xmax=2 * 10 ** 2, ymin=1.8, ymax=7.9):
    """
    Draw the plot, which show how look the velocity dispersion profile for series of time.
    Regime II, corresponding to the situation when the R-procedure simulation is failed.
    So, all data is coming from gravothermal simulation.

    :param isothermal_class: object, which stored accurate data. More details you can find
        at `GravothermalData.py`. Those data is corresponding to the Isothermal profile
        during gravothermal simulation.
        (object)
    :param gravothermal_class: object, which stored accurate data. More details you can find
        at `GravothermalData.py`. Those data is corresponding to the Gravothermal simulation.
        (object)
    :param steps_list: list of time steps (snapshots) of the velocity dispersion, which we want to have.
    :param path_dir: path to the main directory, where we want stored those plots.
    :param xmin: the minimum value of x-axis [r_s] (float)
    :param xmax: the maximum value of x-axis [r_s] (float)
    :param ymin: the minimum value of y-axis [r_s/Gyr] (float)
    :param ymax: the maximum value of y-axis [r_s/Gyr] (float)
    ------
    Just data from NFW.
    """
    # --- create director to stored plots
    aux.clear_and_make_directory(path_dir)  # make directory
    # --- take names from class
    iso_name = isothermal_class.name
    gravo_name = gravothermal_class.name

    for i in range(0, len(steps_list)):
        # --- get proper data
        step = steps_list[i]
        # profile from Rprocedure
        iso_profile = isothermal_class.return_data_at_fixed_time(step, time_step_bool=True)
        # profile from gravothermal (NFW)
        Gravo_profile = gravothermal_class.return_data_at_fixed_time(iso_profile["time"], time_step_bool=False)

        # --- find propre r1
        Mvir, sigma_m, const_c = isothermal_class.return_basic_parameters()
        NFW_profile_class = NFWProfile(Mvir, const_c)  # NFW profile
        r_1 = r1(NFW_profile_class, sigmamx=sigma_m, tage=iso_profile["time"])  # r1 [kpc]

        # --- Different lines
        Rres = 0.01  # [kpc]
        r_s = isothermal_class.parameters["r_s"]  # [kpc]

        # --- plotting
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        # log scale
        ax.set_xscale('log')
        # describe NFW profile
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)
        plt.ylabel(r"$\nu$ [$r_{s}$ / Gyr]", fontsize=18)
        plt.xlabel(r"$r$ [$r_{s}$]", fontsize=18)
        plt.title(f"Velocity dispersion: {gravo_name} (gravothermal)", fontsize=16)
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

        # --- DATA
        # profile from gravothermal (NFW)
        ax.plot(Gravo_profile['r'], Gravo_profile['velDis'],
                label=f'{gravo_name} (gravothermal), at {"{:.2f}".format(Gravo_profile["time"])} [Gyr]')

        # lines
        plt.axvline(x=r_s / r_s, color='black', linestyle='dashdot')
        plt.axvline(x=r_1 / r_s, color='grey', linestyle='dotted')
        plt.axvline(x=Rres / r_s, color='black', linestyle='dotted', alpha=0.9)

        # annotation to lines
        ax.text(r_s / r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
                ha='right', va='top', transform=ax.transData, rotation=90)
        ax.text(r_1 / r_s, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
                ha='right', va='top', transform=ax.transData, rotation=90)
        ax.text(Rres / r_s, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
                ha='left', va='top', transform=ax.transData, rotation=90)
        # --- LEGEND AND SAVE
        ax.legend(fontsize=14)
        plt.savefig('./' + path_dir + f'/Velocity-dispersion-at-step-{step:03d}.png', dpi=300)
        plt.close(fig)

# --------------------------------------- ONE PLOT --------------------------------------- #
