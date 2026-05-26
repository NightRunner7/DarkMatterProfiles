import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math


# ####################### IMPORTING FILES ########################
import os
import sys
# currentdir = os.getcwd()
# parentdir = os.path.dirname(currentdir)
maindir = "C:\\Users\\Krzysztof\\Documents\\GitHub\\DarkMatterProfiles"
sys.path.insert(0, maindir)
from Determining_the_duration_of_evolution import find_paper_parameters

# ####################### FUNCTION TO PLOT ########################

def onePlotDensityProfile(_list_cof,
                          _time_sim, _r_sim, _rho_sim,
                          _fit_rho,
                          _selected_period,
                          _plot_name):
    """
    :param _list_cof: (list): list with values of parameters, characteristic for density profile of model (from paiper
                              2205.02957). More details can be found in `modelGravothermalDensities.py`.
    :param _time_sim: (double): list containing the all times of evolution of selected galaxy. This data coming from
                                the .txt file, which is created in mathematica (simulation).
    :param _r_sim: (double): list containing the all values of radius of selected galaxy. This data coming from
                             the .txt file, which is created in mathematica (simulation).
    :param _rho_sim: (double): the density of DM taking from .txt file (simulation).
    :param _fit_rho: (double): the density of dark matter in selected radius and time \rho (r,t). Returning for all
                               times and all radiuses. This density has been computing according to paper 2205.02957.
    :param _selected_period: (int): the number of selected period, which you want see.
    :param _plot_name: (str): the full name of the plot (containing the path).
    :return: Nothing. This functions has been intended to save/create one plot with specific number picture in
             the name --- useful if you have seen the density profiles on specific one time step or sth.
    """
    # --- set data
    # different values of time. In other words number of steps in time.
    len_t = 0
    while _time_sim[len_t] == _time_sim[0]: len_t += 1

    # we want to present the part of whole list in our plot (one period in each plot)
    left_arg = _selected_period * len_t
    right_arg = (_selected_period + 1) * len_t
    # data, which occur in our plot
    radius_points = _r_sim[left_arg:right_arg]
    rho_code_points = _rho_sim[left_arg:right_arg]
    rho_model_points = _fit_rho[left_arg:right_arg]
    # time point to get the time in SI - in the title of plot
    time_point = _time_sim[_selected_period * len_t + 1]

    # --- Create plot
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    # set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # description
    plt.ylim(2 * 10 ** (-6), 2 * 10 ** (2))
    plt.ylabel(r'$\hat{\rho}$ [$\rho_{s}$]', fontsize=18)
    plt.xlabel(r'$\hat{r}$ [$r_s$]', fontsize=18)
    plt.title('Density profile on:' + "%10.2e" % (10 ** time_point) + ' Gyr', fontsize=18)
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
    description_fit_label = ''
    if _list_cof[0] is True:
        description_fit_label = 'Density profile (fitting early-time).'
    else:
        description_fit_label = 'Density profile (fitting late-time).'
    # ax.plot(radius_points, rho_code_points, label='Density profile (simulation), time step = ' + str(_selected_period))
    # ax.plot(radius_points, rho_model_points, label='Paper, time step = ' + str(_selected_period))
    ax.plot(radius_points, rho_code_points, label='Density profile (simulation).')
    ax.plot(radius_points, rho_model_points, label=description_fit_label)

    # parameters / lines
    if _list_cof[0] is True:
        r_core = _list_cof[1]

        # plt.axvline(x=r_core, color='black', linestyle = 'dotted',
        #             label='radius core for early time,' + "\n" +
        #                   f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
        plt.axvline(x=r_core, color='black', linestyle='dotted')
        # annotation to lines
        ax.text(r_core, 2. * ax.get_ylim()[0], r'$r_{core}$', color='black', fontsize=16,
                ha='right', va='bottom', transform=ax.transData, rotation=90)
    else:
        r_core = _list_cof[1]
        r_out = _list_cof[2]
        rho_core = _list_cof[3]

        # plt.axvline(x=r_core, color='black', linestyle = 'dotted',
        #             label='radius core for late time,' + "\n" +
        #                   f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
        # plt.axvline(x=r_out, color='grey', linestyle = 'dashed',
        #             label='radius out for late time,' + "\n" +
        #                   f'ocured in: {"{:.2f}".format(r_out)} [kpc].')
        # plt.axhline(y=rho_core, color='violet', linestyle = 'dashdot',
        #             label='rho core for late time,' + "\n" +
        #                   f'ocured in: {"{:.2f}".format(rho_core)} [].')
        plt.axvline(x=r_core, color='black', linestyle='dotted')
        plt.axvline(x=r_out, color='grey', linestyle='dashed')
        plt.axhline(y=rho_core, color='violet', linestyle='dashdot')
        # annotation to lines
        # ax.text(r_core, 2. * ax.get_ylim()[0], r'$r_{core} = %.2f$' % (r_core), color='black', fontsize=16,
        #         ha='right', va='bottom', transform=ax.transData, rotation=90)
        ax.text(r_core, 2. * ax.get_ylim()[0], r'$r_{core}$', color='black', fontsize=16,
                ha='right', va='bottom', transform=ax.transData, rotation=90)
        ax.text(r_out, 2. * ax.get_ylim()[0], r'$r_{out}$', color='grey', fontsize=16,
                ha='right', va='bottom', transform=ax.transData, rotation=90)
        ax.text(5. * ax.get_xlim()[0], rho_core, r'$\rho_{core}$', color='violet', fontsize=16,
                ha='right', va='bottom', transform=ax.transData, rotation=0)

    # ax.tick_params(axis='both', which='major', labelsize=12)
    # --- LEGEND AND SAVE
    ax.legend(fontsize=14)
    plt.savefig(_plot_name + '.png', dpi=300)
    # we have to close figure.
    plt.close(fig)
