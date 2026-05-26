"""
This file will be containing some functions, which has been intended to create numerous of useful plots.
    code (simulation) -> corresponding to the mathematica file.
    model (fitting) -> corresponding to the paper 2205.02957.
"""
import matplotlib.pyplot as plt
import numpy as np
import math
# --- import from files
from createHistograms import relative_difference

# ------------------------------------------------ DRAW ------------------------------------------------ #
def onePlotDensityProfile(_list_cof,
                          _time_data, _radius_data, _rho_data,
                          _model_rho,
                          _selected_period,
                          _plot_name):
    """
    :param _list_cof: (list): list with values of parameters, characteristic for density profile of model (from paiper
                              2205.02957). More details can be found in `modelGravothermalDensities.py`
    :param _time_data: (double): list containing the all times of evolution of selected galaxy. This data coming from
                                 the .txt file, which is created in mathematica.
    :param _radius_data: (double): list containing the all values of radius of selected galaxy. This data coming from
                                   the .txt file, which is created in mathematica.
    :param _rho_data: (double): the density of DM taking from .txt file.
    :param _model_rho: (double): the density of dark matter in selected radius and time \rho (r,t). Returning for all
                                 times and all radiuses. This density has been computing according to paper 2205.02957.
    :param _selected_period: (int): the number of selected period, which you want see.
    :param _plot_name: (str): the full name of the plot (containing the path).
    :return: Nothing. This functions has been intended to save/create one plot with specific number picture in
             the name --- useful if you have seen the density profiles on specific one time step or sth.
    """
    # --- set data
    # different values of time. In other words number of steps in time.
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1

    # we want to present the part of whole list in our plot (one period in each plot)
    left_arg = _selected_period * len_t
    right_arg = (_selected_period + 1) * len_t
    # data, which occur in our plot
    radius_points = np.log10(_radius_data[left_arg:right_arg])
    rho_code_points = np.log10(_rho_data[left_arg:right_arg])
    rho_model_points = np.log10(_model_rho[left_arg:right_arg])
    # time point to get the time in SI - in the title of plot
    time_point = _time_data[_selected_period * len_t + 1]

    # --- Create plot
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.plot(radius_points, rho_code_points, label='Mathematica, time step = ' + str(_selected_period))
    ax.plot(radius_points, rho_model_points, label='Paper, time step = ' + str(_selected_period))
    # parameters
    if _list_cof[0] is True:
        r_core = _list_cof[1]
        log_r_core = math.log10(r_core)

        plt.axvline(x=log_r_core, color='black', linestyle='dotted',
                    label='radius core for early time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
    else:
        r_core = _list_cof[1]
        log_r_core = math.log10(r_core)
        r_out = _list_cof[2]
        log_r_out = math.log10(r_out)
        rho_core = _list_cof[3]
        log_rho_core = math.log10(rho_core)

        plt.axvline(x=log_r_core, color='black', linestyle='dotted',
                    label='radius core for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
        plt.axvline(x=log_r_out, color='grey', linestyle='dashed',
                    label='radius out for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_out)} [kpc].')
        plt.axhline(y=log_rho_core, color='violet', linestyle='dashdot',
                    label='rho core for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(rho_core)} [].')
    plt.ylim(-6.2, 2.2)
    plt.ylabel(r'$\log\rho(r)$', fontsize=18)
    plt.xlabel(r'$\log(r)$', fontsize=18)
    plt.title('Density profile on:' + "%10.2e" % (10 ** time_point) + ' Gyr', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    plt.savefig(_plot_name + '.png', dpi=300)
    # we have to close figure.
    plt.close(fig)

def onePlotDensityProfileBackend(fig, ax, _list_cof,
                                 _time_data, _radius_data, _rho_data,
                                 _model_rho,
                                 _selected_period,
                                 _plot_name):
    """Plotting backend for `onePlotDensityProfile`"""

    # --- set data
    # different values of time. In other words number of steps in time.
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1

    # we want to present the part of whole list in our plot (one period in each plot)
    left_arg = _selected_period * len_t
    right_arg = (_selected_period + 1) * len_t
    # data, which occur in our plot
    radius_points = np.log10(_radius_data[left_arg:right_arg])
    rho_code_points = np.log10(_rho_data[left_arg:right_arg])
    rho_model_points = np.log10(_model_rho[left_arg:right_arg])
    # time point to get the time in SI - in the title of plot
    time_point = _time_data[_selected_period * len_t + 1]

    # --- Create plot
    ax.plot(radius_points, rho_code_points, label='Mathematica, time step = ' + str(_selected_period))
    ax.plot(radius_points, rho_model_points, label='Paper, time step = ' + str(_selected_period))
    # parameters
    if _list_cof[0] is True:
        r_core = _list_cof[1]
        log_r_core = math.log10(r_core)

        plt.axvline(x=log_r_core, color='black', linestyle='dotted',
                    label='radius core for early time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
    else:
        r_core = _list_cof[1]
        log_r_core = math.log10(r_core)
        r_out = _list_cof[2]
        log_r_out = math.log10(r_out)
        rho_core = _list_cof[3]
        log_rho_core = math.log10(rho_core)

        plt.axvline(x=log_r_core, color='black', linestyle='dotted',
                    label='radius core for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
        plt.axvline(x=log_r_out, color='grey', linestyle='dashed',
                    label='radius out for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_out)} [kpc].')
        plt.axhline(y=log_rho_core, color='violet', linestyle='dashdot',
                    label='rho core for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(rho_core)} [].')
    plt.ylim(-6.2, 2.2)
    plt.ylabel(r'$\log\rho(r)$', fontsize=18)
    plt.xlabel(r'$\log(r)$', fontsize=18)
    plt.title('Density profile on:' + "%10.2e" % (10 ** time_point) + ' Gyr', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()

def comparisonTwoDensityProfiles(_time_data, _radius_data, _rho_data,
                                 _model_rho,
                                 _period_one, _period_two, plot_show=False):
    """
    :param _time_data: (double): list containing the all times of evolution of selected galaxy. This data coming from
                                 the .txt file, which is created in mathematica.
    :param _radius_data: (double): list containing the all values of radius of selected galaxy. This data coming from
                                   the .txt file, which is created in mathematica.
    :param _rho_data: (double): the density of DM taking from .txt file.
    :param _model_rho: (double): the density of dark matter in selected radius and time \rho (r,t). Returning for all
                                 times and all radiuses. This density has been computing according to paper 2205.02957.
    :param _period_one: first selected period to compare.
    :param _period_two: second selecetd period to compare.
    :param plot_show: did you want to see?
    :return: Show the comparison chart.
    """
    # different values of time. In other words number of steps in time.
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1

    # we want to present only a one time step
    left_arg_one = _period_one * len_t
    right_arg_one = (_period_one + 1) * len_t
    left_arg_two = _period_two * len_t
    right_arg_two = (_period_two + 1) * len_t
    # data, which is connecting with _period_one
    radius_points_one = np.log10(_radius_data[left_arg_one:right_arg_one])
    rho_code_points_one = np.log10(_rho_data[left_arg_one:right_arg_one])
    rho_model_points_one = np.log10(_model_rho[left_arg_one:right_arg_one])
    # data, which is connecting with _period_two
    radius_points_two = np.log10(_radius_data[left_arg_two:right_arg_two])
    rho_code_points_two = np.log10(_rho_data[left_arg_two:right_arg_two])
    rho_model_points_two = np.log10(_model_rho[left_arg_two:right_arg_two])

    # Create plot
    fig, ax = plt.subplots(figsize=(14.0, 14.0))
    ax.plot(radius_points_one, rho_code_points_one, '.',
            label='Our mathematica simulation, time step = ' + str(_period_one))
    ax.plot(radius_points_one, rho_model_points_one, '.',
            label='Fitted equation from paper 2205.02957 = ' + str(_period_one))
    ax.plot(radius_points_two, rho_code_points_two, '.',
            label='Our mathematica simulation, time step = ' + str(_period_two))
    ax.plot(radius_points_two, rho_model_points_two, '.',
            label='Fitted equation from paper 2205.02957, time step = ' + str(_period_two))
    plt.ylabel(r'$\log\rho(r)$', fontsize=18)
    plt.xlabel(r'$\log(r)$', fontsize=18)
    plt.title('Comparison density profiles', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # we have to close figure.
    if plot_show is False:
        plt.show()

def differenceDensityProfiles(_list_cof,
                              _time_data, _radius_data, _rho_data,
                              _model_rho,
                              _selected_period,
                              _plot_name):
    """
    See above.
    :return: Nothing. This functions has been intended to save/create one plot with specific number picture in
             the name --- which presents how differ simulation and fitting density profiles.
    """
    # --- set data
    # different values of time. In other words number of steps in time.
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1

    # we want to present the part of whole list in our plot (one period in each plot)
    left_arg = _selected_period * len_t
    right_arg = (_selected_period + 1) * len_t
    # data, which occur in our plot
    radius_points = np.log10(_radius_data[left_arg:right_arg])
    rho_code_points = _rho_data[left_arg:right_arg]
    rho_model_points = _model_rho[left_arg:right_arg]
    rel_diff = relative_difference(rho_code_points, rho_model_points) * 100
    # time point to get the time in SI - in the title of plot
    time_point = _time_data[_selected_period * len_t + 1]

    # --- Create plot
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.plot(radius_points, rel_diff, '.', label='Relative difference, time step = ' + str(_selected_period))
    # parameters
    if _list_cof[0] is True:
        r_core = _list_cof[1]
        log_r_core = math.log10(r_core)

        plt.axvline(x=log_r_core, color='black', linestyle='dotted',
                    label='radius core for early time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
    else:
        r_core = _list_cof[1]
        log_r_core = math.log10(r_core)
        r_out = _list_cof[2]
        log_r_out = math.log10(r_out)

        plt.axvline(x=log_r_core, color='black', linestyle='dotted',
                    label='radius core for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
        plt.axvline(x=log_r_out, color='grey', linestyle='dashed',
                    label='radius out for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_out)} [kpc].')
    plt.ylabel(r'$\frac{\rho_{sim}(r)-\rho_{fit}(r)}{\rho_{fit}(r)}\ (\%)$', fontsize=18)
    plt.xlabel(r'$\log(r)$', fontsize=18)
    plt.title('Relative difference on:' + "%10.2e" % (10 ** time_point) + ' Gyr', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    plt.savefig(_plot_name + '.png', dpi=300)
    # we have to close figure.
    plt.close(fig)

def differenceDensityProfilesBackend(fig, ax, _list_cof,
                                     _time_data, _radius_data, _rho_data,
                                     _model_rho,
                                     _selected_period,
                                     _plot_name):
    """Plotting backend for `differenceDensityProfiles`"""
    # --- set data
    # different values of time. In other words number of steps in time.
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1

    # we want to present the part of whole list in our plot (one period in each plot)
    left_arg = _selected_period * len_t
    right_arg = (_selected_period + 1) * len_t
    # data, which occur in our plot
    radius_points = np.log10(_radius_data[left_arg:right_arg])
    rho_code_points = _rho_data[left_arg:right_arg]
    rho_model_points = _model_rho[left_arg:right_arg]
    rel_diff = relative_difference(rho_code_points, rho_model_points) * 100
    # time point to get the time in SI - in the title of plot
    time_point = _time_data[_selected_period * len_t + 1]

    # Create plot
    ax.plot(radius_points, rel_diff, '.', label='Relative difference, time step = ' + str(_selected_period))
    # parameters
    if _list_cof[0] is True:
        r_core = _list_cof[1]
        log_r_core = math.log10(r_core)

        plt.axvline(x=log_r_core, color='black', linestyle='dotted',
                    label='radius core for early time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
    else:
        r_core = _list_cof[1]
        log_r_core = math.log10(r_core)
        r_out = _list_cof[2]
        log_r_out = math.log10(r_out)

        plt.axvline(x=log_r_core, color='black', linestyle='dotted',
                    label='radius core for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_core)} [kpc].')
        plt.axvline(x=log_r_out, color='grey', linestyle='dashed',
                    label='radius out for late time,' + "\n" +
                          f'ocured in: {"{:.2f}".format(r_out)} [kpc].')
    plt.ylabel(r'$\frac{\rho_{sim}(r)-\rho_{fit}(r)}{\rho_{fit}(r)}\ (\%)$', fontsize=18)
    plt.xlabel(r'$\log(r)$', fontsize=18)
    plt.title('Relative difference on:' + "%10.2e" % (10 ** time_point) + ' Gyr', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()

def comparisonThreeDifferenceProfiles(_time_data, _radius_data, _rho_data,
                                      _model_rho,
                                      _period_1, _period_2, _period_3, plot_show=False):
    # different values of time. In other words number of steps in time.
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1

    # we want to present only a one time step
    left_arg_1 = _period_1 * len_t
    right_arg_1 = (_period_1 + 1) * len_t
    left_arg_2 = _period_2 * len_t
    right_arg_2 = (_period_2 + 1) * len_t
    left_arg_3 = _period_3 * len_t
    right_arg_3 = (_period_3 + 1) * len_t

    """CREATE PLOT"""
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    # first time period
    r_period_1 = np.log10(_radius_data[left_arg_1:right_arg_1])
    rho_code_period_1 = _rho_data[left_arg_1:right_arg_1]
    rho_paper_period_1 = _model_rho[left_arg_1:right_arg_1]
    rel_diff_1 = relative_difference(rho_code_period_1, rho_paper_period_1) * 100
    ax.plot(r_period_1, rel_diff_1, '.', label=f'Relative difference, time step = {_period_1}')

    # second time period
    r_period_2 = np.log10(_radius_data[left_arg_2:right_arg_2])
    rho_code_period_2 = _rho_data[left_arg_2:right_arg_2]
    rho_paper_period_2 = _model_rho[left_arg_2:right_arg_2]
    rel_diff_2 = relative_difference(rho_code_period_2, rho_paper_period_2) * 100
    ax.plot(r_period_2, rel_diff_2, '.', label=f'Relative difference, time step = {_period_2}')

    # third time period
    r_period_3 = np.log10(_radius_data[left_arg_3:right_arg_3])
    rho_code_period_3 = _rho_data[left_arg_3:right_arg_3]
    rho_paper_period_3 = _model_rho[left_arg_3:right_arg_3]
    rel_diff_3 = relative_difference(rho_code_period_3, rho_paper_period_3) * 100
    ax.plot(r_period_3, rel_diff_3, '.', label=f'Relative difference, time step = {_period_3}')

    plt.ylabel(r'$\frac{\rho_{sim}(r)-\rho_{fit}(r)}{\rho_{fit}(r)}\ (\%)$', fontsize=18)
    plt.xlabel(r'$\log(r)$', fontsize=18)
    plt.title('Relative difference', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    if plot_show is False:
        plt.show()


def numeratePlots(_path_to_file, _file_name, _num_files):
    """
    :param _path_to_file: path to the file (future localization).
    :param _file_name: file name.
    :param _num_files: number of all given charts. E.g. how many pictures of evolution (density profile) you have.
    :return: list containing the full name of the plot.
    """
    # Now I create the names for all saves files
    all_names = []

    for i in range(1, (_num_files + 1)):
        str_zeros = ""
        zeros_to_write = int(math.log10(_num_files)) - int(math.log10(i))

        for j in range(0, zeros_to_write):
            str_zeros = str_zeros + '0'
        name = _path_to_file + '/' + _file_name + str_zeros + str(i)
        all_names.append(name)


    return all_names
