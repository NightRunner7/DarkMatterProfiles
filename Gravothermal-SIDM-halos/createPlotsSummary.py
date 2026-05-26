"""
This file contains functions, which We have shown in the: "Summary of our knowledge" in main.py.
    code (simulation) -> corresponding to the mathematica file.
    model (fitting) -> corresponding to the paper 2205.02957.
"""

import matplotlib.pyplot as plt
from createHistograms import relative_difference, calculate_average, core_density, find_min_rho_core, calculate_rho_core

def maxDifferenceProfiles(_time_data, _rho_data,
                          _model_rho,
                          _path, _title_name,
                          _core_time_simulation, _core_time_fitting, _core_time_fitting_model, plot_show=False):
    """
    :param _time_data: (double): list containing the all times of evolution of selected galaxy. This data coming from
                                 the .txt file, which is created in mathematica.
    :param _rho_data: (double): the density of DM taking from .txt file.
    :param _model_rho: (double): the density of dark matter in selected radius and time \rho (r,t). Returning for all
                                 times and all radiuses. This density has been computing according to paper 2205.02957.
    :param _path: just path to file, where you want to save it.
    :param _title_name:
    :param _core_time_simulation:
    :param _core_time_fitting:
    :param _core_time_fitting_model:
    :param plot_show: did you want to see?
    :return: Nothing. This functions has been intended to create a one
    """
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_data) / len_t)
    # removing the duplicates in _time_data
    time_points = list(dict.fromkeys(_time_data))

    # relative difference
    list_rel_diff = []

    for i in range(0, time_steps):
        left_arg = i * len_t + 1 #neglect the first radius. Przesunołem o jedno w prawo
        right_arg = (i + 1) * len_t
        # part of whole data (one period)
        rho_code_points = _rho_data[left_arg:right_arg]
        rho_model_points = _model_rho[left_arg:right_arg]
        rel_diff = relative_difference(rho_code_points, rho_model_points) * 100
        # the maximum element in one period
        max_rel_diff = max(abs(rel_diff))
        # add this to the list
        # print(max_rel_diff)
        list_rel_diff.append(max_rel_diff)

    # Create plot
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.plot(time_points, list_rel_diff, '.', label='Maximum relative difference.')
    # When we should observe forming the core.
    plt.axvline(x=_core_time_fitting, color='orange',
                label='changing formula in fitting part,' + "\n" +
                      f'ocured in: {"{:.2f}".format(_core_time_fitting)} [log(Gyr)].')
    plt.axvline(x=_core_time_fitting_model, color='black',
                label='changing formula in fitting part,' + "\n" +
                      f'ocured in: {"{:.2f}".format(_core_time_fitting_model)} [log(Gyr)]. -> from model')
    plt.axvline(x=_core_time_simulation, color='green',
                label='forming the core in simulation,' + "\n" +
                      f'ocured in: {"{:.2f}".format(_core_time_simulation)} [log(Gyr)].')
    # Setting of the plot.
    plt.ylabel(r'$\max \left| \frac{\rho_{sim}(r)-\rho_{fit}(r)}{\rho_{fit}(r)} \right| \ (\%)$', fontsize=18)
    plt.xlabel(r'$\log(t \ \left[Gyr\right])$', fontsize=18)
    plt.ylim(0, 150.0)
    plt.title(_title_name, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    plt.savefig(_path + '/' + 'Maximum_relative_difference' + '.png', dpi=300)
    # we have to close figure.
    if plot_show is True:
        plt.show()
    plt.close(fig)

def maxDifferenceProfilesBackend(fig, ax, _time_data, _rho_data,
                                 _model_rho,
                                 _title_name,
                                 _core_time_simulation, _core_time_fitting, _core_time_fitting_model):
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_data) / len_t)
    # removing the duplicates in _time_data
    time_points = list(dict.fromkeys(_time_data))

    # relative difference
    list_rel_diff = []

    for i in range(0, time_steps):
        left_arg = i * len_t + 1 #neglect the first radius. Przesunołem o jedno w prawo
        right_arg = (i + 1) * len_t
        # part of whole data (one period)
        rho_code_points = _rho_data[left_arg:right_arg]
        rho_model_points = _model_rho[left_arg:right_arg]
        rel_diff = relative_difference(rho_code_points, rho_model_points) * 100
        # the maximum element in one period
        max_rel_diff = max(abs(rel_diff))
        # add this to the list
        # print(max_rel_diff)
        list_rel_diff.append(max_rel_diff)

    # Create plot
    ax.plot(time_points, list_rel_diff, '.', label='Maximum relative difference.')
    # When we should observe forming the core.
    plt.axvline(x=_core_time_fitting, color='orange',
                label='changing formula in fitting part,' + "\n" +
                      f'ocured in: {"{:.2f}".format(_core_time_fitting)} [log(Gyr)].')
    plt.axvline(x=_core_time_fitting_model, color='black',
                label='changing formula in fitting part,' + "\n" +
                      f'ocured in: {"{:.2f}".format(_core_time_fitting_model)} [log(Gyr)]. -> from model')
    plt.axvline(x=_core_time_simulation, color='green',
                label='forming the core in simulation,' + "\n" +
                      f'ocured in: {"{:.2f}".format(_core_time_simulation)} [log(Gyr)].')
    # Setting of the plot.
    plt.ylabel(r'$\max \left| \frac{\rho_{sim}(r)-\rho_{fit}(r)}{\rho_{fit}(r)} \right| \ (\%)$', fontsize=18)
    plt.xlabel(r'$\log(t \ \left[Gyr\right])$', fontsize=18)
    plt.ylim(0, 150.0)
    plt.title(_title_name, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()

def evolution_rho_core(_time_data, _rho_data,
                       _model_rho,
                       _path, _title_name,
                       _elements, plot_show=False):
    """
    :param: _elements: (int): to which radius value we are considering the core. If we take: _elements = 10,
                              we will consider firsts ten radiuses as a core.
    :return:
    """
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_data) / len_t)

    """ the data to full up """
    rho_core_sim_list = []
    rho_core_fit_list = []
    time_data_no_repetition = list(dict.fromkeys(_time_data))
    time_points = []
    for i in range(0, len(time_data_no_repetition)):
        time_value = 10**(time_data_no_repetition[i])
        time_points.append(time_value)

    for i in range(0, time_steps):
        left_arg = i * len_t
        rho_to_average_sim = []
        rho_to_average_fit = []
        for j in range(0, _elements):
            rho_to_average_sim.append(_rho_data[left_arg + j])
            # we still have problems with calculating the rho_model for shortest r, so we reject it.
            rho_to_average_fit.append(_model_rho[1 + left_arg + j])

        rho_average_sim = calculate_average(rho_to_average_sim)
        rho_average_fit = calculate_average(rho_to_average_fit)
        # appending to list contains all values of core density
        rho_core_sim_list.append(rho_average_sim)
        rho_core_fit_list.append(rho_average_fit)

    """searching for moment of the last step in forming core density"""
    # for simulation
    min_rho_core_sim = core_density(_time_data, _rho_data, _elements)
    time_min_rho_core_sim = 10**(min_rho_core_sim[1])
    # for fitting
    min_rho_core_fit = core_density(_time_data, _model_rho, _elements)
    time_min_rho_core_fit = 10**(min_rho_core_fit[1])

    # Create plot
    fig, ax = plt.subplots(figsize=(15.0, 10.0))
    # The evolution of core density for simulation
    ax.plot(time_points, rho_core_sim_list, label='The core density from simulation.')
    # The evolution of core density for fitting
    ax.plot(time_points, rho_core_fit_list, label='The core density from fitting.')
    # When we should observe forming the core.
    plt.axvline(x=time_min_rho_core_fit, color='black',
                label='forming the core in fitting,' + "\n" +
                      f'ocured in: {"{:.2f}".format(time_min_rho_core_fit)} [Gyr].')
    plt.axvline(x=time_min_rho_core_sim, color='green',
                label='forming the core in simulation,' + "\n" +
                      f'ocured in: {"{:.2f}".format(time_min_rho_core_sim)} [Gyr].')
    # Setting of the plot.
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(10**(-1), 10**4)
    plt.ylabel(r'$\rho_{core}$', fontsize=18)
    plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
    plt.title(_title_name, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    plt.savefig(_path + '/' + 'Evolution_core_density' + '.png', dpi=300)
    # we have to close figure.
    if plot_show is True:
        plt.show()
    plt.close(fig)


def evolution_rho_core_true(_time_data, _r_data, _rho_data,
                            _model_rho,
                            _path, _title_name,
                            _elements, plot_show=False):
    """
    :param _time_data:
    :param _r_data:
    :param _rho_data:
    :param _model_rho:
    :param _path:
    :param _title_name:
    :param _elements:
    :return:
    """
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_data) / len_t)

    """ the data to full up """
    rho_core_sim_list = []
    rho_core_fit_list = []
    time_data_no_repetition = list(dict.fromkeys(_time_data))
    time_points = []
    for i in range(0, len(time_data_no_repetition)):
        time_value = 10**(time_data_no_repetition[i])
        time_points.append(time_value)

    for i in range(0, time_steps):
        left_arg = i * len_t
        right_arg = i * len_t + _elements
        # data which lies inside the core
        r_in_core = _r_data[left_arg:(right_arg+1)].tolist()
        rho_in_core_sim = _rho_data[(1+left_arg):(1+right_arg)].tolist()# przesunołem o 1 w prawo
        rho_in_core_fit = _model_rho[(1+left_arg):(right_arg+1)].tolist()
        # calculate the value of the core
        value_of_rho_core_sim = calculate_rho_core(r_in_core, rho_in_core_sim, _elements)
        value_of_rho_core_fit = calculate_rho_core(r_in_core, rho_in_core_fit, _elements)

        # appending to list contains all values of core density
        rho_core_sim_list.append(value_of_rho_core_sim)
        rho_core_fit_list.append(value_of_rho_core_fit)

    """searching for moment of the last step in forming core density"""
    # for simulation
    min_rho_core_sim = find_min_rho_core(_time_data, _r_data, _rho_data, _elements)
    time_min_rho_core_sim = 10**(min_rho_core_sim[1])
    # for fitting
    min_rho_core_fit = find_min_rho_core(_time_data, _r_data, _model_rho, _elements)
    time_min_rho_core_fit = 10**(min_rho_core_fit[1])

    # Create plot
    fig, ax = plt.subplots(figsize=(15.0, 10.0))
    # The evolution of core density for simulation
    ax.plot(time_points, rho_core_sim_list, label='The core density from simulation.')
    # The evolution of core density for fitting
    ax.plot(time_points, rho_core_fit_list, label='The core density from fitting.')
    # When we should observe forming the core.
    plt.axvline(x=time_min_rho_core_fit, color='black',
                label='forming the core in fitting,' + "\n" +
                      f'ocured in: {"{:.2f}".format(time_min_rho_core_fit)} [Gyr].')
    plt.axvline(x=time_min_rho_core_sim, color='green',
                label='forming the core in simulation,' + "\n" +
                      f'ocured in: {"{:.2f}".format(time_min_rho_core_sim)} [Gyr].')
    # Setting of the plot.
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(10**(-1), 10**4)
    plt.ylabel(r'$\rho_{core}$', fontsize=18)
    plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
    plt.title(_title_name, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    plt.savefig(_path + '/' + 'Evolution_core_density' + '.png', dpi=300)
    if plot_show is True:
        plt.show()
    # we have to close figure.
    plt.close(fig)

def evolution_rho_core_true_backend(fig, ax, _time_data, _r_data, _rho_data,
                                    _model_rho,
                                    _title_name,
                                    _elements):
    """Backend for `evolution_rho_core_true`"""
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_data) / len_t)

    """ the data to full up """
    rho_core_sim_list = []
    rho_core_fit_list = []
    time_data_no_repetition = list(dict.fromkeys(_time_data))
    time_points = []
    for i in range(0, len(time_data_no_repetition)):
        time_value = 10**(time_data_no_repetition[i])
        time_points.append(time_value)

    for i in range(0, time_steps):
        left_arg = i * len_t
        right_arg = i * len_t + _elements
        # data which lies inside the core
        r_in_core = _r_data[left_arg:(right_arg+1)].tolist()
        rho_in_core_sim = _rho_data[(1+left_arg):(1+right_arg)].tolist()# przesunołem o 1 w prawo
        rho_in_core_fit = _model_rho[(1+left_arg):(right_arg+1)].tolist()
        # calculate the value of the core
        value_of_rho_core_sim = calculate_rho_core(r_in_core, rho_in_core_sim, _elements)
        value_of_rho_core_fit = calculate_rho_core(r_in_core, rho_in_core_fit, _elements)

        # appending to list contains all values of core density
        rho_core_sim_list.append(value_of_rho_core_sim)
        rho_core_fit_list.append(value_of_rho_core_fit)

    """searching for moment of the last step in forming core density"""
    # for simulation
    min_rho_core_sim = find_min_rho_core(_time_data, _r_data, _rho_data, _elements)
    time_min_rho_core_sim = 10**(min_rho_core_sim[1])
    # for fitting
    min_rho_core_fit = find_min_rho_core(_time_data, _r_data, _model_rho, _elements)
    time_min_rho_core_fit = 10**(min_rho_core_fit[1])

    # Create plot
    # The evolution of core density for simulation
    ax.plot(time_points, rho_core_sim_list, label='The core density from simulation.')
    # The evolution of core density for fitting
    ax.plot(time_points, rho_core_fit_list, label='The core density from fitting.')
    # When we should observe forming the core.
    plt.axvline(x=time_min_rho_core_fit, color='black',
                label='forming the core in fitting,' + "\n" +
                      f'ocured in: {"{:.2f}".format(time_min_rho_core_fit)} [Gyr].')
    plt.axvline(x=time_min_rho_core_sim, color='green',
                label='forming the core in simulation,' + "\n" +
                      f'ocured in: {"{:.2f}".format(time_min_rho_core_sim)} [Gyr].')
    # Setting of the plot.
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(10**(-1), 10**4)
    plt.ylabel(r'$\rho_{core}$', fontsize=18)
    plt.xlabel(r'$t \ \left[Gyr\right]$', fontsize=18)
    plt.title(_title_name, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
