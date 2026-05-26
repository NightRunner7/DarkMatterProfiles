"""
This script provides various functions for comparing different models, like error calculation methods etc.
Also file contains functions, which has been intended to create histograms.
    code (simulation) -> corresponding to the mathematica file (Ayuki).
    model (fitting) -> corresponding to the paper 2205.02957.
"""
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------ BASIC FUNCTION ------------------------------ #
def relative_difference(A, B, correct_log10=False):
    """
    Calculates relative between arrays A and B.
    if correct_log == True, it will treat A and B as log10 of values to compare and exponentiate them before comparison.
    """
    if correct_log10:
        return np.log10((10 ** A - 10 ** B) / 10 ** B)
    else:
        return (A - B) / B

def calculate_average(_values):
    """
    :param _values: list containg the values, which we want to average.
    :return: averaged value.
    """
    average = 0
    for i in range(0, len(_values)):
        average = average + _values[i] / (len(_values))

    return average

def core_density(_time_data, _rho_data, _elements):
    """
    :param _time_data: (double): list containing the all times of evolution of selected galaxy. This data coming from
                                 the .txt file, which is created in mathematica.
    :param _rho_data:  (double): the density of DM taking from .txt file (or from fitting values)
    :param _elements: (int): to which radius value we are considering the core. If we take: _elements = 10,
                             we will consider firsts ten radiuses as a core.
    :return: list, which contains information about the core.
             list[0] -> value of core density.
             list[1] -> value of time [log_10(Gyr)], when we can observe forming a core (before collapsing).
             list[2] -> time step, which is corresponding to forming a core.
    """
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_data) / len_t)

    # the density of the core
    rho_core = _rho_data[1]
    time_core = 0
    time_step_core = 0

    for i in range(0, time_steps):
        left_arg = i * len_t
        rho_to_average = []
        for j in range(0, _elements):
            rho_to_average.append(_rho_data[left_arg + j])

        rho_average = calculate_average(rho_to_average)

        if rho_average < rho_core:
            rho_core = rho_average
            time_core = _time_data[i * len_t]
            time_step_core = i
        else:
            continue

    return [rho_core, time_core, time_step_core]

def calculate_rho_core(_values_r, _values_rho, _elements):
    """
    :param _values_r:
    :param _values_rho:
    :param _elements:
    :return:
    """
    rho_core = 0

    for i in range(0, _elements):
        part_volume = (_values_r[i+1] / _values_r[_elements])**3 - (_values_r[i] / _values_r[_elements])**3
        rho_core = rho_core + part_volume * _values_rho[i]

    return rho_core

def find_min_rho_core(_time_data, _r_data, _rho_data, _elements):
    """
    :param _time_data:
    :param _r_data:
    :param _rho_data:
    :param _elements:
    :return:
    """
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_data) / len_t)

    # the minimum density of core
    min_rho_core = _rho_data[1]
    time_min_core = 0
    time_step_min_core = 0

    for i in range(0, time_steps):
        left_arg = i * len_t
        right_arg = i * len_t + _elements
        # data which lies inside the core
        r_in_core = _r_data[left_arg:(right_arg+1)].tolist()
        rho_in_core = _rho_data[left_arg:right_arg].tolist()
        # calculate the value of the core
        value_of_rho_core = calculate_rho_core(r_in_core, rho_in_core, _elements)

        if min_rho_core > value_of_rho_core:
            min_rho_core = value_of_rho_core
            time_min_core = _time_data[i * len_t]
            time_step_min_core = i

    return [min_rho_core, time_min_core, time_step_min_core]


# ------------------------------ HISTOGRAMS ------------------------------ #
def difference_histogram(_time_data, _rho_data,
                         _model_rho,
                         _path, _time_core_simulation):
    """
    Creates a histogram of errors.
    :param _time_data:
    :param _rho_data:
    :param _model_rho:
    :param _path:
    :param _time_core_simulation:
    :return:
    """
    # We present data only to forming the core
    lines_to_core = 0
    while _time_data[lines_to_core] != _time_core_simulation: lines_to_core += 1
    # Computing the differences between the simulation and fitting formula.
    rel_error = relative_difference(_rho_data[0:lines_to_core], _model_rho[0:lines_to_core]) * 100

    # create two histogram
    #
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(11.0, 12.0))
    # first histogram
    ax0.hist(rel_error, bins=100)
    ax0.set_xlabel('Relative difference (%)', fontsize=18)
    ax0.set_ylabel('Occurrence', fontsize=18)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.set_title('Up to forming a core', fontsize=18)
    # second histogram
    ax1.hist(rel_error, bins=100, log=True)
    ax1.set_xlabel('Relative difference (%)', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title('Up to forming a core', fontsize=18)

    # third histogram
    ax2.hist(rel_error, bins=100, density=True)
    ax2.set_xlabel('Relative difference (%)', fontsize=18)
    ax2.set_ylabel('Normalised', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    # fourth histogram
    ax3.hist(rel_error, bins=100, log=True, density=True)
    ax3.set_xlabel('Relative difference (%)', fontsize=18)
    ax3.tick_params(axis='both', which='major', labelsize=12)

    # Corresponds to both histograms
    plt.savefig(_path + '/' + 'histogram_relative_difference_to_forming_core' + '.png')
    # we have to close figure.
    # plt.show()
    plt.close(fig)
