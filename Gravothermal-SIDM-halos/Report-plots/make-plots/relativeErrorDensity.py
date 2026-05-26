import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ################################### NECESSARY FUNCTIONS ####################################
def relative_difference(A, B, correct_log10=False):
    """
    Calculates relative between arrays A and B.
    if correct_log == True, it will treat A and B as log10 of values to compare and exponentiate them before comparison.
    """
    if correct_log10:
        return np.log10((10**A-10**B)/10**B)
    else:
        return (A-B)/B

# ################################### FUNCTION TO PLOT ####################################
def relativeError_denisty(_time_data, _rho_data,
                          _model_rho,
                          _path, _title_name,
                          _core_time_simulation, _core_time_fitting, _core_time_fitting_model, plot_show=False):
    """
    See file: 'createPlots.py'.
    :param _path: just path to file, where you want to save it.
    :param _core_time_simulation:
    :param _core_time_fitting:
    :param _core_time_fitting_model:
    :return: Nothing. This functions has been intended to create a one
    """
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_data[len_t] == _time_data[0]: len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_data) / len_t)
    # removing the duplicates in _time_data
    time_data_no_repetition = list(dict.fromkeys(_time_data))
    time_points = [10**(time) for time in time_data_no_repetition]

    # relative difference
    list_rel_diff = []

    for i in range(0, time_steps):
        left_arg = i * len_t + 1  # neglect the first radius. Przesunołem o jedno w prawo
        right_arg = (i + 1) * len_t
        # part of whole data (one period)
        rho_code_points = _rho_data[left_arg:right_arg]
        rho_model_points = _model_rho[left_arg:right_arg]
        rel_diff = relative_difference( rho_code_points, rho_model_points) * 100
        # the maximum element in one period
        max_rel_diff = max(abs(rel_diff))
        # add this to the list
        # print(max_rel_diff)
        list_rel_diff.append(max_rel_diff)

    # Create plot
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    # set log scale
    ax.set_xscale('log')
    # description
    plt.ylabel(r'$\max \left| \frac{\rho_{sim}(r)-\rho_{fit}(r)}{\rho_{fit}(r)} \right| \ (\%)$', fontsize=18)
    plt.xlabel(r'$t$ [Gyr]', fontsize=18)
    plt.ylim(0, 150.0)
    plt.title(_title_name, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)

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
    # locmaj = mticker.LogLocator(base=10, numticks=12)
    # locmin = mticker.LogLocator(base=10.0,
    #                             subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    #                             numticks=12)
    # ax.yaxis.set_major_locator(locmaj)
    # ax.yaxis.set_minor_locator(locmin)
    # ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # tick length
    ax.tick_params('both', direction='in', top='on', right='on', length=10,
                   width=1, which='major', zorder=301)
    ax.tick_params('both', direction='in', top='on', right='on', length=5,
                   width=1, which='minor', zorder=301)

    # ----- DATA
    ax.plot(time_points, list_rel_diff, '.', label='Maximum relative difference.')
    # When we should observe forming the core.
    plt.axvline(x=10**_core_time_fitting, color='grey', linestyle='dashdot',
                label='changing formula in fitting part,' + "\n" +
                      f'occurred in: {"{:.2f}".format(10**_core_time_fitting)} [Gyr].')
    # plt.axvline(x=10**_core_time_fitting_model, color='black',
    #             label='changing formula in fitting part,' + "\n" +
    #                   occurred in: {"{:.2f}".format(10**_core_time_fitting_model)} [log(Gyr)]. -> from model')
    plt.axvline(x=10**_core_time_simulation, color='black', linestyle='dotted',
                label='forming the core in simulation,' + "\n" +
                      f'occurred in: {"{:.2f}".format(10**_core_time_simulation)} [Gyr].')
    # Setting of the plot.
    ax.legend(fontsize=14)
    plt.savefig(_path + '/' + 'Relative_error_density' + '.png', dpi=300)
    # we have to close figure.
    if plot_show is True:
        plt.show()
    plt.close(fig)
