"""
In this file we will use following shortcuts:
    sim: simulation, which denotes to origin. Simulation refers to data, which we get from mathematica file, where
         we used gravothermal code.
    fit: fitting, which denotes to origin. Fitting refers to data, which we get from fitting function, presented
         in paper 2205.02957. This fitting is developed to approximate the gravothermal solution (speed-up calculating
         process). It occurs that this fitting has some troubles, but it will present later.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ################################### NECESSARY FUNCTIONS ####################################
def calculate_rho_core(_values_r, _values_rho, _elements):
    """
    :param _values_rho: (double) list of density values. One important remark, we treat that density [i-th] density
                        is constant from [i-th] to [i+1-th] radius. That means radius list have one more element
                        than list of densities
    :param _values_r: (double) list of radius values
    :param _elements: (int) number of elements from whole list of radiuses, densities; which we treat a part of a core.
                      One important remark: only o few first elements we will treat in such a way.
    :return: the value of averaged core density (double).
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
    :param _elements: (int) number of elements from whole list of radiuses, densities; which we treat a part of a core.
                      One important remark: only o few first elements we will treat in such a way.
    :return: [min_rho_core, time_min_core, time_step_min_core], where
             min_rho_core: (double) minimum value of rho core. We know that \rho_core(t). So this variable refers to
                            time when core end forming and starts to collapse.
             time_min_core: (log10(double)) time when core is formed, or time when forming core is end up.
             time_step_min_core: (int) time step, which is connected with time_min_core.
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


# ################################### FUNCTION TO PLOT ################################### #

def evolution_rho_core_true(_time_sim, _r_sim, _rho_sim,
                            _fit_rho,
                            _path, _title_name,
                            _elements, plot_show=False):
    """
    :param _time_sim: (double): list containing the all times of evolution of selected galaxy. This data coming from
                                the .txt file, which is created in mathematica (simulation).
                                [log10(Gyr)].
    :param _r_sim: (double): list containing the all values of radius of selected galaxy. This data coming from
                             the .txt file, which is created in mathematica (simulation).
                             [r_s].
    :param _rho_sim: (double): the density of DM taking from .txt file, which is created in mathematica (simulation).
                               [rho_s].
    :param _fit_rho: (double): the density of dark matter in selected radius and time \rho (r,t). Returning for all
                               times and all radiuses. This density has been computing according to paper 2205.02957.
                               [rho_s].
    :param _path: (string): location from working directory to save plot.
    :param _title_name: (string): contains the title name of plot, for example the necessary fixed variables.
    :param _elements: (int): the number of elements, how many first radiuses we take into account and treat as
                      'the core'.
    :return: nothing just save plot in directed path.
    """
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_sim[len_t] == _time_sim[0]:
        len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_sim) / len_t)

    """ the data to full up """
    rho_core_sim_list = []
    rho_core_fit_list = []
    time_data_no_repetition = list(dict.fromkeys(_time_sim))
    time_points = [10**time for time in time_data_no_repetition]  # Gyr
    # for i in range(0, len(time_data_no_repetition)):
    #     time_value = 10**(time_data_no_repetition[i])
    #     time_points.append(time_value)

    for i in range(0, time_steps):
        left_arg = i * len_t
        right_arg = i * len_t + _elements
        # data which lies inside the core
        r_in_core = _r_sim[left_arg:(right_arg+1)].tolist()
        rho_in_core_sim = _rho_sim[(1+left_arg):(1+right_arg)].tolist()# przesunołem o 1 w prawo
        rho_in_core_fit = _fit_rho[(1+left_arg):(1+right_arg)].tolist()
        # calculate the value of the core
        value_of_rho_core_sim = calculate_rho_core(r_in_core, rho_in_core_sim, _elements)
        value_of_rho_core_fit = calculate_rho_core(r_in_core, rho_in_core_fit, _elements)

        # appending to list contains all values of core density
        rho_core_sim_list.append(value_of_rho_core_sim)
        rho_core_fit_list.append(value_of_rho_core_fit)

    """searching for moment of the last step in forming core density"""
    # for simulation
    min_rho_core_sim = find_min_rho_core(_time_sim, _r_sim, _rho_sim, _elements)
    # time_min_rho_core_sim = 10**(min_rho_core_sim[1])
    time_min_rho_core_sim = 10**min_rho_core_sim[1]

    # for fitting
    min_rho_core_fit = find_min_rho_core(_time_sim, _r_sim, _fit_rho, _elements)
    # time_min_rho_core_fit = 10**(min_rho_core_fit[1])
    time_min_rho_core_fit = 10**min_rho_core_fit[1]


    # Create plot
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # describe plot
    plt.ylim(10**(0), 2*10**(3))
    plt.ylabel(r'$\hat{\rho}_{core} \, [\rho_s]$', fontsize=18)
    plt.xlabel(r'$t \, \left[Gyr\right]$', fontsize=18)
    plt.title(_title_name, fontsize=18)
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
    # The evolution of core density for simulation
    ax.plot(time_points, rho_core_sim_list, label='The core density from simulation.')
    # The evolution of core density for fitting
    ax.plot(time_points, rho_core_fit_list, label='The core density from fitting.')
    # When we should observe forming the core.
    plt.axvline(x=time_min_rho_core_fit, color='grey', linestyle='dashdot',
                label='forming the core in fitting,' + "\n" +
                      f'occurred in: {"{:.2f}".format(time_min_rho_core_fit)} [Gyr].')
    plt.axvline(x=time_min_rho_core_sim, color='black', linestyle='dotted',
                label='forming the core in simulation,' + "\n" +
                      f'occurred in: {"{:.2f}".format(time_min_rho_core_sim)} [Gyr].')

    # --- LEGEND AND SAVE
    ax.legend(fontsize=14)
    plt.savefig(_path + '/' + 'Evolution_core_density' + '.png', dpi=300)
    if plot_show is True:
        plt.show()
    # we have to close figure.
    plt.close(fig)

def evolution_rho_core_true_dimless(_time_sim, _r_sim, _rho_sim,
                                    _fit_rho,
                                    _path, _title_name,
                                    _elements, plot_show=False):
    """
    :param _time_sim: (double): list containing the all times of evolution of selected galaxy. This data coming from
                                the .txt file, which is created in mathematica (simulation).
                                [dimensionless].
    :param _r_sim: (double): list containing the all values of radius of selected galaxy. This data coming from
                             the .txt file, which is created in mathematica (simulation).
                             [r_s].
    :param _rho_sim: (double): the density of DM taking from .txt file, which is created in mathematica (simulation).
                               [rho_s].
    :param _fit_rho: (double): the density of dark matter in selected radius and time \rho (r,t). Returning for all
                               times and all radiuses. This density has been computing according to paper 2205.02957.
                               [rho_s].
    :param _path: (string): location from working directory to save plot.
    :param _title_name: (string): contains the title name of plot, for example the necessary fixed variables.
    :param _elements: (int): the number of elements, how many first radiuses we take into account and treat as
                      'the core'.
    :return: nothing just save plot in directed path.
    """
    # Same values of time - how many radius values we have for each one fixed time
    len_t = 0
    while _time_sim[len_t] == _time_sim[0]:
        len_t += 1
    # how many steps on time is in the .txt file
    time_steps = int(len(_time_sim) / len_t)

    """ the data to full up """
    rho_core_sim_list = []
    rho_core_fit_list = []
    time_points = list(dict.fromkeys(_time_sim))  # [dimensionless]
    # for i in range(0, len(time_data_no_repetition)):
    #     time_value = 10**(time_data_no_repetition[i])
    #     time_points.append(time_value)

    for i in range(0, time_steps):
        left_arg = i * len_t
        right_arg = i * len_t + _elements
        # data which lies inside the core
        r_in_core = _r_sim[left_arg:(right_arg+1)].tolist()
        rho_in_core_sim = _rho_sim[(1+left_arg):(1+right_arg)].tolist()# przesunołem o 1 w prawo
        rho_in_core_fit = _fit_rho[(1+left_arg):(1+right_arg)].tolist()
        # calculate the value of the core
        value_of_rho_core_sim = calculate_rho_core(r_in_core, rho_in_core_sim, _elements)
        value_of_rho_core_fit = calculate_rho_core(r_in_core, rho_in_core_fit, _elements)

        # appending to list contains all values of core density
        rho_core_sim_list.append(value_of_rho_core_sim)
        rho_core_fit_list.append(value_of_rho_core_fit)

    """searching for moment of the last step in forming core density"""
    # for simulation
    min_rho_core_sim = find_min_rho_core(_time_sim, _r_sim, _rho_sim, _elements)
    # time_min_rho_core_sim = 10**(min_rho_core_sim[1])
    time_min_rho_core_sim = min_rho_core_sim[1]

    # for fitting
    min_rho_core_fit = find_min_rho_core(_time_sim, _r_sim, _fit_rho, _elements)
    # time_min_rho_core_fit = 10**(min_rho_core_fit[1])
    time_min_rho_core_fit = min_rho_core_fit[1]


    # Create plot
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # describe plot
    plt.ylim(10**(0), 2*10**(3))
    plt.ylabel(r'$\hat{\rho}_{core} \, [\rho_s]$', fontsize=18)
    plt.xlabel(r'$\hat{t} \ \left[\frac{4}{\sqrt{\pi}} \hat{\sigma}_{m} \hat{\nu} \hat{\rho} \right]$', fontsize=18)
    plt.title(_title_name, fontsize=18)
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
    # The evolution of core density for simulation
    ax.plot(time_points, rho_core_sim_list, label='The core density from simulation.')
    # The evolution of core density for fitting
    ax.plot(time_points, rho_core_fit_list, label='The core density from fitting.')
    # When we should observe forming the core.
    plt.axvline(x=time_min_rho_core_fit, color='grey', linestyle='dashdot',
                label='forming the core in fitting,' + "\n" +
                      f'occurred in: {"{:.2f}".format(time_min_rho_core_fit)} [dimless].')
    plt.axvline(x=time_min_rho_core_sim, color='black', linestyle='dotted',
                label='forming the core in simulation,' + "\n" +
                      f'occurred in: {"{:.2f}".format(time_min_rho_core_sim)} [dimless].')

    # --- LEGEND AND SAVE
    ax.legend(fontsize=14)
    plt.savefig(_path + '/' + 'Evolution_core_density_dimless' + '.png', dpi=300)
    if plot_show is True:
        plt.show()
    # we have to close figure.
    plt.close(fig)
