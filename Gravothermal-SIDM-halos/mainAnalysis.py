# ################################# IMPORTING ################################# #
# --- Importing packages
import pprint
import pandas as pd
# --- to create gif
from pathlib import Path
import imageio.v2 as imageio

import os
import sys
currentdir = "C:\\Users\\Krzysztof\\Documents\\GitHub\\DarkMatterProfiles\\Gravothermal-SIDM-halos"
parentdir = os.path.dirname(currentdir)
# --- take file from same directory
sys.path.insert(0, currentdir)

from createHistograms import relative_difference, core_density, find_min_rho_core, difference_histogram
from createPlots import onePlotDensityProfile, comparisonTwoDensityProfiles, numeratePlots, differenceDensityProfiles,\
    comparisonThreeDifferenceProfiles
from createPlotsSummary import maxDifferenceProfiles, evolution_rho_core, evolution_rho_core_true
from createPlotsModelParameters import differenceWtithFLRW, evolutionRadiusCore_Model
import modelGravothermalDensities as mod

plot_show = False  # this is meant to be true if you want to show plot while running code

# ################################# RUNNING FOR ALL INPUT FILES ################################# #
# ---The localization and lists of files in there.
data_dir = 'Input'
list_data_files_names = []
beta = 0.75
redshift = 0

for file in os.listdir(currentdir + '\\' + data_dir):
    # for file in os.listdir(currentdir+'/'+data_dir):
    if file.endswith(".txt"):
        list_data_files_names.append(file[:-4])  # this is how to erase '.txt' in file names

for _selected_file in list_data_files_names:
    # for all files in directory
    data_file_name = _selected_file
    rho_path = data_dir + '/' + data_file_name + '.txt'

    # splitting into parts the file name
    split_file_name = data_file_name.split("_")

    """
    Some variables is fixed during the executing the file in mathematica. That fixed variables have shown in the name
    of the file, which containing the data. Those variables is:

        power_mass: (float): the initial mass of the galaxy. If init_M = 10.1 means that our galaxy have
            Mass = 10**10.1 M_solar. Unit is [M_solar].
        evolution_time: (float): evolution time of the galaxy. Unit is [Gyr].
        sigma_m: (float): cross section of selected DM (SIDM). Unit is [cm^2/g].
    """
    # --- set values of variables
    power_mass = float(split_file_name[2])
    evolution_time = float(split_file_name[4])
    sigma_m = float(split_file_name[6])
    mvir = 10 ** power_mass  # viral mass

    print(split_file_name)
    print('power_mass:', power_mass, ', evolution_time:', evolution_time, ', sigma_m:', sigma_m)
    # Based on fixed variables during simulation we will set the title name of important plots
    Title_name = f'Mass: {("{:.1e}".format(10**power_mass))} [M_odot], sigma: {("{:.1e}".format(sigma_m))} [cm^2/g], ' \
                 f't: {("{:.1e}".format(evolution_time))} [Gyr]'
    print(Title_name)

    # ---------------------------------- PREPARE DIRECTORIES ---------------------------------- #
    """
    Important! If the name of the variables is started with 'output' it means that variables has been connecetd with
    charts or something which was created during the executioning this code.

    Shortcuts:
        dir: director
        var: variables
    Definition:
        output_dir_var: the main directory, which has been named by fixed variables using in simulation. 
        This is the main directory for all plots creating in this code.
    """
    output_dir_plots = 'Plots'
    output_dir_var = 'M_' + str(power_mass) + '_t_' + str(evolution_time) + '_sigma_' + str(sigma_m)
    output_plots_path = output_dir_plots + '/' + output_dir_var
    # creating the main director, which will contains plots
    try:
        # Create target Directory
        os.mkdir(output_plots_path)
        print("Directory ", output_plots_path, " Created ")
    except FileExistsError:
        print("Directory ", output_plots_path, " already exists")

    """Creating subdirectories"""
    output_density_profiles_path = output_plots_path + '/' + 'Evolution-Density-Profiles'
    output_density_profiles_errors_path = output_plots_path + '/' + 'Errors-Density-Profiles'
    output_special_points = output_plots_path + '/' + 'Important-points'

    list_subdirectories = [output_density_profiles_path, output_density_profiles_errors_path, output_special_points]

    for i in range(0, len(list_subdirectories)):
        try:
            # Directory containg the Evolution Density Profiles
            os.mkdir(list_subdirectories[i])
            print("Directory ", list_subdirectories[i], " Created ")
        except FileExistsError:
            print("Directory ", list_subdirectories[i], " already exists")

    # print(output_plots_path)

    # ---------------------------------- TAKE AND SET DATA FROM .txt FILE ---------------------------------- #
    # --- create map, which containg all data coming from file created by mathematica code.
    rho_data = pd.read_csv(rho_path, sep='\t', names=['t', 'r', 'rho'])
    rho_data = rho_data.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r
    # add new element to map: t_dimless, which will refer to time in dimensionless units.
    rho_s = mod.cal_rho_s(mvir, redshift)
    r_s = mod.cal_r_s(mvir, redshift)
    rho_data['t_dimless'] = mod.cal_time_tilda(rho_data['t'], rho_s, r_s, sigma_m, log_time=True)
    # print(" rho_data['t_dimless']")
    # pprint.pprint(rho_data['t_dimless'])
    # print()
    # print(" rho_data['t']")
    # pprint.pprint(rho_data['t'])
    # print()

    # --- create the density profile according to the procedure using in paper 2205.02957
    ModelGravoClass = mod.ModelGravothermal(beta=beta)
    model_rho = ModelGravoClass.rho_model(rho_data['r'], rho_data['t_dimless'])
    # print("model_rho")
    # pprint.pprint(model_rho)
    # print()

    # different values of time. In other words number of steps in time
    len_t = 0
    while rho_data['t'][len_t] == rho_data['t'][0]:
        len_t += 1
    # len_t -= 1

    # how many steps on time is in the .txt file
    time_steps = int(len(rho_data['t']) / len_t)
    print('Number of time steps in .txt file:', time_steps)

    # --- differentiate two regime during evolution of density profile
    """
    Now we will calculate when the core occurs.

    core_density returns list, which contains information about core.
       list[0] -> value of core density.
       list[1] -> value of time [log_10(Gyr)], when we can observe forming a core (before collapsing).
       list[2] -> time step, which is corresponding to forming a core.
    """
    core_simulation = find_min_rho_core(rho_data['t'], rho_data['r'], rho_data['rho'], 10)
    print("core_simulation:", core_simulation)
    core_fitting = find_min_rho_core(rho_data['t'], rho_data['r'], model_rho, 10)
    print("core_fitting", core_fitting)

    # time of changing formula taking explicite from data_model calculating time
    T_dimless_change = ModelGravoClass.return_time_transition()
    log_T_change = mod.convert_time_tilda(T_dimless_change, rho_s, r_s, sigma_m, log_time=True)
    print("T_dimless_change:", T_dimless_change)
    print("Time of changing formula using data_model:", log_T_change, "[log(Gyr)]")

    # ---------------------------------- STARTING DRAW PLOTS ---------------------------------- #
    # ------------------- 3D PLOT
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # slice = 550 # decrease to include more points, may make the plot more cluttered and decreases performance. @ slice = 10 becomes practically solid

    # Will propably devise a better rendering technique later.
    # t_3D_plot = rho_data['t'][::slice]
    # r_3D_plot = np.log10(rho_data['r'][::slice])
    # ax.plot(t_3D_plot, r_3D_plot, np.log10(rho_data['rho'][::slice]), '.', label='Our mathematica simulation ρsol_M_10.1_t_10.0_sigma_0.008_beta_0.75.txt')
    # ax.plot(t_3D_plot, r_3D_plot, np.log10(model_rho[::slice]), '.', label='Fitted equation from paper 2205.02957')
    # ax.set_xlabel('$\log_{10}(t)$')
    # ax.set_ylabel('$\log_{10}(r)$')
    # ax.set_zlabel(r'$\log_{10}(\rho)$')
    # ax.legend()

    # plt.show()

    # ------------------- DENSITY PROFILE IN BEGINNING AND IN THE END
    # I will draw a comparison from model and code the density profile for selected period time
    # (or step time as you prefer).

    # period_1 = 3
    # period_2 = (time_steps - 10)
    #
    # comparisonTwoDensityProfiles(rho_data['t'], rho_data['r'], rho_data['rho'],
    #                              model_rho,
    #                              period_1, period_2)

    # ------------------- EVOLUTION OF DENSITY PROFILES: with gif
    """
    beg_period: (double): the beginning time step.
    end_period: (double):  the ending time step.
    nums_plot: (double): how many density profiles I want to save.
    power_mass: (double): the initial mass of the galaxy. If init_m = 10.1 means that our galaxy 
        have Mass = 10**10.1 M_solar.

    time_steps: (int): number of steps in our .txt file
    """
    beg_period = 10  # 500
    end_period = time_steps
    nums_plot = 60
    periods = []
    for i in range(0, nums_plot+1):
        jump = int((end_period - beg_period) / nums_plot)
        add_period = beg_period + i * jump

        periods.append(add_period)

    """
    If you want to add some specific time period just change extra_periods.
    I will used that, because in time period:
        2791: a moment before the collapse.
        2792: a moment after the collapse.
    """
    extra_periods = [2791, 2792]

    if (time_steps > extra_periods[0]) and (time_steps > extra_periods[1]):
        """we have to sure that we can add this extra_periods"""
        for i in range(0, len(extra_periods)):
            add_period = extra_periods[i]
            nums_plot = nums_plot + 1

            periods.append(add_period)

    # sort the list
    periods.sort()

    # create the list, which will contains the names of plots connected with periods
    density_plots_names = numeratePlots(output_density_profiles_path, "Density_profile_", nums_plot+1)

    # print(periods)
    # for item in density_plots_names:
    #     print(item)

    # take coefficients
    whole_cof_list = []
    for i in range(0, len(periods)):
        period = periods[i]
        T_dimless = rho_data["t_dimless"][len_t * period + 1]  # [dimensionless]
        # take coefficient for fixed time
        cof_list = ModelGravoClass.return_coefficients(T_dimless)
        whole_cof_list.append(cof_list)

    # Now we are going to create the plots, which presents the evolution of the density profile
    for i in range(0, nums_plot+1):
        onePlotDensityProfile(whole_cof_list[i],
                              rho_data['t'], rho_data['r'], rho_data['rho'],
                              model_rho,
                              periods[i],
                              density_plots_names[i])

    """
    Now we will gona present this as gif
        fps: figures per second.
        loop = False - the ouroboros style (if he reaches end, the will continue from the beginning).
        loop = True - do only a one loop.
    """
    image_path = Path(currentdir+'\\'+output_density_profiles_path)
    images = list(image_path.glob('*.png'))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))
    # creating the gif
    gif_name = 'animated_density_profile.gif'
    gif_path = output_plots_path + '/' + gif_name
    imageio.mimwrite(gif_path, image_list, fps=5, loop=True)

    # ------------------- RELATIVE DIFFERENCE BETWEEN THE SIMULATION AND FITTING
    """
    We make the plots, which presents relative difference between simulation (mathematica) and fitting (paper) in
    setting the denisty profiles.

    We make the plots for same time step as for evolution density profiles.
    """
    # create the list, which will contains the names of plots connected with periods
    density_plots_names = numeratePlots(output_density_profiles_errors_path, "Relative_difference_", nums_plot)

    # print(periods)
    # for item in density_plots_names:
    #     print(item)

    # Now we are going to create the plots, which presents how evolves differences between
    # simulation and fitting formula.
    for i in range(0, nums_plot):
        differenceDensityProfiles(whole_cof_list[i],
                                  rho_data['t'], rho_data['r'], rho_data['rho'],
                                  model_rho,
                                  periods[i],
                                  density_plots_names[i])

    # ------------------- THREE DIFFERENT PROFILES
    # I will draw a comparsion from model and code the density profile for selected period
    # time (or step time as you prever).

    # period_1 = 3
    # period_2 = 1000
    # period_3 = time_steps - 80
    #
    # comparisonThreeDifferenceProfiles(rho_data['t'], rho_data['r'], rho_data['rho'],
    #                                   model_rho,
    #                                   period_1, period_2, period_3)

    # ------------------- SUMMARY OF KNOWLEDGE
    # Now we draw the Maximum relative difference in time between density profiles coming from simulation and fitting.
    # set_y_limit = 1000
    maxDifferenceProfiles(rho_data['t'], rho_data['rho'],
                          model_rho,
                          output_plots_path, Title_name,
                          core_simulation[1], core_fitting[1], log_T_change, plot_show)

    # Now we can draw histograms, which presenting the relative differences between simulation and fitting part.
    difference_histogram(rho_data['t'], rho_data['rho'],
                         model_rho,
                         output_plots_path, core_simulation[1])

    # Evolution of core density.
    evolution_rho_core_true(rho_data['t'], rho_data['r'], rho_data['rho'],
                            model_rho,
                            output_plots_path, Title_name,
                            10, plot_show)

    # ------------------- TINY ISSUE
    """
    We still have a problem with density in fitting for smallest radius!
    """
    print(model_rho)

    # ------------------- ANALYSIS OF THE MODEL
    evolutionRadiusCore_Model(mvir, sigma_m, beta, evolution_time,
                              output_plots_path)

    differenceWtithFLRW(mvir, sigma_m, beta, evolution_time,
                        output_plots_path)


    # ------------------- SPECIAL POINTS
    # Create plot density and errors for special points (beginning, transition and end)

    transition_time = log_T_change
    closest_time = rho_data.iloc[(rho_data['t'] - transition_time).abs().argsort()[:1]]
    closest_time_index = (closest_time['t'].index)[0]  # how to get index
    closest_time_step = int(closest_time_index / len_t)

    nums_special = 3
    # data
    beginning_period = []
    transition_period = []
    ending_period = []
    # names
    beg_names = []
    trans_names = []
    ends_names = []

    for i in range (0, nums_special):
        # appening beginning period
        beginning_period.append(i)
        # appending transition period
        transition_index = closest_time_step - int(nums_special / 2) + i
        transition_period.append(transition_index)
        # appending ending periods
        ending_index = (time_steps) - nums_special + i
        ending_period.append(ending_index)

        # names
        beg_name = output_special_points + "/" + "Beginning_period_" + str(i + 1)
        beg_names.append(beg_name)
        trans_name = output_special_points + "/" + "Transition_period_" + str(i + 1)
        trans_names.append(trans_name)
        ends_name = output_special_points + "/" + "Ending_period_" + str(i + 1)
        ends_names.append(ends_name)

    # adding to the main list
    special_periods = []
    special_periods.extend(beginning_period)
    if transition_period[0] < ending_period[0]:
        # transition occurred
        special_periods.extend(transition_period)
    special_periods.extend(ending_period)
    # adding names
    special_names = []
    special_names.extend(beg_names)
    if transition_period[0] < ending_period[0]:
        # transition occurred
        special_names.extend(trans_names)
    special_names.extend(ends_names)
    # size
    special_plots = len(special_periods)

    print(special_periods)
    # take coefficients
    whole_cof_list = []
    for i in range(0, len(special_periods)):
        period = special_periods[i]
        T_dimless = rho_data["t_dimless"][len_t * period + 1]  # [dimensionless]
        # take coefficient for fixed time
        cof_list = ModelGravoClass.return_coefficients(T_dimless)
        whole_cof_list.append(cof_list)

    for i in range(0, special_plots):
        onePlotDensityProfile(whole_cof_list[i],
                              rho_data['t'], rho_data['r'], rho_data['rho'],
                              model_rho,
                              special_periods[i],
                              special_names[i])

    for i in range(0, len(special_names)):
        special_names[i] = special_names[i] + '_difference'

    for i in range(0, special_plots):
        differenceDensityProfiles(whole_cof_list[i],
                                  rho_data['t'], rho_data['r'], rho_data['rho'],
                                  model_rho,
                                  special_periods[i],
                                  special_names[i])
