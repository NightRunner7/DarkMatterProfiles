import pandas as pd
from pathlib import Path

# ####################### IMPORTING FILES ####################### #
import os
import sys
import modelGravothermalDensities as mod

# ------------ LOCALIZATION OF DIRECTORIES ------------#
currentdir = "C:\\Users\\Krzysztof\\Documents\\GitHub\\DarkMatterProfiles\\Gravothermal-SIDM-halos\\Report-plots"
makeplotsdir = currentdir + '\\' + 'make-plots'
parentdir = os.path.dirname(currentdir)
datadir = parentdir + '\\' + 'Input'

# ------------ FILES MAKEPLOTDIR ------------#
sys.path.insert(0, makeplotsdir)
from evolutionRhoInTime import find_min_rho_core, evolution_rho_core_true, evolution_rho_core_true_dimless
from fittingParametersCheck import evolutionRadiusCore_Model, differenceWtithNFW
from densityProfiles import onePlotDensityProfile
from relativeErrorDensity import relativeError_denisty

# ####################### USER SETS ########################
plot_show = False  # this is meant to be true if you want to show plot while running code
# file names
file_name = 'ρsol_M_10._t_10._sigma_100.'
file_path = datadir + '/' + file_name + '.txt'
output_plots_path = currentdir
output_special_points = currentdir
# other
n_elements = 10  # how many firsts radius we treat as a belonging to 'core'.
beta = 0.75
redshift = 0

# ####################### DEAL WITH THE FILES ####################### #
# ------------ SPLITING INTO PARTS THE FILE ------------#
split_file_name = file_name.split("_")
"""
Some variables is fixed during the executing the file in mathematica. That fixed variables have shown in the name
of the file, which containing the data. Those variables is:

    power_mass: (float): the inital mass of the galaxy. If init_M = 10.1 means that our galaxy have 
        Mass = 10**10.1 M_solar. Unit is [M_solar].
    evolution_time: (float): evolution time of the galaxy. Unit is [Gyr].
    sigma_m: (float): cross section of selected DM (SIDM). Unit is [cm^2/g].
"""
power_mass = float(split_file_name[2])
evolution_time = float(split_file_name[4])
sigma_m = float(split_file_name[6])
mvir = 10 ** power_mass  # viral mass

print(split_file_name)
print('init_M:', power_mass, ', evolution_time:', evolution_time, ', sigma_m:', sigma_m)

# ------------ TITLED NAME ------------ #
"""Based on fixed variables during simulation we will set the title name of important plots."""

Title_name = f'Mass: {("{:.1e}".format(10 ** power_mass))} [M_sun], sigma: {("{:.1e}".format(sigma_m))} [cm^2/g], ' \
             f't: {("{:.1e}".format(evolution_time))} [Gyr]'
print(Title_name)

# ####################### DEAL WITH DATA AND PLOTS ####################### #
# --- create map, which containg all data coming from file created by mathematica code.
rho_data = pd.read_csv(file_path, sep='\t', names=['t', 'r', 'rho'])
rho_data = rho_data.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r
# add new element to map: t_dimless, which will refer to time in dimensionless units.
rho_s = mod.cal_rho_s(mvir, redshift)
r_s = mod.cal_r_s(mvir, redshift)
rho_data['t_dimless'] = mod.cal_time_tilda(rho_data['t'], rho_s, r_s, sigma_m, log_time=True)

# --- create the density profile according to the procedure using in paper 2205.02957
ModelGravoClass = mod.ModelGravothermal(beta=beta)
model_rho = ModelGravoClass.rho_model(rho_data['r'], rho_data['t_dimless'])

# different values of time. In other words number of steps in time
len_t = 0
while rho_data['t'][len_t] == rho_data['t'][0]:
    len_t += 1

# how many steps on time is in the .txt file
time_steps = int(len(rho_data['t']) / len_t)
print('Number of time steps in .txt file:', time_steps)

# --------------------------------------- CORE PLOT --------------------------------------- #
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

evolution_rho_core_true(rho_data['t'], rho_data['r'], rho_data['rho'],
                        model_rho,
                        output_plots_path, Title_name,
                        n_elements, plot_show)

evolution_rho_core_true_dimless(rho_data['t_dimless'], rho_data['r'], rho_data['rho'],
                                model_rho,
                                output_plots_path, Title_name,
                                n_elements, plot_show)

relativeError_denisty(rho_data['t'], rho_data['rho'],
                      model_rho,
                      output_plots_path, Title_name,
                      core_simulation[1], core_fitting[1], log_T_change, plot_show)

# ------------ ANALYSIS OF THE MODEL ------------#

evolutionRadiusCore_Model(mvir, sigma_m, beta, evolution_time,
                          output_plots_path)

differenceWtithNFW(mvir, sigma_m, beta, evolution_time,
                   output_plots_path)

# ------------ DENSITY PLOTS: SPECIAL CASSES ------------#
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

for i in range(0, nums_special):
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
