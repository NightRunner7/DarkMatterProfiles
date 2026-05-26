"""
!!!!!!!!!!!!!!!!!!!! BUGS: WORKING PROGRES !!!!!!!!!!!!!!!!!!!!


In this file we're interested in getting data till(!) a time, when the Rprocedure fails to describe evolution
of dark matter in galaxy. When and why this procedure failed at some time was mentioned in the
`IsothermalSIDMModel.py`. Shortly speaking at some time the unphysical solution become the physical one.
Thus, we evolve our system to that point and save the data, which describes the dark matter at that time
-> we will use that data as initial condition in the gravothemal simulation thanks to which we get the
results farther in the future.

Important remark: in the ending step in the evolution we somehow combine the `Isothermal` and `NFW` profile
of dark matter. To better understanding see files: `CoreNFWProfile.py` and `NFWProfile.py`. Moreover, the ending
step we get from fitting to the `core NFW` profile. We do so, because previously the profile of velocity dispersion
have discontinuity (discontinuity of the derivative at one point), which is numerical obstacle during gravothermal
simulation -> in that approach, we do not have sth like this.

Plus: calculate really quickly the velocity dispersion through the R-procedure simulation.
Minus: the velocity dispersion, which we obtained from `coreNFW` is not accurate to put
    into gravothermal simulation (because of dimple for small radius).
"""
# ############# IMPORTING ############# #
import os
import sys
import numpy as np
import pandas as pd
import csv  # save file
# ------------ SETS PATH ------------ #
currentdir = os.getcwd()
makePlotsDir = currentdir + '\\make-plots'
# ------------ FILES makePlotsDir ------------ #
sys.path.insert(0, makePlotsDir)
from plotEvolutionRhoISO import plotEvolutionRhoISO
# ------------ FILES CURRENTDIR ------------ #
sys.path.insert(0, currentdir)
from NFWProfile import NFWProfile, r1  # CDM profile (halo)
from IsothermalSIDMModel import IsoEvolution  # ISO profile (halo)
import config as cfg
import auxiliaryFunctions as aux
import units as uni
from IsoAndHalo import ISO_and_NFW  # ISO + NFW profile (halo)
from CoreNFWProfile import find_fitting_parm, CoreNFWProfile

# ############# SETS OF SIMULATION: USER SETS! ############# #
rel_err_margin = 1.0 + 1e-2  # see `calRelErr_mergin` in `auxiliaryFunctions.py`
sigma_m = 1.0  # [cm^2/g] annihilation cross-section
sigma_m_SI = sigma_m * 10**(-4) * 10**3  # [m^2/kg]
M_vir = 5*10**10.0  # [M_sun] Viral mass
# const_c = 15.0  # [dimensionless] concentration of DM
const_c = aux.DMc_gravo(M_vir, 0, cfg.const_h)  # [dimensionless] concentration of DM
print('Dark matter concentration:', const_c)

# ###################### CREATING NFW CLASS ###################### #
"""
The initial profile of DM (we starting our evolution from that)
"""
NFW_profile = NFWProfile(M_vir, const_c)
r_s = NFW_profile.r_s
rounded_r_s = cfg.rounded_number(r_s, 3)

rho_s = NFW_profile.rho_s
rounded_rho_s = cfg.rounded_number(rho_s, 3)
print(f"rho_s = {rounded_rho_s} [M_solar / kpc^3], r_s = {rounded_r_s} [kpc]")

# ###################### .txt FILE: USER SETS! ###################### #
nums_r = 500  # for how many radius we want data in .txt  file
r_tilda_start = 0.01  # [dimensionless]
r_tilda_end = 100.0  # [dimensionless]
r_start = uni.convert_r_tilda(r_tilda_start, r_s)  # [kpc]
r_end = uni.convert_r_tilda(r_tilda_end, r_s)  # [kpc]
# list which contains radius
r = np.logspace(np.log10(r_start), np.log10(r_end), nums_r)  # (array) [kpc]

# Evolution time of galaxy
nums_time = 300  # how many we want to have time steps: to find merging
time_tilda_start = 6*10**(-1)  # [dimensionless]
time_tilda_end = 540.0  # [dimensionless]

time_start = uni.convert_time_tilda(time_tilda_start, rho_s, r_s, sigma_m)  # [Gyr]
time_end = uni.convert_time_tilda(time_tilda_end, rho_s, r_s, sigma_m)  # [Gyr]
# list which contains time
tage_grid = np.logspace(np.log10(time_start), np.log10(time_end), nums_time)  # (array) [kpc]
# tage_grid = np.linspace(time_start, time_end, nums_time)  # (array) [kpc]

diff_tim = len(tage_grid)  # how many times we set

# ---------------- AUTOMATIC SETTINGS ---------------- #
central_data = dict()
central_data["rho"] = np.array([])  # [M_sun/kpc^3]
central_data["velDis"] = np.array([])  # [kpc/Gyr]
rho0_LoDens_tilda_l = []  # [dimensionless]
rho0_HiDens_tilda_l = []  # [dimensionless]
time_tilda_l = []  # [dimensionless]

params_at_tmerge = {'Mvir': M_vir, 'c_const': const_c, 'cross_section': sigma_m}
params_at_tmerge['r1'] = 0.0  # [kpc], value of r1 when merging occurs
params_at_tmerge['t'] = 0.0  # merging time in [Gyr]
params_at_tmerge['t_tilda'] = 0.0  # merging time in [dimensionless]
params_at_tmerge['rho0_LoDens'] = 0.0  # fitted central density (low dense) in [M_sun/kpc^3]
params_at_tmerge['rho0_LoDens_tilda'] = 0.0  # fitted central density (low dense) in [dimensionless]
params_at_tmerge['sigma0_LoDens'] = 0.0  # fitted central velocity dispersion (low dense) in [kpc/Gyr]
params_at_tmerge['sigma0_LoDens_tilda'] = 0.0  # fitted central velDis (low dense) in [dimensionless]
params_at_tmerge['rho0_HiDens'] = 0.0  # fitted central density (high dense) in [M_sun/kpc^3]
params_at_tmerge['rho0_HiDens_tilda'] = 0.0  # fitted central density (high dense) in [dimensionless]
params_at_tmerge['sigma0_HiDens'] = 0.0  # fitted central velocity dispersion (high dense) in [kpc/Gyr]
params_at_tmerge['sigma0_HiDens_tilda'] = 0.0  # fitted central velDis (high dense) in [dimensionless]

merged_appear = False  # flag to find the first time when physical and unphysical have comparable values

# ###################### CREATING ISO_CDM CLASS ###################### #
r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage_grid[0])
ISO_CDM_class = IsoEvolution(NFW_profile, r_1)

# ###################### CREATING DICTIONARY WITH ALL DATA ###################### #
# dictionary which will contains all the data that we will use and save into .csv file
coreNFW_data = dict()
coreNFW_data['time'] = np.array([])  # radius [kpc]
coreNFW_data['r'] = np.array([])  # radius [kpc]
coreNFW_data["rho"] = np.array([])  # density [M_sun/kpc^3]
coreNFW_data["mass"] = np.array([])  # enclosed mass [M_sun]
coreNFW_data["velDis"] = np.array([])  # nu in [kpc/Gyr]


# ###################### LOOP OVER TIME ###################### #
for time in range(0, diff_tim):
    print("time:", time)
    # ###################### BASIC VARIABLES ###################### #
    tage = tage_grid[time]  # [Gyr] selected time
    sim_parms = [M_vir, const_c, sigma_m, tage]  # simulation parameters

    # -------------- FINDING R1 OF OUR PROFILE --------------#
    r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage)

    # ###################### ISOTHERMAL ###################### #
    ISO_CDM_class.new_evolution_step(r_1)  # one step in evolution find central density value

    # ---------------------- SAVE NECESSARY DATA: PLOT DENSITY EVOLUTION ---------------------- #
    # find the density values
    rhodm0_LoDens = ISO_CDM_class.retrun_rho0_LoDen()  # [M_sun/kpc^3]
    rhodm0_HiDens = ISO_CDM_class.return_rho0_HiDen()  # [M_sun/kpc^3]
    # calculate the tilda value
    rho0_LoDens_tilda = uni.rho_tilda(rhodm0_LoDens, rho_s)  # [dimensionless]
    rho0_HiDens_tilda = uni.rho_tilda(rhodm0_HiDens, rho_s)  # [dimensionless]
    # append to list
    rho0_LoDens_tilda_l.append(rho0_LoDens_tilda)  # [dimensionless]
    rho0_HiDens_tilda_l.append(rho0_HiDens_tilda)  # [dimensionless]

    # calculate time tilda time
    time_tilda = uni.time_tilda(tage, rho_s, r_s, sigma_m)  # [dimensionless]
    # append to list
    time_tilda_l.append(time_tilda)  # [dimensionless]

    # ---------------------- SEARCHING TMERGE: USER SETS! ---------------------- #
    Is_merging = aux.calRelErr_mergin(rho0_HiDens_tilda, rho0_LoDens_tilda, rel_err_margin)

    if (Is_merging is True) and merged_appear is False:
        # --- update vale of central data
        central_data["rho"] = np.append(central_data["rho"], rhodm0_LoDens)  # [M_sun/kpc^3]
        sigma0_LoDens = ISO_CDM_class.return_sigma0_LoDen()  # [kpc/Gyr]
        central_data["velDis"] = np.append(central_data["velDis"], sigma0_LoDens)  # [kpc/Gyr]

        # we find margin two densities
        merged_appear = True
        # --- values of parameters
        params_at_tmerge['r1'] = r_1  # [kpc]
        params_at_tmerge['t'] = tage  # [Gyr]
        params_at_tmerge['t_tilda'] = uni.time_tilda(tage, rho_s, r_s, sigma_m)  # [dimensionless]
        params_at_tmerge['rho0_LoDens'] = rhodm0_LoDens  # [M_sun/kpc^3]
        params_at_tmerge['rho0_LoDens_tilda'] = rho0_LoDens_tilda  # [dimensionless]
        params_at_tmerge['sigma0_LoDens'] = ISO_CDM_class.return_sigma0_LoDen()  # [kpc/Gyr]
        sigma0_LoDens_tilda = uni.nu_tilda(ISO_CDM_class.return_sigma0_LoDen(), r_s, rho_s)
        params_at_tmerge['sigma0_LoDens_tilda'] = sigma0_LoDens_tilda  # [dimensionless]
        params_at_tmerge['rho0_HiDens'] = rhodm0_HiDens  # [M_sun/kpc^3]
        params_at_tmerge['rho0_HiDens_tilda'] = rho0_HiDens_tilda  # [dimensionless]
        params_at_tmerge['sigma0_HiDens'] = ISO_CDM_class.return_sigma0_HiDen()  # [kpc/Gyr]
        sigma0_HiDens_tilda = uni.nu_tilda(ISO_CDM_class.return_sigma0_HiDen(), r_s, rho_s)
        params_at_tmerge['sigma0_HiDens_tilda'] = sigma0_HiDens_tilda  # [dimensionless]

        # --- print some important one
        print('r1 value during merging:', r_1, '[kpc]')
        print('time value during merging:', tage, '[Gyr]')

        # --- Find parameter `r_c`: model `core NFW`
        # ISO and NFW profile: get data (low dense)
        SIDM_LoDens = ISO_CDM_class.get_ISO_data_evolution()[0]
        ISO_data = dict()
        ISO_data["r"] = SIDM_LoDens[4]  # radius [kpc]
        ISO_data["rho"] = SIDM_LoDens[2]  # density [M_sun/kpc^3]
        ISO_data["mass"] = SIDM_LoDens[5]  # enclosed mass [M_sun]

        # Create `ISO and NFW` profile
        ISO_and_NFW_class = ISO_and_NFW([ISO_data["r"],
                                         ISO_data["rho"],
                                         ISO_data["mass"]],
                                        params_at_tmerge)

        # using fitting function find `r_c`
        r_c = find_fitting_parm(ISO_and_NFW_class, r)[0]

        # creating a class contains `core NFW`
        coreNFW_class = CoreNFWProfile(M_vir, const_c, r_c)

        # Here we stored data from `core NFW`
        coreNFW_data["time"] = np.append(coreNFW_data["time"], nums_r*[tage])  # time [Gyr]
        coreNFW_data["r"] = np.append(coreNFW_data["r"], r)  # radius [kpc]
        coreNFW_data["rho"] = np.append(coreNFW_data["rho"], coreNFW_class.rho(r))  # density [M_sun/kpc^3]
        coreNFW_data["mass"] = np.append(coreNFW_data["mass"], coreNFW_class.M(r))  # enclosed mass [M_sun]
        coreNFW_data["velDis"] = np.append(coreNFW_data["velDis"], coreNFW_class.sigma_accurate(r))  # nu in [kpc/Gyr]

    if (Is_merging is False) and merged_appear is False:
        # save proper data in each step: after finding the merging point we are not interested in saving
        # --- update vale of central data
        central_data["rho"] = np.append(central_data["rho"], rhodm0_LoDens)  # [M_sun/kpc^3]
        sigma0_LoDens = ISO_CDM_class.return_sigma0_LoDen()  # [kpc/Gyr]
        central_data["velDis"] = np.append(central_data["velDis"], sigma0_LoDens)  # [kpc/Gyr]

        # --- Find parameter `r_c`: model `core NFW`
        # ISO and NFW profile: get data (low dense)
        SIDM_LoDens = ISO_CDM_class.get_ISO_data_evolution()[0]
        ISO_data = dict()
        ISO_data["r"] = SIDM_LoDens[4]  # radius [kpc]
        ISO_data["rho"] = SIDM_LoDens[2]  # density [M_sun/kpc^3]
        ISO_data["mass"] = SIDM_LoDens[5]  # enclosed mass [M_sun]

        # --- values of parameters: really important we have take into account that r1 changes with time!!!
        params_at_tmerge['r1'] = r_1  # [kpc]

        # Create `ISO and NFW` profile
        ISO_and_NFW_class = ISO_and_NFW([ISO_data["r"],
                                         ISO_data["rho"],
                                         ISO_data["mass"]],
                                        params_at_tmerge)

        # using fitting function find `r_c`
        r_c = find_fitting_parm(ISO_and_NFW_class, r)[0]

        # creating a class contains `core NFW`
        coreNFW_class = CoreNFWProfile(M_vir, const_c, r_c)

        # Here we stored data from `core NFW`
        coreNFW_data["time"] = np.append(coreNFW_data["time"], nums_r*[tage])  # time [Gyr]
        coreNFW_data["r"] = np.append(coreNFW_data["r"], r)  # radius [kpc]
        coreNFW_data["rho"] = np.append(coreNFW_data["rho"], coreNFW_class.rho(r))  # density [M_sun/kpc^3]
        coreNFW_data["mass"] = np.append(coreNFW_data["mass"], coreNFW_class.M(r))  # enclosed mass [M_sun]
        coreNFW_data["velDis"] = np.append(coreNFW_data["velDis"], coreNFW_class.sigma_accurate(r))  # nu in [kpc/Gyr]

# ###################### AFTER LOOP: DRAW ###################### #
# Time evolution of central density in physical and unphysical case
# we still have this function here: to make sure that we do not get sth
# stupid in the end.
plotEvolutionRhoISO(params_at_tmerge,
                    time_tilda_l,
                    rho0_LoDens_tilda_l,
                    rho0_HiDens_tilda_l,
                    savePlot=False,
                    path_to_directory="")

# ###################### AFTER LOOP: SAVE DATA ###################### #
# --- add parameters
keys_list = list(params_at_tmerge.keys())
params_dict = {"name": [], "value": []}
for i in range(0, len(params_at_tmerge)):
    # append string: name of parameter
    str_par = keys_list[i]
    params_dict["name"].append(str_par)
    # read value of parameter
    val_par = params_at_tmerge[str_par]
    params_dict["value"].append(val_par)

print(params_dict)

# --- dictionary contains whole data
whole_data = []
for i in range(0, len(coreNFW_data["r"])):
    if i < len(params_dict["name"]):
        # important note: in txt file we use specific units,
        # so we have to change units
        whole_data.append({"time": np.log10(coreNFW_data["time"][i]),  # [log10(Gyr)]
                           "r": uni.r_tilda(coreNFW_data["r"][i], r_s),  # [dimensionless]
                           "rho": uni.rho_tilda(coreNFW_data["rho"][i], rho_s),  # [dimensionless]
                           "mass": uni.mass_tilde(coreNFW_data["mass"][i], rho_s, r_s),  # [dimensionless]
                           "velDis": uni.nu_txt(coreNFW_data["velDis"][i], r_s),  # [r_s/Gyr]
                           "central-rho": uni.rho_tilda(central_data["rho"][i], rho_s),  # [dimensionless]
                           "central-velDis": uni.nu_txt(central_data["velDis"][i], r_s),  # [r_s/Gyr]
                           "name": params_dict["name"][i],
                           "value": params_dict["value"][i]
                           })
    elif i < len(central_data["rho"]):
        whole_data.append({"time": np.log10(coreNFW_data["time"][i]),  # [log10(Gyr)]
                           "r": uni.r_tilda(coreNFW_data["r"][i], r_s),  # [dimensionless]
                           "rho": uni.rho_tilda(coreNFW_data["rho"][i], rho_s),  # [dimensionless]
                           "mass": uni.mass_tilde(coreNFW_data["mass"][i], rho_s, r_s),  # [dimensionless]
                           "velDis": uni.nu_txt(coreNFW_data["velDis"][i], r_s),  # [r_s/Gyr]
                           "central-rho": uni.rho_tilda(central_data["rho"][i], rho_s),  # [dimensionless]
                           "central-velDis": uni.nu_txt(central_data["velDis"][i], r_s),  # [r_s/Gyr]
                           })
    else:
        whole_data.append({"time": np.log10(coreNFW_data["time"][i]),  # [log10(Gyr)]
                           "r": uni.r_tilda(coreNFW_data["r"][i], r_s),  # [dimensionless]
                           "rho": uni.rho_tilda(coreNFW_data["rho"][i], rho_s),  # [dimensionless]
                           "mass": uni.mass_tilde(coreNFW_data["mass"][i], rho_s, r_s),  # [dimensionless]
                           "velDis": uni.nu_txt(coreNFW_data["velDis"][i], r_s)  # [r_s/Gyr]
                           })

# --- file name of .csv file
format_t = f'{"{:.3f}".format(params_at_tmerge["t"])}'
format_c = f'{"{:.3f}".format(const_c)}'
format_log10mass = f'{"{:.5f}".format(np.log10(M_vir))}'

file_name = f'Rcore_M_{format_log10mass}_t_{format_t}_sigma_{sigma_m}_con_{format_c}.csv'
file_path = 'data-at-tmerge' + '/' + file_name  # now we have to write the path to this file, where we want to save it

# --- save using DataFrame
df = pd.DataFrame(whole_data)
df.to_csv(file_path, sep='\t', header=False, index=False)
