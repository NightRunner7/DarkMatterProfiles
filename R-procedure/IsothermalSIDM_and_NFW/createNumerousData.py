"""
In this file we're interested in getting data till(!) a time, when the Rprocedure fails to describe evolution
of dark matter in galaxy. When and why this procedure failed at some time was mentioned in the
`IsothermalSIDMModel.py`. Shortly speaking at some time the unphysical solution become the physical one.
Thus, we evolve our system to that point and save the data, which describes the dark matter at that time
-> we will use that data as initial condition in the geothermal simulation thanks to which we get the
results farther in the future.

Important remark: in the ending step in the evolution we somehow combine the `Isothermal` and `NFW` profile
of dark matter. To better understanding see files: `IsoAndHalo.py` and `NFWProfile.py`. The profile of velocity
dispersion have discontinuity (discontinuity of the derivative at one point).

During some check I find out:
    1) time_tilda_start = t/t0: refers to dimensionless time. We can start from the arbitrary
    time, but in turns out: `time_tilda_start = 6*10**(-1)` is a good choice.
    2) rel_err_mergin: refers to accuracy of distinguishable `LoDen` and `HiDen` solution of R1-procedure.
    At now, we're not having the best guess, how should be fixed.
    3) nums_r: how many radi points we want in e.g. density profile (`rho(r)`) in each time step of Isothermal
    evolution. Usually all of such profiles will be stored and then saved in `.csv` file. But to save data
    in such a way, please see `createData.py`. Typically: `nums_r = 400`.
    4) r_low: `r_s * 10^r_low` is the smallest radi, which you want to consider / save in `.csv` file.
    Typically: `r_low=-2.0`.
    5) r_up: `r_s * 10^r_low` is the biggest radi, which you want to consider / save in `.csv` file.
     Typically: `r_up=2.0`.
    6) cff_to_Rres: `cff * r_s` is a spatial resolution. Typically: `cff=0.001`. The lower spatial resolution
    is the lower time of starting simulation we can take.
"""
# ################################### IMPORTING ################################### #
from timeit import default_timer as timer
import numpy as np
import pandas as pd
# ------------ FROM FILES ------------ #
from plotEvolutionRhoISO import plotEvolutionRhoISO
from NFWProfile import NFWProfile, r1  # CDM profile (halo)
from IsothermalSIDMModel import IsoEvolution  # ISO profile (halo) and their evolution with NFW
import config as cfg
import auxiliaryFunctions as aux
import units as uni

# ################################### SETS OF SIMULATION: USER SETS! ################################### #
# --- cosmological setting: list, what data you want create
LowDense = True  # differentiate Low dense or High dense: which data save.
savePlot = False  # save a plot, which presents evolution of central density of `LoDens` and `HiDens`
sigma_m_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]  # [cm^2/g]
M_vir_list = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]) * 10 ** 10  # [M_sun]
# --- find merging
rel_err_mergin = 1.0 + 20e-2  # see `calRelErr_mergin` in `auxiliaryFunctions.py`, 1.0 + 20e-2, 1.0 + 40e-2
rel_err_mergin_vol2 = 1.0 + 5e-2  # in Isothermal Evolution this should be lower than rel_err_mergin, 1.0 + 5e-2
# --- Space resolution
num_r_to_IsoEvo = 500  # 500, how many radius points during Isothermal evolution
cff_to_Rres = 0.001  # typically: 0.001
if LowDense is True:
    """
    If we want put Isothermal profile as initial in gravothermal simulation we shouldn't take care
    that our output data cannot reproduce central velocity dispersion for low time. Because
    we're interested that profile at tmerge is correct (for that setting this is true). To differentiate
    the difference go to `compare-to-gravothermal` directory.
    """
    r_low = -2.0  # 10**r_low [dimensionless]: lowest radi
    nums_r = 400  # 400, how many radius points we want data in `.csv`  file, for each time step
else:
    """
    For high dense solution we actually do care about accuracy of core density (in general we
    have to reproduce central density by using data - core density).We actually care, because of
    the `Mirror method` were we reverse the time - so the beginning of the high dense became
    a results around collapse.
    """
    r_low = -4.0  # 10**r_low [dimensionless]: lowest radi
    nums_r = 600  # 600, how many radius points we want data in `.csv`  file, for each time step
r_up = 2.0  # 10**r_up [dimensionless]: biggest radi
# --- Evolution time of galaxy
# dimensionless time
nums_time_1 = 200  # 200, 300, 50 how many we want to have time steps: to find merging
nums_time_2 = 200  # 200, 100, 0
time_tilda_start = 6*10**(-1)  # [dimensionless]
time_tilda_end_1 = 300.0  # 300, 500 [dimensionless]
time_tilda_end_2 = 500.0  # 500, 500 [dimensionless]
# --- where save `.csv` file
directory_to_save = '.'

for M_vir in M_vir_list:
    # iteration for each viral mass
    for sigma_m in sigma_m_list:
        print(f">>>>>>>>>>>>> I start working on sigma_m = {'{:.1f}'.format(sigma_m)} and "
              f"log10(M_vir) = {'{:.5f}'.format(np.log10(M_vir))}")
        # iteration for each cross-section

        # --- set pair of cosmological setting
        const_c = aux.DMc_gravo(M_vir, 0, cfg.const_h)  # [dimensionless] concentration of DM
        # ################################### CREATING NFW CLASS ################################### #
        """
        The initial profile of DM (we starting our evolution from that)
        """
        NFW_profile = NFWProfile(M_vir, const_c)
        r_s = NFW_profile.r_s
        rho_s = NFW_profile.rho_s

        # ################################### TIME LIST ################################### #
        # dimensional time
        time_start = uni.convert_time_tilda(time_tilda_start, rho_s, r_s, sigma_m)  # [Gyr]
        time_end_1 = uni.convert_time_tilda(time_tilda_end_1, rho_s, r_s, sigma_m)  # [Gyr]
        time_end_2 = uni.convert_time_tilda(time_tilda_end_2, rho_s, r_s, sigma_m)  # [Gyr]

        # list which contains time
        tage_grid_log = np.logspace(np.log10(time_start), np.log10(time_end_1),
                                    nums_time_1, endpoint=False)  # (array) [kpc]
        tage_grid_lin = np.linspace(time_end_1, time_end_2, nums_time_2)  # (array) [kpc]
        tage_grid = [*tage_grid_log, *tage_grid_lin]
        tage_grid.sort()

        diff_tim = len(tage_grid)  # how many times we set

        # ---------------- AUTOMATIC SETTINGS ---------------- #
        rho0_LoDens_tilda_l = []  # [dimensionless]
        rho0_HiDens_tilda_l = []  # [dimensionless]
        time_tilda_l = []  # [dimensionless]

        merged_appear = False  # flag to find the first time when physical and unphysical have comparable values

        # ################################### CREATING ISO_CDM CLASS ################################### #
        r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage_grid[0])
        IsoEvolution_class = IsoEvolution(NFW_profile, r_1, cff=cff_to_Rres, nr=num_r_to_IsoEvo,
                                          rel_err_mergin=rel_err_mergin_vol2)

        # ################################### LOOP OVER TIME ################################### #
        for time in range(0, diff_tim):
            # print("time:", time)
            # ###################### BASIC VARIABLES ###################### #
            tage = tage_grid[time]  # [Gyr] selected time
            sim_parms = [M_vir, const_c, sigma_m, tage]  # simulation parameters

            # -------------- FINDING R1 OF OUR PROFILE --------------#
            r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=tage)

            # ###################### ISOTHERMAL ###################### #
            IsoEvolution_class.new_evolution_step(r_1)  # one step in evolution find central density value

            # ---------------------- SAVE NECESSARY DATA: PLOT DENSITY EVOLUTION ---------------------- #
            # find the density values
            rhodm0_LoDens = IsoEvolution_class.retrun_rho0_LoDen()  # [M_sun/kpc^3]
            rhodm0_HiDens = IsoEvolution_class.return_rho0_HiDen()  # [M_sun/kpc^3]
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
            Is_merging = aux.calRelErr_mergin(rho0_HiDens_tilda, rho0_LoDens_tilda, rel_err_mergin)

            if (Is_merging is True) and merged_appear is False:
                # --- we find merging two densities
                merged_appear = True

                # --- update data at merging
                IsoEvolution_class.update_at_merging(tage, sigma_m,
                                                     nr=nums_r, r_low=r_low, r_up=r_up,
                                                     lowDens=LowDense)

            if (Is_merging is False) and merged_appear is False:
                # save proper data in each step: after finding the merging point we are not interested in saving
                # --- update vale of central data
                IsoEvolution_class.update_before_merging(tage,
                                                         nr=nums_r, r_low=r_low, r_up=r_up,
                                                         lowDens=LowDense)

        # ###################### AFTER LOOP: COLLECT ###################### #
        # --- output data
        params_at_tmerge = IsoEvolution_class.return_params_at_tmerge()
        IsoAndNFW_data = IsoEvolution_class.return_IsoAndNFW_data()
        # --- Draw
        # Time evolution of central density in physical and unphysical case
        # we still have this function here: to make sure that we do not get sth
        # stupid in the end.
        if savePlot is True:
            plotEvolutionRhoISO(params_at_tmerge,
                                time_tilda_l,
                                rho0_LoDens_tilda_l,
                                rho0_HiDens_tilda_l,
                                savePlot=savePlot,
                                path_to_directory=directory_to_save)

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
        for i in range(0, len(IsoAndNFW_data["r"])):
            if i < len(params_dict["name"]):
                # important note: in txt file we use specific units,
                # so we have to change units
                whole_data.append({"time": np.log10(IsoAndNFW_data["time"][i]),  # [log10(Gyr)]
                                   "r": uni.r_tilda(IsoAndNFW_data["r"][i], r_s),  # [dimensionless]
                                   "rho": uni.rho_tilda(IsoAndNFW_data["rho"][i], rho_s),  # [dimensionless]
                                   "mass": uni.mass_tilde(IsoAndNFW_data["mass"][i], rho_s, r_s),  # [dimensionless]
                                   "velDis": uni.nu_txt(IsoAndNFW_data["velDis"][i], r_s),  # [r_s/Gyr]
                                   "central-rho": uni.rho_tilda(IsoAndNFW_data["central-rho"][i], rho_s),  # [dimensionless]
                                   "central-velDis": uni.nu_txt(IsoAndNFW_data["central-velDis"][i], r_s),  # [r_s/Gyr]
                                   "name": params_dict["name"][i],
                                   "value": params_dict["value"][i]
                                   })
            elif i < len(IsoAndNFW_data["central-rho"]):
                whole_data.append({"time": np.log10(IsoAndNFW_data["time"][i]),  # [log10(Gyr)]
                                   "r": uni.r_tilda(IsoAndNFW_data["r"][i], r_s),  # [dimensionless]
                                   "rho": uni.rho_tilda(IsoAndNFW_data["rho"][i], rho_s),  # [dimensionless]
                                   "mass": uni.mass_tilde(IsoAndNFW_data["mass"][i], rho_s, r_s),  # [dimensionless]
                                   "velDis": uni.nu_txt(IsoAndNFW_data["velDis"][i], r_s),  # [r_s/Gyr]
                                   "central-rho": uni.rho_tilda(IsoAndNFW_data["central-rho"][i], rho_s),  # [dimensionless]
                                   "central-velDis": uni.nu_txt(IsoAndNFW_data["central-velDis"][i], r_s),  # [r_s/Gyr]
                                   })
            else:
                whole_data.append({"time": np.log10(IsoAndNFW_data["time"][i]),  # [log10(Gyr)]
                                   "r": uni.r_tilda(IsoAndNFW_data["r"][i], r_s),  # [dimensionless]
                                   "rho": uni.rho_tilda(IsoAndNFW_data["rho"][i], rho_s),  # [dimensionless]
                                   "mass": uni.mass_tilde(IsoAndNFW_data["mass"][i], rho_s, r_s),  # [dimensionless]
                                   "velDis": uni.nu_txt(IsoAndNFW_data["velDis"][i], r_s)  # [r_s/Gyr]
                                   })

        # --- file name of .csv file
        format_t = f'{"{:.3f}".format(params_at_tmerge["t"])}'
        format_c = f'{"{:.3f}".format(const_c)}'
        format_log10mass = f'{"{:.5f}".format(np.log10(M_vir))}'
        if LowDense is True:
            file_name = f'Riso_M_{format_log10mass}_t_{format_t}_sigma_{sigma_m}_con_{format_c}.csv'
        else:
            file_name = f'RisoHiDens_M_{format_log10mass}_t_{format_t}_sigma_m_{sigma_m}_con_{format_c}.csv'
        # now we have to write the path to this file, where to save it
        file_path = directory_to_save + '/' + file_name

        # --- save using DataFrame
        df = pd.DataFrame(whole_data)
        df.to_csv(file_path, sep='\t', header=False, index=False)
