"""
This file contains all method to investigate the data coming from the `gravothermal` simulation.
We try our best to get the results in the same manner, regardless of the initial condition, which
we set into these simulations: for instances we use as initial condition: `pure NFW`, `NFW and ISO`
and `core NFW`.

`pure NFW`: initial density profile of dark matter is determined by just NFW profile.
`NFW and ISO` the initial density profile of dark matter is determined after applying the R-procedure.
So, we take into accounts the Isothermal evolution of dark matter. After R-procedure fails we switch
into gravothermal.
`core NFW`: the initial density profile of dark matter is determined after applying the R-procedure.
So, we take into accounts the Isothermal evolution of dark matter. Moreover, we fit our results
(from Isothermal) to the `core NFW` model - more details in R-procedure folder. After R-procedure fails
we switch into gravothermal.
"""
import numpy as np
import sys
import pandas as pd
# --- IMPORT FROM FILES
import config as cfg
import auxiliaryFunctions as aux
# import units as uni  # units
from scipy.interpolate import PchipInterpolator

# --- FUNCTION WHICH WE USED IN NFW (pure NFW)
def f(x):
    """
    Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)

    Syntax:
        .f(x)

    where
        x: dimensionless radius r/r_s (float or array)
    """
    return np.log(1. + x) - x / (1. + x)

# -------------------------- DEAL WITH GravothermalData ----------------------------------- #
def create_gravothermalData_from_file(file_name, localization_of_file, beta,
                                      one_file=True,
                                      veldis_file_name=None, localization_of_veldis_file=None):
    """
    Creates a GravothermalData object from one or two CSV files.

    Parameters:
    - file_name: str
        The name of the CSV file containing gravothermal data.
    - localization_of_file: str
        The directory where the main CSV file is located.
    - beta: float
        Scaling parameter in the gravothermal model.
    - one_file: bool, optional, default=True
        If True, all data is stored in a single file.
        If False, data is split into two files: one for density and one for velocity dispersion.
    - veldis_file_name: str, optional
        The name of the CSV file containing velocity dispersion data (if data is split into two files).
    - localization_of_veldis_file: str, optional
        The directory where the velocity dispersion CSV file is located (if data is split into two files).

    Returns:
    - gravoEvolution: GravothermalData
        An object containing the parsed gravothermal data.
    """
    # If all data is in one file
    if one_file:
        # Load the single CSV file containing time, radius, density, and velocity dispersion
        df_gravothermal = aux.load_csv_from_dir(localization_of_file, file_name,
                                                sep='\t', names=['t', 'r', 'rho', 'velDis'], low_memory=False)
        # Convert all columns to numeric, dropping any rows with invalid (non-numeric) data
        df_gravothermal = df_gravothermal.apply(pd.to_numeric, errors='coerce').dropna()

        # Create a GravothermalData object using the loaded data
        gravoEvolution = GravothermalData(df_gravothermal['t'],
                                          df_gravothermal['r'],
                                          df_gravothermal['rho'],
                                          df_gravothermal['velDis'],
                                          beta=beta)
    else:
        # Load the CSV file containing time, radius, and density
        df_gravothermal_rho = aux.load_csv_from_dir(localization_of_file, file_name,
                                                    sep='\t', names=['t', 'r', 'rho'], low_memory=False)
        # Convert all columns to numeric, dropping any rows with invalid (non-numeric) data
        df_gravothermal_rho = df_gravothermal_rho.apply(pd.to_numeric, errors='coerce').dropna()

        # Load the CSV file containing time, radius, and velocity dispersion
        df_gravothermal_sigma = aux.load_csv_from_dir(localization_of_veldis_file, veldis_file_name,
                                                      sep='\t', names=['t', 'r', 'velDis'], low_memory=False)
        # Convert all columns to numeric, dropping any rows with invalid (non-numeric) data
        df_gravothermal_sigma = df_gravothermal_sigma.apply(pd.to_numeric, errors='coerce').dropna()

        # Create a GravothermalData object using the loaded data
        gravoEvolution = GravothermalData(df_gravothermal_rho['t'],
                                          df_gravothermal_rho['r'],
                                          df_gravothermal_rho['rho'],
                                          df_gravothermal_sigma['velDis'],
                                          beta=beta)
    return gravoEvolution

# ########################### FUNCTION TO RETURN DATA FROM FILE ########################### #
class GravothermalData(object):
    """
    Class, which will stored all the necessary data fom gravothermal simulation.
    """
    def __init__(self, time_list, r_list, rho_list, veldis_list, beta: float):
        """
        Initialize class to stored data.

        time_list: array, which contains time data from .csv file.
            (array or list) [log10(Gyr)]
        r_list: array, which contains radius's from .csv file.
            (array or list) [r_s]
        rho_list: array, which contains density from .csv file.
            (array or list) [rho_s]
        veldis_list: array, which contains velocity dispersion from .csv file.
            (array or list) [r_s/Gyr]
        beta: adjustable parameter to match the fluid numerical solutions with N-body
              or gravothermal simulations.
              (float) [dimensionless]

        We will use dictionary, which will stored all necessary parameters:

            parameters["Mvir"]: halo mass [M_sun]
            parameters["c_const"]: halo concentration [dimensionless]
            parameters["time-end"]: time when the Rprocedure fails [Gyr]
            parameters["sigma_m"]: annihilation cross-section [cm^2/g]
            parameters["Deltah"]: spherical overdensity with respect to the critical
                density of the universe (default is 200.) [kpc]
            parameters["z"]: redshift
            parameters["rhoc"]: critical density [M_sun kpc^-3]
            parameters["rhoh"]: average density of halo [M_sun kpc^-3]
            parameters["rh"]: halo radius within which density is Delta times rhoc [kpc]
            parameters["r_s"]: scale radius [kpc]
            parameters["rho_s"]: scale density [M_sun kpc^-3]

        """
        self.beta = beta  # set the parameter beta
        self.name = None  # to differentiate the data stored in class
        # additional parameters
        self.parameters = None
        # the basic information
        self.data = dict()
        self.data["time"] = np.array([10**x for x in time_list])  # CHECK IT! [log10(Gyr)]
        time_no_repetition = list(dict.fromkeys(self.data["time"]))
        self.data["time-no-repetition"] = np.array(time_no_repetition)
        self.data["r"] = np.array(r_list)  # [r_s]
        self.data["rho"] = np.array(rho_list)  # [rho_s]
        self.data["velDis"] = np.array(veldis_list)  # [r_s/Gyr]

        # to parametrise the time
        len_t = 0
        while self.data["time"][len_t] == self.data["time"][0]:
            len_t += 1
        self.number_radi = len_t  # Same values of time - how many radius values we have for each one fixed time
        self.time_steps = int(len(self.data["time"]) / len_t)  # how many steps on time is in the .txt file

    # -------------------------------- INPUT SOME INFO -------------------------------- #
    def put_the_name(self, name):
        """
        This function we use to put the name to somehow differentiate, which data is stored
        in class. For instances, we may have:

        name == 'NFW' -> those class stored data, created in gravothermal simulation with
            NFW initial profile.
        name == 'Isothermal' -> those class stored data, created in gravothermal simulation
            with initial profile coming from Rprocedure (without fitting).
        name == 'coreNFW' -> those class stored data, created in gravothermal simualtion
            with initial profile coming from Rprocedure (with fitting to the `core NFW` profile).

        :param name: name to differentiate the data, stored in class
        """
        self.name = name

    def put_extra_parameters(self, M_vir, sigma_m, z=0.0, Delta=200.0):
        """
        Maybe sometime it will be necessary to put for instances extra parameters of
        our simulation. This extra information we can use for example to get different
        units: change [rho] = [rho_s] -> [M_sun / kpc^3]

        Mvir: value of viral mass.
            (float) [M_sun]
        sigma_m: annihilation cross-section.
            (float) [cm^2/g]
        Delta: spherical overdensity with respect to the critical
            density of the universe (default is 200.)
            (float) [kpc]
         z: redshift (float)
        """
        self.parameters = dict()
        self.parameters["Deltah"] = Delta  # [kpc]
        self.parameters["z"] = z

        self.parameters["Mvir"] = M_vir  # [M_sun]
        self.parameters["const_c"] = aux.DMc_gravo(M_vir, 0, cfg.const_h)  # [dimensionless]
        self.parameters["sigma_m"] = sigma_m  # [cm^2/g]

        self.parameters["rhoc"] = cfg.rho_c  # [M_sun kpc^-3], not best: THINK!
        self.parameters["rhoh"] = self.parameters["Deltah"] * self.parameters["rhoc"]  # [M_sun kpc^-3]
        self.parameters["rh"] = (3. * self.parameters["Mvir"] / (cfg.FourPi * self.parameters["rhoh"])) ** (
                    1. / 3.)  # [kpc]
        self.parameters["r_s"] = self.parameters["rh"] / self.parameters["const_c"]  # [kpc]
        self.parameters["rho_s"] = self.parameters["rhoc"] * self.parameters["Deltah"] / 3. * \
                                   self.parameters["const_c"] ** 3. / f(self.parameters["const_c"])  # [M_sun / kpc^3]

    # -------------------------------- RETURN PARAMETERS -------------------------------- #
    def return_basic_parameters(self):
        """Return Mvir and sigma_m and dark mater concentration"""
        return [self.parameters["Mvir"], self.parameters["sigma_m"], self.parameters["const_c"]]

    # -------------------------------- CORE DATA -------------------------------- #
    def return_rho_core_evolution(self, elements=2, index=1, calculate_rho_core=False):
        """
        Find the central density (or core density) value during time evolution of the galaxy.

        param: elements: how many first radius's we treat as belonging to a core.
        (int)
        """
        # deal with core density
        rho_core_list = np.array([])


        for i in range(0, self.time_steps):
            left_arg = i * self.number_radi
            right_arg = i * self.number_radi + elements
            # data which lies inside the core
            r_in_core = self.data["r"][left_arg:(right_arg + 1)]
            rho_in_core = self.data["rho"][(0 + left_arg):(0 + right_arg)]  # bez
            if calculate_rho_core:
                # calculate the value of the core
                rho_core = aux.calculate_rho_core(r_in_core, rho_in_core, elements)
            else:
                rho_core=rho_in_core[index]
            # appending to list contains all values of core density
            rho_core_list = np.append(rho_core_list, rho_core)

        return [self.data["time-no-repetition"], rho_core_list]

    def return_veldis_core_evolution(self, elements=1):
        """
        Find the central velocity dispersion (or core velocity dispersion) value during time
        evolution of the galaxy.

        param: elements: how many first radius's we treat as belonging to a core.
        (int)
        """
        # deal with core velocity dispersion
        veldis_core_list = np.array([])

        for i in range(0, self.time_steps):
            left_arg = i * self.number_radi
            right_arg = i * self.number_radi + elements
            # data which lies inside the core
            r_in_core = self.data["r"][left_arg:(right_arg + 1)]
            veldis_in_core = self.data["velDis"][(0 + left_arg):(0 + right_arg)]  # bez przesunięcia
            # calculate the value of the core
            veldis_core = aux.calculate_veldis_core(r_in_core, veldis_in_core, elements)
            # appending to list contains all values of core density
            veldis_core_list = np.append(veldis_core_list, veldis_core)

        return [self.data["time-no-repetition"], veldis_core_list]

    import numpy as np

    def rc_from_profile(self, r, rho, rho_core, *, use_logrho=True):
        """
        Compute rc such that rho(rc) = rho_core/2, using the full profile.
        Returns np.nan if target not reached in the provided r-range.
        """
        r = np.asarray(r)
        rho = np.asarray(rho)

        target = 0.5 * rho_core

        # If already below target at the innermost bin, core is unresolved (smaller than r[0])
        if rho[0] <= target:
            return np.nan

        # Find first index where rho falls below target
        j = np.where(rho <= target)[0]
        if j.size == 0:
            return np.nan  # never drops to half within grid

        j = j[0]  # first time it goes below
        if j == 0:
            return np.nan

        r1, r2 = r[j - 1], r[j]
        y1, y2 = rho[j - 1], rho[j]

        # Interpolate to get rc
        if use_logrho and (y1 > 0) and (y2 > 0) and (target > 0):
            ly1, ly2, lt = np.log(y1), np.log(y2), np.log(target)
            t = (lt - ly1) / (ly2 - ly1)
            return r1 + t * (r2 - r1)
        else:
            t = (target - y1) / (y2 - y1)
            return r1 + t * (r2 - r1)

    def return_rc_evolution(self, elements=2, use_logrho=True):
        """
        Return [time, rc] where rc is defined by rho(rc)=rho_core/2.
        rho_core is estimated from the inner-most bins using `elements`.
        """
        time_arr = np.asarray(self.data["time-no-repetition"])
        rc_arr = np.empty(self.time_steps, dtype=float)

        for i in range(self.time_steps):
            left = i * self.number_radi
            right = (i + 1) * self.number_radi

            r_snap = np.asarray(self.data["r"][left:right])
            rho_snap = np.asarray(self.data["rho"][left:right])

            # your core density estimator (inner bins only)
            r_in_core = r_snap[:elements + 1]
            rho_in_core = rho_snap[:elements]  # adjust if aux expects different slicing
            rho_core = aux.calculate_rho_core(r_in_core, rho_in_core, elements)

            # physical core size from half-density radius (global scan!)
            rc_arr[i] = self.rc_from_profile(r_snap, rho_snap, rho_core, use_logrho=use_logrho)

        return [time_arr, rc_arr]

    # -------------------------------- LAST MOMENT OF FORMING THE CORE -------------------------------- #
    def find_min_rho_core(self, elements=2):
        """
        Find the moment and also the value of minimal central density in our data.

        :param elements: how many first radius's we treat as belonging to a core.
        (int)
        :return: [minimal core density, time when minimal core is occurring, time step of that]
        """
        # the minimum density of core
        min_rho_core = self.data["rho"][0]
        time_min_core = 0
        time_step_min_core = 0

        for i in range(0, self.time_steps):
            left_arg = i * self.number_radi
            right_arg = i * self.number_radi + elements
            # data which lies inside the core
            r_in_core = self.data["r"][left_arg:(right_arg + 1)]
            rho_in_core = self.data["rho"][left_arg:right_arg]
            # calculate the value of the core
            value_of_rho_core = aux.calculate_rho_core(r_in_core, rho_in_core, elements)

            if min_rho_core > value_of_rho_core:
                min_rho_core = value_of_rho_core
                time_min_core = self.data["time-no-repetition"][i]
                time_step_min_core = i

        return [min_rho_core, time_min_core, time_step_min_core]

    def return_data_last_moment_forming(self, elements=2):
        """
        Return all data, which describes the moment, when core starts to contracting.
        In the different words, after that moment core will be shrinking and
        starts collapsing.

        :param elements: how many first radius's we treat as belonging to a core.
        (int)
        """
        # find the proper time step to collapse
        time_step_collapse = self.find_min_rho_core(elements=elements)[2]

        # arguments
        left_arg = time_step_collapse * self.number_radi
        right_arg = (time_step_collapse + 1) * self.number_radi - 1

        # data
        data_last_forming = dict()
        data_last_forming["time-step"] = time_step_collapse
        data_last_forming["time"] = self.find_min_rho_core(elements=elements)[1]
        data_last_forming["r"] = self.data["r"][left_arg:right_arg]
        data_last_forming["rho"] = self.data["rho"][left_arg:right_arg]
        data_last_forming["velDis"] = self.data["velDis"][left_arg:right_arg]

        return data_last_forming

    # -------------------------------- MOMENT OF CORE COLLAPSING -------------------------------- #
    def return_veldis_hat_profile(
            self,
            time_argument=None,
            *,
            time_step_bool=True,
            elements_core=2,
            M0=None,
            return_time_step=False,
    ):
        """
        Return (r, nu_hat) at a selected time.

        nu_hat(r) = nu(r) / nu0
        where nu0 = sqrt(G M0 / r_s).

        Parameters
        ----------
        time_argument : int or float or None
            If time_step_bool=True: interpreted as time_step (int).
            If time_step_bool=False: interpreted as physical time [Gyr] and nearest step is used.
            If None: uses the time step where rho_core is minimal (from find_min_rho_core()).
        time_step_bool : bool
            Controls interpretation of time_argument.
        elements_core : int
            Passed to find_min_rho_core when time_argument is None (your definition of core density).
        M0 : float or None
            Mass scale used in nu0. Default: self.parameters["Mvir"].
        return_time_step : bool
            If True, also return the chosen time_step and time [Gyr].

        Returns
        -------
        (r, nu_hat) or (r, nu_hat, time_step, time_value)
            r is in [r_s] (your stored units), nu_hat is dimensionless.
        """
        if self.parameters is None:
            raise RuntimeError("Call put_extra_parameters(M_vir, sigma_m, ...) first.")

        # pick time step
        if time_argument is None:
            time_step = self.find_min_rho_core(elements=elements_core)[2]
            time_value = self.data["time"][time_step * self.number_radi]  # [Gyr]
        else:
            if time_step_bool:
                time_step = int(time_argument)
                time_value = self.data["time"][time_step * self.number_radi]  # [Gyr]
            else:
                time_value, time_step = aux.find_nearest_and_index(
                    self.data["time-no-repetition"], time_argument
                )

        # slice snapshot (use your own convention; I’ll mirror return_data_at_fixed_time)
        left = time_step * self.number_radi
        right = (time_step + 1) * self.number_radi - 1

        r_snap = self.data["r"][left:right]  # [r_s]
        nu_snap = self.data["velDis"][left:right]  # [r_s/Gyr]

        nu0 = self._nu0_rs_per_gyr(M0=M0)  # [r_s/Gyr]
        nu_hat = nu_snap / nu0

        if return_time_step:
            return r_snap, nu_hat, time_step, time_value
        return r_snap, nu_hat, time_step

    # -------------------------------- MOMENT OF CORE COLLAPSING -------------------------------- #
    def find_collapse(self, elements=2, fixed_limit=10**10):
        """
        Find the time and also the value of central density when collapsing occurred.

        :param elements: how many first radius's we treat as belonging to a core.
        (int)
        :param fixed_limit: fixed value to differentiate the core collapse.
        (int)
        :return: [core density, time when collapse is occurring, time step of that]
        """
        find_collapse = False

        for i in range(0, self.time_steps):
            argument = i * self.number_radi
            density = self.data["rho"][argument]

            if density > fixed_limit:
                find_collapse = True
                # we find when collapse occurred
                time_step_collapse = i
                time_collapse = self.data["time"][i * self.number_radi]

                # the value of core density at collapse
                left_arg = i * self.number_radi
                right_arg = i * self.number_radi + elements
                # data which lies inside the core
                r_in_core = self.data["r"][left_arg:(right_arg + 1)]
                rho_in_core = self.data["rho"][left_arg:right_arg]
                # calculate the value of the core
                value_of_rho_core = aux.calculate_rho_core(r_in_core, rho_in_core, elements)

                return [value_of_rho_core, time_collapse, time_step_collapse]

        if find_collapse is False:
            sys.exit("We don't find a collapse. Maybe change value of `fixed_limit`.")

    # -------------------------------- DATA AT FIXED TIME -------------------------------- #
    def return_data_at_fixed_time(self, time_argument, time_step_bool=True):
        """
        Return all necessary data at fixed time.

        param: time_step_bool: if is a `True` it means the `time_argument` is given as `time_step`.
            Otherwise, if you put value of time it will be searching for the closes `time_step`
            to the value, which you put.
        param: time_argument: can differentiate, but it is necessary argument to find the\
            evolution profile of our interest.
        """
        if time_step_bool is True:
            time_step = time_argument
            time_value = self.data['time'][time_step * self.number_radi]  # [Gyr]
        else:
            time_value, time_step = aux.find_nearest_and_index(self.data["time-no-repetition"], time_argument)

        # arguments
        left_arg = time_step * self.number_radi
        right_arg = (time_step + 1) * self.number_radi - 1

        # data
        data_at_fixed_time = dict()
        data_at_fixed_time["time-step"] = time_step
        data_at_fixed_time["time"] = time_value  # [Gyr]
        data_at_fixed_time["r"] = self.data["r"][left_arg:right_arg]
        data_at_fixed_time["rho"] = self.data["rho"][left_arg:right_arg]
        data_at_fixed_time["velDis"] = self.data["velDis"][left_arg:right_arg]

        return data_at_fixed_time

    # -------------------------------- CONVER UNITS -------------------------------- #
    def _nu0_rs_per_gyr(self, M0=None):
        """
        Return nu0 in the same units as self.data['velDis'], i.e. [r_s / Gyr].

        nu0 = sqrt(G M0 / r_s)   (physical: kpc/Gyr)
        convert to [r_s/Gyr] by dividing by r_s:  nu0_hat_units = nu0 / r_s
        """
        if self.parameters is None:
            raise RuntimeError("Call put_extra_parameters(...) before using nu0 conversions.")

        if M0 is None:
            M0 = self.parameters["Mvir"]

        r_s = self.parameters["r_s"]  # [kpc]
        nu0_phys = np.sqrt(cfg.const_G_starUnits * M0 / r_s)  # [kpc/Gyr]
        nu0_in_rs_per_gyr = nu0_phys / r_s  # [(kpc/Gyr)/kpc] = [1/Gyr]?? -> numerically equals [r_s/Gyr] scale
        # Interpreted as "how many r_s per Gyr"
        return nu0_in_rs_per_gyr

    # --------------------------------------------------------------------------- #
    def _reshape_gravo_profiles(self):
        """
        Reshape flat stored arrays into 2D arrays:
            rho[t, r], velDis[t, r]

        Returns
        -------
        time_arr : ndarray
            Time grid [Gyr]
        r_arr : ndarray
            Radius grid [r_s]
        rho_2d : ndarray
            Density profiles [rho_s]
        vel_2d : ndarray
            Velocity-dispersion profiles [r_s/Gyr]
        """
        n_t = self.time_steps
        n_r = self.number_radi

        time_arr = np.asarray(self.data["time-no-repetition"], dtype=float)
        r_arr = np.asarray(self.data["r"][:n_r], dtype=float)

        rho_2d = np.asarray(self.data["rho"], dtype=float).reshape(n_t, n_r)
        vel_2d = np.asarray(self.data["velDis"], dtype=float).reshape(n_t, n_r)

        return time_arr, r_arr, rho_2d, vel_2d

    def _get_snapshot_arrays(self, time_argument, time_step_bool=True, drop_r_zero=True):
        """
        Return one snapshot arrays: r, rho, velDis.

        Parameters
        ----------
        time_argument : int or float
            If time_step_bool=True, interpreted as time-step index.
            Otherwise interpreted as physical time [Gyr], nearest snapshot chosen.
        time_step_bool : bool
            Interpret time_argument as step index if True.
        drop_r_zero : bool
            If True, remove r <= 0 entries.

        Returns
        -------
        time_value : float
            Snapshot time [Gyr]
        time_step : int
            Snapshot index
        r_snap : ndarray
            Radius [r_s]
        rho_snap : ndarray
            Density [rho_s]
        vel_snap : ndarray
            Velocity dispersion [r_s/Gyr]
        """
        time_arr, r_arr, rho_2d, vel_2d = self._reshape_gravo_profiles()

        if time_step_bool:
            time_step = int(time_argument)
            time_value = time_arr[time_step]
        else:
            time_value, time_step = aux.find_nearest_and_index(time_arr, time_argument)

        r_snap = np.asarray(r_arr, dtype=float)
        rho_snap = np.asarray(rho_2d[time_step, :], dtype=float)
        vel_snap = np.asarray(vel_2d[time_step, :], dtype=float)

        if drop_r_zero:
            mask = r_snap > 0.0
            r_snap = r_snap[mask]
            rho_snap = rho_snap[mask]
            vel_snap = vel_snap[mask]

        return time_value, time_step, r_snap, rho_snap, vel_snap

    def build_snapshot_interpolators(
            self,
            time_argument,
            *,
            time_step_bool=True,
            drop_r_zero=True,
            rho_mode="loglog",
            vel_mode="loglog",
    ):
        """
        Build radial interpolators for one snapshot.

        Parameters
        ----------
        time_argument : int or float
            Snapshot selector.
        time_step_bool : bool
            If True, time_argument is time-step index.
        drop_r_zero : bool
            If True, exclude r=0 from interpolation.
        rho_mode : str
            'loglog' or 'logx_linear_y'
        vel_mode : str
            'loglog' or 'logx_linear_y'

        Returns
        -------
        out : dict
            Contains snapshot arrays and interpolation callables.
        """
        time_value, time_step, r_snap, rho_snap, vel_snap = self._get_snapshot_arrays(
            time_argument,
            time_step_bool=time_step_bool,
            drop_r_zero=drop_r_zero,
        )

        # basic validity masks
        rho_mask = np.isfinite(r_snap) & np.isfinite(rho_snap) & (r_snap > 0) & (rho_snap > 0)
        vel_mask = np.isfinite(r_snap) & np.isfinite(vel_snap) & (r_snap > 0) & (vel_snap > 0)

        r_rho = r_snap[rho_mask]
        rho_use = rho_snap[rho_mask]

        r_vel = r_snap[vel_mask]
        vel_use = vel_snap[vel_mask]

        if len(r_rho) < 2:
            raise ValueError("Not enough valid density points to build interpolator.")
        if len(r_vel) < 2:
            raise ValueError("Not enough valid velocity-dispersion points to build interpolator.")

        log_r_rho = np.log(r_rho)
        log_r_vel = np.log(r_vel)

        # --- density interpolator
        if rho_mode == "loglog":
            log_rho = np.log(rho_use)
            rho_interp_raw = PchipInterpolator(log_r_rho, log_rho, extrapolate=False)

            def rho_interp(r):
                r = np.asarray(r, dtype=float)
                out = np.full_like(r, np.nan, dtype=float)
                mask = r > 0
                out[mask] = np.exp(rho_interp_raw(np.log(r[mask])))
                return out

        elif rho_mode == "logx_linear_y":
            rho_interp_raw = PchipInterpolator(log_r_rho, rho_use, extrapolate=False)

            def rho_interp(r):
                r = np.asarray(r, dtype=float)
                out = np.full_like(r, np.nan, dtype=float)
                mask = r > 0
                out[mask] = rho_interp_raw(np.log(r[mask]))
                return out

        else:
            raise ValueError("rho_mode must be 'loglog' or 'logx_linear_y'.")

        # --- velocity interpolator
        if vel_mode == "loglog":
            log_vel = np.log(vel_use)
            vel_interp_raw = PchipInterpolator(log_r_vel, log_vel, extrapolate=False)

            def vel_interp(r):
                r = np.asarray(r, dtype=float)
                out = np.full_like(r, np.nan, dtype=float)
                mask = r > 0
                out[mask] = np.exp(vel_interp_raw(np.log(r[mask])))
                return out

        elif vel_mode == "logx_linear_y":
            vel_interp_raw = PchipInterpolator(log_r_vel, vel_use, extrapolate=False)

            def vel_interp(r):
                r = np.asarray(r, dtype=float)
                out = np.full_like(r, np.nan, dtype=float)
                mask = r > 0
                out[mask] = vel_interp_raw(np.log(r[mask]))
                return out

        else:
            raise ValueError("vel_mode must be 'loglog' or 'logx_linear_y'.")

        return {
            "time": time_value,
            "time-step": time_step,
            "r": r_snap,
            "rho": rho_snap,
            "velDis": vel_snap,
            "rho_interp": rho_interp,
            "velDis_interp": vel_interp,
            "rho_mode": rho_mode,
            "vel_mode": vel_mode,
        }

    def evaluate_snapshot_interpolators(
            self,
            time_argument,
            *,
            time_step_bool=True,
            drop_r_zero=True,
            rho_mode="loglog",
            vel_mode="loglog",
            n_eval=400,
    ):
        """
        Evaluate snapshot interpolators on a dense logarithmic radial grid.

        Returns
        -------
        out : dict
            Contains original arrays plus dense-grid interpolated curves.
        """
        out = self.build_snapshot_interpolators(
            time_argument,
            time_step_bool=time_step_bool,
            drop_r_zero=drop_r_zero,
            rho_mode=rho_mode,
            vel_mode=vel_mode,
        )

        r_snap = out["r"]
        rmin = np.nanmin(r_snap[r_snap > 0])
        rmax = np.nanmax(r_snap)

        r_dense = np.logspace(np.log10(rmin), np.log10(rmax), int(n_eval))

        out["r_dense"] = r_dense
        out["rho_dense"] = out["rho_interp"](r_dense)
        out["velDis_dense"] = out["velDis_interp"](r_dense)

        return out

    def interpolation_self_check(
            self,
            time_argument,
            *,
            time_step_bool=True,
            drop_r_zero=True,
            rho_mode="loglog",
            vel_mode="loglog",
    ):
        """
        Compare interpolated values evaluated back on the original grid
        with the stored profile values.

        Returns
        -------
        stats : dict
            Relative-error diagnostics for rho and velDis
        """
        out = self.build_snapshot_interpolators(
            time_argument,
            time_step_bool=time_step_bool,
            drop_r_zero=drop_r_zero,
            rho_mode=rho_mode,
            vel_mode=vel_mode,
        )

        r = out["r"]
        rho_true = out["rho"]
        vel_true = out["velDis"]

        rho_fit = out["rho_interp"](r)
        vel_fit = out["velDis_interp"](r)

        # relative errors
        rho_rel = np.abs(rho_fit - rho_true) / np.maximum(np.abs(rho_true), 1e-300)
        vel_rel = np.abs(vel_fit - vel_true) / np.maximum(np.abs(vel_true), 1e-300)

        return {
            "time": out["time"],
            "time-step": out["time-step"],
            "rho_rel_max": np.nanmax(rho_rel),
            "rho_rel_mean": np.nanmean(rho_rel),
            "vel_rel_max": np.nanmax(vel_rel),
            "vel_rel_mean": np.nanmean(vel_rel),
            "rho_rel_array": rho_rel,
            "vel_rel_array": vel_rel,
        }

    def build_interpolated_profiles(
            self,
            *,
            rho_mode="loglog",
            vel_mode="loglog",
            drop_r_zero=True,
            stop_at_collapse=True,
            collapse_elements=2,
    ):
        """
        Build and store radial interpolation functions for rho(r) and velDis(r)
        for each snapshot BEFORE collapse.

        Results are stored in:
            self.interp_profiles

        Parameters
        ----------
        rho_mode : str
            'loglog' or 'logx_linear_y'
        vel_mode : str
            'loglog' or 'logx_linear_y'
        drop_r_zero : bool
            Remove r=0 from interpolation grid
        stop_at_collapse : bool
            If True, do not build interpolators after collapse
        collapse_elements : int
            Passed to find_collapse()
        """

        time_arr, r_arr, rho_2d, vel_2d = self._reshape_gravo_profiles()

        n_t = len(time_arr)

        # -----------------------------------
        # determine cutoff time
        # -----------------------------------

        if stop_at_collapse:
            try:
                collapse_step = self.find_collapse(elements=collapse_elements)[2]
            except Exception:
                collapse_step = n_t
        else:
            collapse_step = n_t

        interp_profiles = []

        # -----------------------------------
        # build interpolators snapshot-by-snapshot
        # -----------------------------------

        for i in range(collapse_step):

            r = np.asarray(r_arr, dtype=float)
            rho = np.asarray(rho_2d[i], dtype=float)
            vel = np.asarray(vel_2d[i], dtype=float)

            if drop_r_zero:
                mask = r > 0
                r = r[mask]
                rho = rho[mask]
                vel = vel[mask]

            # clean masks
            rho_mask = np.isfinite(r) & np.isfinite(rho) & (rho > 0)
            vel_mask = np.isfinite(r) & np.isfinite(vel) & (vel > 0)

            r_rho = r[rho_mask]
            rho_use = rho[rho_mask]

            r_vel = r[vel_mask]
            vel_use = vel[vel_mask]

            if len(r_rho) < 2 or len(r_vel) < 2:
                continue

            log_r_rho = np.log(r_rho)
            log_r_vel = np.log(r_vel)

            # -------------------------
            # density interpolator
            # -------------------------

            if rho_mode == "loglog":

                log_rho = np.log(rho_use)
                interp_raw = PchipInterpolator(log_r_rho, log_rho, extrapolate=False)

                def rho_interp_factory(interp_raw):
                    return lambda r: np.exp(interp_raw(np.log(r)))

                rho_interp = rho_interp_factory(interp_raw)

            else:

                interp_raw = PchipInterpolator(log_r_rho, rho_use, extrapolate=False)

                def rho_interp_factory(interp_raw):
                    return lambda r: interp_raw(np.log(r))

                rho_interp = rho_interp_factory(interp_raw)

            # -------------------------
            # velocity interpolator
            # -------------------------

            if vel_mode == "loglog":

                log_vel = np.log(vel_use)
                interp_raw = PchipInterpolator(log_r_vel, log_vel, extrapolate=False)

                def vel_interp_factory(interp_raw):
                    return lambda r: np.exp(interp_raw(np.log(r)))

                vel_interp = vel_interp_factory(interp_raw)

            else:

                interp_raw = PchipInterpolator(log_r_vel, vel_use, extrapolate=False)

                def vel_interp_factory(interp_raw):
                    return lambda r: interp_raw(np.log(r))

                vel_interp = vel_interp_factory(interp_raw)

            interp_profiles.append({
                "time": time_arr[i],
                "time_step": i,
                "rho_interp": rho_interp,
                "vel_interp": vel_interp,
                "r_min": np.min(r),
                "r_max": np.max(r),
            })

        self.interp_profiles = interp_profiles

        return interp_profiles

    def build_scattering_matrix_from_interpolators(
            self,
            sigma_m,
            *,
            C=4.0 / np.sqrt(np.pi),
            sigma_input_vel_unit="km/s",
            ngrid=600,
            rmin=None,
            rmax=None,
            store_result=True,
    ):
        """
        Build the local integrand matrix and cumulative scattering matrix using
        the already-stored pre-collapse radial interpolators.

        Requires:
            self.interp_profiles
        to exist (created by build_interpolated_profiles()).

        Parameters
        ----------
        sigma_m : float or callable
            Constant cross section [cm^2/g] or callable sigma_m(nu).
        C : float
            Prefactor, default 4/sqrt(pi)
        sigma_input_vel_unit : str
            Unit passed into sigma_m when sigma_m is callable:
                "km/s", "cm/s", or "rs/Gyr"
        ngrid : int
            Number of radii in the new dense log radial grid.
        rmin, rmax : float or None
            Radial interval [r_s] used for matrix construction.
            If None, inferred from stored interpolators.
        store_result : bool
            If True, save into self.scattering_data

        Returns
        -------
        out : dict
            {
              "time_arr": ...,
              "r_grid": ...,
              "rho_grid": ...,
              "vel_grid": ...,
              "sigma_grid": ...,
              "integrand": ...,
              "N_scatt": ...
            }
        """
        if not hasattr(self, "interp_profiles") or self.interp_profiles is None:
            raise RuntimeError("Run build_interpolated_profiles() first.")

        profiles = self.interp_profiles
        if len(profiles) < 2:
            raise RuntimeError("Not enough interpolated profiles stored.")

        time_arr = np.array([p["time"] for p in profiles], dtype=float)

        # valid common radial domain
        rmins = np.array([p["r_min"] for p in profiles], dtype=float)
        rmaxs = np.array([p["r_max"] for p in profiles], dtype=float)

        if rmin is None:
            rmin = np.max(rmins)
        if rmax is None:
            rmax = np.min(rmaxs)

        if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin <= 0 or rmax <= rmin:
            raise ValueError("Invalid common radial range for interpolators.")

        r_grid = np.logspace(np.log10(rmin), np.log10(rmax), int(ngrid))

        n_t = len(time_arr)
        n_r = len(r_grid)

        rho_grid = np.empty((n_t, n_r), dtype=float)  # [rho_s]
        vel_grid = np.empty((n_t, n_r), dtype=float)  # [r_s / Gyr]

        # evaluate all interpolators on common r-grid
        for i, p in enumerate(profiles):
            rho_grid[i, :] = np.asarray(p["rho_interp"](r_grid), dtype=float)
            vel_grid[i, :] = np.asarray(p["vel_interp"](r_grid), dtype=float)

        # convert to physical units
        rho_s_msun_kpc3 = self.parameters["rho_s"]  # [M_sun/kpc^3]
        r_s_kpc = self.parameters["r_s"]  # [kpc]

        MSUN_G = 1.98847e33
        KPC_CM = 3.08567758e21
        GYR_S = 3.15576e16

        rho_phys = rho_grid * rho_s_msun_kpc3 * MSUN_G / (KPC_CM ** 3)  # [g/cm^3]
        nu_cm_s = vel_grid * r_s_kpc * KPC_CM / GYR_S  # [cm/s]
        nu_km_s = nu_cm_s / 1.0e5  # [km/s]

        if sigma_input_vel_unit == "km/s":
            nu_for_sigma = nu_km_s
        elif sigma_input_vel_unit == "cm/s":
            nu_for_sigma = nu_cm_s
        elif sigma_input_vel_unit == "rs/Gyr":
            nu_for_sigma = vel_grid
        else:
            raise ValueError("sigma_input_vel_unit must be 'km/s', 'cm/s', or 'rs/Gyr'.")

        # evaluate sigma_m
        if np.isscalar(sigma_m):
            sigma_grid = float(sigma_m) * np.ones_like(nu_for_sigma, dtype=float)
        elif callable(sigma_m):
            sigma_grid = np.asarray(sigma_m(nu_for_sigma), dtype=float)
        else:
            raise TypeError("sigma_m must be a scalar or callable.")

        # local integrand [1/s]
        integrand = C * rho_phys * nu_cm_s * sigma_grid

        # cumulative trapezoidal integral in time
        N_scatt = np.zeros_like(integrand, dtype=float)
        dt_s = np.diff(time_arr) * GYR_S

        for i in range(1, n_t):
            N_scatt[i, :] = (
                    N_scatt[i - 1, :]
                    + 0.5 * (integrand[i - 1, :] + integrand[i, :]) * dt_s[i - 1]
            )

        out = {
            "time_arr": time_arr,
            "r_grid": r_grid,
            "rho_grid": rho_grid,
            "vel_grid": vel_grid,
            "sigma_grid": sigma_grid,
            "integrand": integrand,
            "N_scatt": N_scatt,
        }

        if store_result:
            self.scattering_data = out

        return out

    # def _find_all_roots_on_grid_logr(self, r_grid, F_grid):
    #     """
    #     Find all roots of F(r)=0 from tabulated values on a log-r grid
    #     using sign-change detection + linear interpolation in log(r).
    #     """
    #     r_grid = np.asarray(r_grid, dtype=float)
    #     F_grid = np.asarray(F_grid, dtype=float)
    #
    #     valid = np.isfinite(r_grid) & np.isfinite(F_grid) & (r_grid > 0)
    #     r = r_grid[valid]
    #     F = F_grid[valid]
    #
    #     if len(r) < 2:
    #         return []
    #
    #     roots = []
    #
    #     # exact zeros
    #     exact = np.where(F == 0.0)[0]
    #     for j in exact:
    #         roots.append(float(r[j]))
    #
    #     # sign changes
    #     idx = np.where(F[:-1] * F[1:] < 0.0)[0]
    #     for j in idx:
    #         r1, r2 = r[j], r[j + 1]
    #         f1, f2 = F[j], F[j + 1]
    #
    #         x1, x2 = np.log(r1), np.log(r2)
    #         xr = x1 - f1 * (x2 - x1) / (f2 - f1)
    #         roots.append(float(np.exp(xr)))
    #
    #     roots = sorted(set([round(rr, 14) for rr in roots]))
    #     return [float(rr) for rr in roots]

    def _find_all_roots_on_grid_logr(
            self,
            r_grid,
            F_grid,
            *,
            F_tol=1e-5,
            dlogr_merge=0.05,
            # F_tol=1e-3,
            # dlogr_merge=0.02,
    ):
        """
        Find all robust roots of F(r)=0 on a log-r grid.

        Strategy
        --------
        1. Detect sign changes.
        2. Reject tiny-amplitude wiggle crossings.
        3. Interpolate root in log(r).
        4. Merge roots that are too close in log-radius.
        """
        r_grid = np.asarray(r_grid, dtype=float)
        F_grid = np.asarray(F_grid, dtype=float)

        valid = np.isfinite(r_grid) & np.isfinite(F_grid) & (r_grid > 0)
        r = r_grid[valid]
        F = F_grid[valid]

        if len(r) < 2:
            return []

        roots_raw = []

        # exact zeros
        exact = np.where(F == 0.0)[0]
        for j in exact:
            roots_raw.append(float(r[j]))

        # sign-change brackets
        idx = np.where(F[:-1] * F[1:] < 0.0)[0]
        for j in idx:
            r1, r2 = r[j], r[j + 1]
            f1, f2 = F[j], F[j + 1]

            # reject tiny numerical wiggles
            if max(abs(f1), abs(f2)) < F_tol:
                continue

            x1, x2 = np.log(r1), np.log(r2)
            xr = x1 - f1 * (x2 - x1) / (f2 - f1)
            roots_raw.append(float(np.exp(xr)))

        if len(roots_raw) == 0:
            return []

        roots_raw = sorted(roots_raw)

        # merge clustered roots
        roots_merged = [roots_raw[0]]
        for rr in roots_raw[1:]:
            if abs(np.log(rr) - np.log(roots_merged[-1])) < dlogr_merge:
                roots_merged[-1] = float(np.sqrt(roots_merged[-1] * rr))
            else:
                roots_merged.append(rr)

        return roots_merged

    def find_all_r1_roots_from_scattering_matrix(
            self,
            *,
            target=1.0,
            pick="outermost",
            return_debug=True,
    ):
        """
        Use self.scattering_data to find all r1 roots at every stored time.

        Solves:
            N_scatt(t, r) - target = 0

        Returns
        -------
        time_arr : ndarray
        r1_arr : ndarray
        roots_list : list of lists
        debug : dict (optional)
        """
        if not hasattr(self, "scattering_data") or self.scattering_data is None:
            raise RuntimeError("Run build_scattering_matrix_from_interpolators() first.")

        data = self.scattering_data
        time_arr = data["time_arr"]
        r_grid = data["r_grid"]
        N_scatt = data["N_scatt"]

        n_t = len(time_arr)

        roots_list = []
        r1_arr = np.full(n_t, np.nan, dtype=float)

        for i in range(n_t):
            F = N_scatt[i, :] - target
            roots = self._find_all_roots_on_grid_logr(r_grid, F)
            roots_list.append(roots)

            if len(roots) == 0:
                r1_arr[i] = np.nan
            else:
                if pick == "outermost":
                    r1_arr[i] = roots[-1]
                elif pick == "innermost":
                    r1_arr[i] = roots[0]
                elif pick == "middle":
                    r1_arr[i] = roots[1]
                elif pick == "myChoice":
                    if len(roots) == 3:
                        r1_arr[i] = roots[1]
                    elif len(roots) == 2:
                        r1_arr[i] = roots[0]
                    else:
                        r1_arr[i] = roots[0]
                else:
                    raise ValueError("pick must be 'outermost', 'innermost', or 'myChoice'.")

        if return_debug:
            return time_arr, r1_arr, roots_list, {
                "r_grid": r_grid,
                "N_scatt": N_scatt,
                "target": target,
            }

        return time_arr, r1_arr, roots_list

    def return_knudsen_core_evolution(
            self,
            sigma_m,
            *,
            elements_rho=2,
            elements_vel=1,
            sigma_input_vel_unit="km/s",
            use_core_average=True,
            stop_at_collapse=False,
            collapse_elements=2,
    ):
        """
        Return the time evolution of the central/core Knudsen number.

        Definition
        ----------
        Kn_core = lambda_core / H_core

        where
            lambda_core = 1 / (rho_core_phys * sigma_core)
            H_core      = nu_core_phys / sqrt(4*pi*G*rho_core_phys)

        Thus
            Kn_core = sqrt(4*pi*G*rho_core_phys) / (rho_core_phys * sigma_core * nu_core_phys)

        Parameters
        ----------
        sigma_m : float or callable
            Constant cross section [cm^2/g] or callable sigma_m(nu).
        elements_rho : int
            Number of inner bins used to estimate rho_core through aux.calculate_rho_core().
        elements_vel : int
            Number of inner bins used to estimate velDis_core through aux.calculate_veldis_core().
        sigma_input_vel_unit : str
            Velocity unit passed into sigma_m if callable:
            "km/s", "cm/s", or "rs/Gyr"
        use_core_average : bool
            If True, use the core estimators.
            If False, use the first nonzero-radius bin directly.
        stop_at_collapse : bool
            If True, stop before collapse.
        collapse_elements : int
            Passed to find_collapse() when stop_at_collapse=True.

        Returns
        -------
        out : dict
            {
                "time": time array [Gyr],
                "rho_core_dimless": ...,
                "vel_core_dimless": ...,
                "rho_core_phys": ...,
                "vel_core_cm_s": ...,
                "sigma_core": ...,
                "lambda_core_cm": ...,
                "H_core_cm": ...,
                "Kn_core": ...
            }
        """
        if self.parameters is None:
            raise RuntimeError("Call put_extra_parameters(...) first.")

        MSUN_G = 1.98847e33
        KPC_CM = 3.08567758e21
        GYR_S = 3.15576e16

        # You need this in config.py:
        # const_G_cgs = 6.67430e-8
        G_cgs = 6.67430e-8

        if stop_at_collapse:
            try:
                n_t = self.find_collapse(elements=collapse_elements)[2]
            except Exception:
                n_t = self.time_steps
        else:
            n_t = self.time_steps

        time_arr = []
        rho_core_arr = []
        vel_core_arr = []

        rho_s_msun_kpc3 = self.parameters["rho_s"]
        r_s_kpc = self.parameters["r_s"]

        for i in range(n_t):
            left = i * self.number_radi
            right = (i + 1) * self.number_radi

            r_snap = np.asarray(self.data["r"][left:right], dtype=float)
            rho_snap = np.asarray(self.data["rho"][left:right], dtype=float)
            vel_snap = np.asarray(self.data["velDis"][left:right], dtype=float)

            # drop r<=0 if present
            mask = np.isfinite(r_snap) & np.isfinite(rho_snap) & np.isfinite(vel_snap) & (r_snap > 0)
            r_snap = r_snap[mask]
            rho_snap = rho_snap[mask]
            vel_snap = vel_snap[mask]

            if len(r_snap) == 0:
                time_arr.append(self.data["time-no-repetition"][i])
                rho_core_arr.append(np.nan)
                vel_core_arr.append(np.nan)
                continue

            if use_core_average:
                # --- rho_core
                n_rho = min(elements_rho, len(rho_snap) - 1)
                if n_rho < 1:
                    rho_core = rho_snap[0]
                else:
                    r_in_core = r_snap[:n_rho + 1]
                    rho_in_core = rho_snap[:n_rho]
                    rho_core = aux.calculate_rho_core(r_in_core, rho_in_core, n_rho)

                # --- vel_core
                n_vel = min(elements_vel, len(vel_snap) - 1)
                if n_vel < 1:
                    vel_core = vel_snap[0]
                else:
                    r_in_core_v = r_snap[:n_vel + 1]
                    vel_in_core = vel_snap[:n_vel]
                    vel_core = aux.calculate_veldis_core(r_in_core_v, vel_in_core, n_vel)
            else:
                rho_core = rho_snap[0]
                vel_core = vel_snap[0]

            time_arr.append(self.data["time-no-repetition"][i])
            rho_core_arr.append(rho_core)
            vel_core_arr.append(vel_core)

        time_arr = np.asarray(time_arr, dtype=float)
        rho_core_arr = np.asarray(rho_core_arr, dtype=float)  # [rho_s]
        vel_core_arr = np.asarray(vel_core_arr, dtype=float)  # [r_s/Gyr]

        # Convert to physical units
        rho_core_phys = rho_core_arr * rho_s_msun_kpc3 * MSUN_G / (KPC_CM ** 3)  # [g/cm^3]
        vel_core_cm_s = vel_core_arr * r_s_kpc * KPC_CM / GYR_S  # [cm/s]
        vel_core_km_s = vel_core_cm_s / 1e5

        # Evaluate sigma at the core velocity
        if np.isscalar(sigma_m):
            sigma_core = float(sigma_m) * np.ones_like(vel_core_cm_s)
        elif callable(sigma_m):
            if sigma_input_vel_unit == "km/s":
                sigma_core = np.asarray(sigma_m(vel_core_km_s), dtype=float)
            elif sigma_input_vel_unit == "cm/s":
                sigma_core = np.asarray(sigma_m(vel_core_cm_s), dtype=float)
            elif sigma_input_vel_unit == "rs/Gyr":
                sigma_core = np.asarray(sigma_m(vel_core_arr), dtype=float)
            else:
                raise ValueError("sigma_input_vel_unit must be 'km/s', 'cm/s', or 'rs/Gyr'.")
        else:
            raise TypeError("sigma_m must be a scalar or callable.")

        # Mean free path and gravitational scale height
        lambda_core_cm = 1.0 / (rho_core_phys * sigma_core)
        H_core_cm = vel_core_cm_s / np.sqrt(4.0 * np.pi * G_cgs * rho_core_phys)

        Kn_core = lambda_core_cm / H_core_cm

        return {
            "time": time_arr,
            "rho_core_dimless": rho_core_arr,
            "vel_core_dimless": vel_core_arr,
            "rho_core_phys": rho_core_phys,
            "vel_core_cm_s": vel_core_cm_s,
            "sigma_core": sigma_core,
            "lambda_core_cm": lambda_core_cm,
            "H_core_cm": H_core_cm,
            "Kn_core": Kn_core,
        }

# --- do test