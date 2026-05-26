"""
This file contains all method to investigate the data coming from the `Rprocedure` simulation.
So, here we deal with `Isothermal` evolution till the time, when this procedure fails.
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
import config as cfg
import units as uni
import auxiliaryFunctions as aux
# from Rprocedure
from NFWProfile import NFWProfile, r1  # CDM profile (halo)


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

################################################################################################################
# ##################################### LOW DENSE SOLUTION #####################################################
################################################################################################################
def create_RprocedureData_from_file(file_name, localization_of_file):
    """
    Creates an RprocedureData object from a CSV file containing isothermal evolution data.

    Parameters:
    - file_name: str
        The name of the CSV file containing the R1-procedure data.
    - localization_of_file: str
        The directory where the CSV file is located.

    Returns:
    - isoEvolution: RprocedureData
        An object containing the parsed isothermal evolution data.
    """
    # Define the column names
    column_names = [
        't', 'r', 'rho', 'mass', 'velDis',
        'central-rho', 'central-velDis', 'names', 'values'
    ]

    # Define the data types for each column
    dtype = {
        't': 'float64',
        'r': 'float64',
        'rho': 'float64',
        'mass': 'float64',
        'velDis': 'float64',
        'central-rho': 'float64',
        'central-velDis': 'float64',
        'names': 'str',
        'values': 'float64'
    }

    # Load the single CSV file containing various columns related to isothermal evolution
    df_isothermal = aux.load_csv_from_dir(localization_of_file, file_name, sep='\t', names=column_names, dtype=dtype)

    # Create an RprocedureData object using the loaded data
    isoEvolution = RprocedureData(df_isothermal['t'], df_isothermal['r'], df_isothermal['rho'],
                                  df_isothermal['velDis'], df_isothermal['names'], df_isothermal['values'])

    # Set central data for the object
    isoEvolution.set_central_data(df_isothermal['central-rho'], df_isothermal['central-velDis'])

    return isoEvolution

def create_RprocedureDataMirror_from_file(file_name, localization_of_file, proportion=2.0):
    """
    Creates an RprocedureData object from a CSV file containing isothermal evolution data.

    Parameters:
    - file_name: str
        The name of the CSV file containing the R1-procedure data.
    - localization_of_file: str
        The directory where the CSV file is located.
    - proportion: float
        time of collapse / time of merging: `tcolla / tmerge`, should be > 1.0

    Returns:
    - isoEvolution: RprocedureData
        An object containing the parsed isothermal evolution data.
    """
    # Define the column names
    column_names = [
        't', 'r', 'rho', 'mass', 'velDis',
        'central-rho', 'central-velDis', 'names', 'values'
    ]

    # Define the data types for each column
    dtype = {
        't': 'float64',
        'r': 'float64',
        'rho': 'float64',
        'mass': 'float64',
        'velDis': 'float64',
        'central-rho': 'float64',
        'central-velDis': 'float64',
        'names': 'str',
        'values': 'float64'
    }

    # Load the single CSV file containing various columns related to isothermal evolution
    df_isothermal = aux.load_csv_from_dir(localization_of_file, file_name, sep='\t', names=column_names, dtype=dtype)

    # Create an RprocedureDataMirror object using the loaded data
    MirrorEvolution = RprocedureDataMirror(df_isothermal['t'], df_isothermal['r'], df_isothermal['rho'],
                                           df_isothermal['velDis'], df_isothermal['names'], df_isothermal['values'],
                                           proportion=proportion)

    # Set central data for the object
    MirrorEvolution.set_central_data(df_isothermal['central-rho'], df_isothermal['central-velDis'])

    return MirrorEvolution


# ##################################### FUNCTION TO RETURN DATA FROM FILE ##################################### #
class RprocedureData(object):
    """
    Class, which will stored all the necessary data fom Rprocedure simulation.
    """

    def __init__(self, time_list, r_list, rho_list, veldis_list, parameters_names, parameters_values,
                 Delta=200.0, z=0.0):
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
            parameters_names: array, which contains names of parameters, which we set / get during
                Rprocedure simulation.
                (array or list)
            parameters_values: array, which contains values of parameters, mentioned above.
                (array or list)
            Delta: spherical overdensity with respect to the critical
                density of the universe (default is 200.)
                (float) [kpc]
            z: redshift (float)

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
        self.name = None  # to differentiate the data stored in class
        # --- the basic information
        self.data = dict()
        self.data["time"] = [10 ** x for x in time_list]  # CHECK IT! [log10(Gyr)]
        time_no_repetition = list(dict.fromkeys(self.data["time"]))
        self.data["time-no-repetition"] = np.array(time_no_repetition)
        self.data["r"] = r_list.tolist()  # [r_s]
        self.data["rho"] = rho_list.tolist()  # [rho_s]
        self.data["velDis"] = veldis_list.tolist()  # [r_s/Gyr]

        # to parametrise the time
        len_t = 0
        while self.data["time"][len_t] == self.data["time"][0]:
            len_t += 1
        self.number_radi = len_t  # Same values of time - how many radius values we have for each one fixed time
        self.time_steps = int(len(self.data["time"]) / len_t)  # how many steps on time is in the .txt file

        # --- additional parameters
        self.parameters = dict()
        number_par = len(parameters_names)
        for i in range(0, number_par):
            par_name = parameters_names[i]
            if par_name == "Mvir":
                self.parameters["Mvir"] = parameters_values[i]  # [M_sun]
            elif par_name == "c_const":
                self.parameters["const_c"] = parameters_values[i]  # [dimensionless]
            elif par_name == "t":
                self.parameters["time-end"] = parameters_values[i]  # [Gyr]
            elif par_name == "cross_section":
                self.parameters["sigma_m"] = parameters_values[i]  # [cm^2/g]
            elif par_name == "r1":
                self.parameters["r1"] = parameters_values[i]  # [kpc]

        self.parameters["Deltah"] = Delta  # [kpc]
        self.parameters["z"] = z

        self.parameters["rhoc"] = cfg.rho_c  # [M_sun kpc^-3], not best: THINK!
        self.parameters["rhoh"] = self.parameters["Deltah"] * self.parameters["rhoc"]  # [M_sun kpc^-3]
        self.parameters["rh"] = (3. * self.parameters["Mvir"] / (cfg.FourPi * self.parameters["rhoh"])) ** (
                    1. / 3.)  # [kpc]
        self.parameters["r_s"] = self.parameters["rh"] / self.parameters["const_c"]  # [kpc]
        self.parameters["rho_s"] = self.parameters["rhoc"] * self.parameters["Deltah"] / 3. * \
                                   self.parameters["const_c"] ** 3. / f(self.parameters["const_c"])  # [M_sun / kpc^3]

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

    # -------------------------------- RETURN PARAMETERS -------------------------------- #
    def return_basic_parameters(self):
        """Return Mvir and sigma_m and dark mater concentration"""
        return [self.parameters["Mvir"], self.parameters["sigma_m"], self.parameters["const_c"]]

    def return_parameters(self):
        """Return Mvir and sigma_m and dark mater concentration"""
        return self.parameters

    # -------------------------------- CENTRAL DENSITY, NU -------------------------------- #
    def set_central_data(self, central_density_arr, central_nu_arr):
        """
        In txt file we can also can find the central density and central velocity dispersion.
        But we have take care about time list, which describes when we obtained those values.
        In txt file those time should be exactly the same as `self.data["time-no-repetition"]`.

        :param central_density_arr: array contains evolution of central density.
            (array) [rho_s]
        :param central_nu_arr: array contains evolution of central velocity dispersion.
            (array) [r_s/Gyr]
        """
        self.central_data = dict()
        self.central_data["time"] = self.data["time-no-repetition"].copy()  # [Gyr]
        self.central_data["rho"] = central_density_arr[:self.time_steps]  # [rho_s]
        self.central_data["velDis"] = central_nu_arr[:self.time_steps]  # [r_s/Gyr]

    def return_central_rho(self):
        """Return central density and time"""
        return [self.central_data['time'], self.central_data['rho']]

    def return_central_veldis(self):
        """Return central velocity dispersion and time"""
        return [self.central_data['time'], self.central_data['velDis']]

    # -------------------------------- CORE DATA -------------------------------- #
    def return_rho_core_evolution(self, elements=1):
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
            rho_in_core = self.data["rho"][(0 + left_arg):(0 + right_arg)]  # bez przesunięcia
            # calculate the value of the core
            rho_core = aux.calculate_rho_core(r_in_core, rho_in_core, elements)
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

    # -------------------------------- LAST MOMENT OF FORMING THE CORE -------------------------------- #
    def find_min_rho_core(self, elements=1):
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

    def return_data_last_moment_forming(self, elements=1):
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


################################################################################################################
# ##################################### HIGH DENSE SOLUTION ################################################## #
################################################################################################################

# ##################################### FUNCTION TO RETURN DATA FROM FILE ##################################### #
class RprocedureDataMirror(object):
    """
    Class, which will stored all the necessary data fom Rprocedure simulation. This class will store
    data, which describes evolution of `high dense` solution after we have adapted the mirror method.
    ----
    Important to notice that here we introduce new variable `proportion`, which is essential to
    `mirror method`.

        proportion = time of collapse / time of merging,

    where
        time of merging: tmerge has been taken from data.
        time of collapse: tcolla is somehow fixed.

    The `mirror method` will turn back time by taking mirror image, so it goes:

        t -> tcalla - t

    The part of the debate or investigation is the proportion between two important time, which
    determinate the evolution of halo: merging and collapsing. So we class enables freely changes
    that constant.
    ----
    Be aware!
    `time-end` == tmerge
    """

    def __init__(self, time_list, r_list, rho_list, veldis_list, parameters_names, parameters_values,
                 Delta=200.0, z=0.0, proportion=2.0):
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
            parameters_names: array, which contains names of parameters, which we set / get during
                Rprocedure simulation.
                (array or list)
            parameters_values: array, which contains values of parameters, mentioned above.
                (array or list)
            Delta: spherical overdensity with respect to the critical
                density of the universe (default is 200.)
                (float) [kpc]
            z: redshift (float)
            proportion: time of collapse / time of merging: tcolla / tmerge (float)
                should be > 1.0

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
        self.name = None  # to differentiate the data stored in class
        # --- additional parameters
        self.parameters = dict()
        number_par = len(parameters_names)
        for i in range(0, number_par):
            par_name = parameters_names[i]
            if par_name == "Mvir":
                self.parameters["Mvir"] = parameters_values[i]  # [M_sun]
            elif par_name == "c_const":
                self.parameters["const_c"] = parameters_values[i]  # [dimensionless]
            elif par_name == "t":
                self.parameters["time-end"] = parameters_values[i]  # [Gyr]
            elif par_name == "cross_section":
                self.parameters["sigma_m"] = parameters_values[i]  # [cm^2/g]
            elif par_name == "r1":
                self.parameters["r1"] = parameters_values[i]  # [kpc]

        self.parameters["Deltah"] = Delta  # [kpc]
        self.parameters["z"] = z

        self.parameters["rhoc"] = cfg.rho_c  # [M_sun kpc^-3], not best: THINK!
        self.parameters["rhoh"] = self.parameters["Deltah"] * self.parameters["rhoc"]  # [M_sun kpc^-3]
        self.parameters["rh"] = (3. * self.parameters["Mvir"] / (cfg.FourPi * self.parameters["rhoh"])) ** (
                    1. / 3.)  # [kpc]
        self.parameters["r_s"] = self.parameters["rh"] / self.parameters["const_c"]  # [kpc]
        self.parameters["rho_s"] = self.parameters["rhoc"] * self.parameters["Deltah"] / 3. * \
                                   self.parameters["const_c"] ** 3. / f(self.parameters["const_c"])  # [M_sun / kpc^3]

        # --- the basic information
        self.data = dict()
        self.data["time-file"] = time_list  # [log10(Gyr)]
        self.data["proportion"] = proportion
        self.data["tcolla"] = proportion * self.parameters["time-end"]
        # self.data["time"] = [proportion * self.parameters["time-end"] - 10 ** x for x in time_list]  # [log10(Gyr)]
        self.data["time"] = [self.cal_time_mirror(10 ** x) for x in time_list]  # [Gyr]
        time_no_repetition = list(dict.fromkeys(self.data["time"]))
        self.data["time-no-repetition"] = np.array(time_no_repetition)
        self.data["r"] = r_list.tolist()  # [r_s]
        self.data["rho"] = rho_list.tolist()  # [rho_s]
        self.data["velDis"] = veldis_list.tolist()  # [r_s/Gyr]

        # to parametrise the time
        len_t = 0
        while self.data["time"][len_t] == self.data["time"][0]:
            len_t += 1
        self.number_radi = len_t  # Same values of time - how many radius values we have for each one fixed time
        self.time_steps = int(len(self.data["time"]) / len_t)  # how many steps on time is in the .txt file

    # -------------------------------- BASIC FUN -------------------------------- #
    def cal_time_mirror(self, time):
        """
        This function deals with swapping time from `high dense` into `mirror method`.
        So, somehow here we implement the method of taking mirror time.

        where,

        param time: time of the evolution, which we get from the `high dense solution`
        (in the Rprocedure) and we reinterpret in new way.
        (float) [Gyr]
        """
        mirror_time = - (self.data["proportion"] - 1) * time + self.parameters["time-end"] * self.data["proportion"]
        return mirror_time

    def change_proportion_value(self, proportion=2.0):
        """
        This function will change the value of the `proportion` parameter. This change in parameter
        will influence only on time! So it can be useful to have such a function.

        param _proportion: time of collapse / time of merging: tcolla / tmerge
            (float). should be > 1.0

        -----
        This function does not return anything, just have changed the parameter stored in class.
        """
        self.data["proportion"] = proportion
        self.data["tcolla"] = proportion * self.parameters["time-end"]
        self.data["time"] = [self.cal_time_mirror(10 ** x) for x in self.data["time-file"]]  # [Gyr]
        time_no_repetition = list(dict.fromkeys(self.data["time"]))
        self.data["time-no-repetition"] = np.array(time_no_repetition)

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

    # -------------------------------- RETURN PARAMETERS -------------------------------- #
    def return_basic_parameters(self):
        """Return Mvir and sigma_m and dark mater concentration"""
        return [self.parameters["Mvir"], self.parameters["sigma_m"], self.parameters["const_c"]]

    # -------------------------------- CENTRAL DENSITY, NU -------------------------------- #
    def set_central_data(self, central_density_arr, central_nu_arr):
        """
        In txt file we can also can find the central density and central velocity dispersion.
        But we have take care about time list, which describes when we obtained those values.
        In txt file those time should be exactly the same as `self.data["time-no-repetition"]`.

        :param central_density_arr: array contains evolution of central density.
            (array) [rho_s]
        :param central_nu_arr: array contains evolution of central velocity dispersion.
            (array) [r_s/Gyr]
        """
        self.central_data = dict()
        self.central_data["time"] = self.data["time-no-repetition"].copy()  # [Gyr]
        self.central_data["rho"] = central_density_arr[:self.time_steps]  # [rho_s]
        self.central_data["velDis"] = central_nu_arr[:self.time_steps]  # [r_s/Gyr]

    def return_central_rho(self):
        """Return central density and time"""
        return [self.central_data['time'], self.central_data['rho']]

    def return_central_veldis(self):
        """Return central velocity dispersion and time"""
        return [self.central_data['time'], self.central_data['velDis']]

    # -------------------------------- CORE DATA -------------------------------- #
    def return_rho_core_evolution(self, elements=1):
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
            rho_in_core = self.data["rho"][(0 + left_arg):(0 + right_arg)]  # bez przesunięcia
            # calculate the value of the core
            rho_core = aux.calculate_rho_core(r_in_core, rho_in_core, elements)
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

    # -------------------------------- FIND COLLAPSE -------------------------------- #
    def find_collapse(self, r_1_at_coll=1.85, elements=2):
        """
        Find the time and also the value of central density when collapsing occurred.

        :param elements: how many first radius's we treat as belonging to a core.
            (int)
        :param r_1_at_coll: the value of `r_1`, which refers to the first stable solution in the
            high dense (and mirror method). I estimate it should be in range [1.847, 1.991] [r_s]
            (float) [r_s]
        :return: [core density, time when collapse is occurring, time step of that]
        """
        # --- create NFW Class
        NFW_class = NFWProfile(self.parameters["Mvir"], self.parameters["const_c"])
        r_s = NFW_class.r_s
        rho_s = NFW_class.rho_s

        # --- find proper time
        r_1_cal = 0
        time_evolution_tilda = 0  # [dimensionless], 40, 0
        time_evolution_SU = uni.convert_time_tilda(time_evolution_tilda, rho_s, r_s, self.parameters["sigma_m"])  # [Gyr]
        while r_1_cal < r_1_at_coll:
            r_1_cal = r1(NFW_class, sigmamx=self.parameters["sigma_m"], tage=time_evolution_SU) / r_s
            time_evolution_tilda = time_evolution_tilda + 0.1  # new step [dimensionless]
            time_evolution_SU = uni.convert_time_tilda(time_evolution_tilda, rho_s, r_s, self.parameters["sigma_m"])  # [Gyr]

        time_mirror_SU = self.data["tcolla"] - time_evolution_SU
        print("r_1_cal:", r_1_cal)
        time_collapse, time_step_collapse = aux.find_nearest_and_index(self.data["time-no-repetition"], time_mirror_SU)

        # --- Find the proper data
        # the value of core density at collapse
        left_arg = time_step_collapse * self.number_radi
        right_arg = time_step_collapse * self.number_radi + elements
        # data which lies inside the core
        max_density = self.data["rho"][left_arg]
        r_in_core = self.data["r"][left_arg:(right_arg + 1)]
        rho_in_core = self.data["rho"][left_arg:right_arg]
        # calculate the value of the core
        value_of_rho_core = aux.calculate_rho_core(r_in_core, rho_in_core, elements)

        return [value_of_rho_core, time_collapse, time_step_collapse, max_density]

    def return_data_at_collapse(self, r_1_at_coll=1.85):
        """
        Return all necessary data at collapse (or that what we denote as collapse)

        :param r_1_at_coll: the value of `r_1`, which refers to the first stable solution in the
            high dense (and mirror method). I estimate it should be in range [1.847, 1.991] [r_s]
            (float) [r_s]
        """
        # --- create NFW Class
        NFW_class = NFWProfile(self.parameters["Mvir"], self.parameters["const_c"])
        r_s = NFW_class.r_s
        rho_s = NFW_class.rho_s

        # --- find proper time
        r_1_cal = 0
        time_evolution_tilda = 0  # [dimensionless], 40, 0
        time_evolution_SU = uni.convert_time_tilda(time_evolution_tilda, rho_s, r_s, self.parameters["sigma_m"])  # [Gyr]
        while r_1_cal < r_1_at_coll:
            r_1_cal = r1(NFW_class, sigmamx=self.parameters["sigma_m"], tage=time_evolution_SU) / r_s
            time_evolution_tilda = time_evolution_tilda + 0.1  # new step [dimensionless]
            time_evolution_SU = uni.convert_time_tilda(time_evolution_tilda, rho_s, r_s, self.parameters["sigma_m"])  # [Gyr]

        time_mirror_SU = self.data["tcolla"] - time_evolution_SU
        print("r_1_cal:", r_1_cal)
        time_collapse, time_step_collapse = aux.find_nearest_and_index(self.data["time-no-repetition"], time_mirror_SU)

        # arguments
        left_arg = time_step_collapse * self.number_radi
        right_arg = (time_step_collapse + 1) * self.number_radi - 1

        # data
        data_at_fixed_time = dict()
        data_at_fixed_time["time-step"] = time_step_collapse
        data_at_fixed_time["time"] = time_collapse  # [Gyr]
        data_at_fixed_time["r"] = self.data["r"][left_arg:right_arg]
        data_at_fixed_time["rho"] = self.data["rho"][left_arg:right_arg]
        data_at_fixed_time["velDis"] = self.data["velDis"][left_arg:right_arg]

        return data_at_fixed_time

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
