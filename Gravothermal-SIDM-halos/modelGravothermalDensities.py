"""
In this file we will code a model, which takes a .txt file, which was created by mathematica programme and then
calculate how looks the density profile of dark matter in selected galaxy. In other words how looks like \rho (r,t).
This empirical model for LMFP gravothermal densities profiles, has been described in paper 2205.02957 (especially
see Appendix: A).

This .txt file has had three columns of data, which containg:
    column[0]: time [log_10 Gyr]
    column[1]: radius [GeV**-1]
    column[2]: the density [GeV**4]

    #[density] = [mass/distance**3] = [mass]/[distance]**3 = [GeV]/[GeV]**-3 = [GeV]**4
    #[mass] = [GeV]
    #[distance] = [GeV]**-1
"""
import numpy as np
import config as cfg

# ------------------------------------ USEFUL FUNCTIONS ------------------------------------ #
# all comes from mathematica file (Ayuki)
def a(_z):
    """
    helpfully function.
    z: redshift usually `z=0`. (float)
    """
    return 0.520 + (0.905 - 0.520) * np.exp(-0.617 * _z ** 1.21)

def b(_z):
    """
    helpfully function.
    z: redshift  usually `z=0`. (float)
    """
    return -0.101 + 0.026 * _z

def c(_mvir, _z):
    """
    helpfully function.
    z: redshift  usually `z=0`. (float)
    Mvir: value of viral mass.
    (float) [M_sun]
    """
    return 10**(a(_z)) * (_mvir / (10 ** 12 * cfg.const_h ** (-1))) ** (b(_z))

def cal_r_s(_mvir, _z):
    """
    Find `r_s`.

    z: redshift, usually `z=0`. (float)
    Mvir: value of viral mass.
    (float) [M_sun]
    """
    return (200 * 4 / 3 * np.pi * cfg.rho_c * 10**-9) ** (-1/3) * _mvir ** (1/3) * (c(_mvir, _z))**(-1) * 10 ** (-3)  # [kpc]

def cal_rho_s(_mvir, _z):
    """
    Find `rho_s`.

    z: redshift usually `z=0`. (float)
    Mvir: value of viral mass.
    (float) [M_sun]
    """
    return _mvir / (4 * np.pi * (cal_r_s(_mvir, _z)) ** 3) \
        * 1 / (np.log(1 + c(_mvir, _z)) - c(_mvir, _z) / (1 + c(_mvir, _z)))

# rho_s = cal_rho_s(10**10, 0)
# r_s = cal_r_s(10**10, 0)
# c_val = c(10**10, 0)
# print("rho_s:", rho_s, "[M/kpc^3]")
# print("r_s:", r_s, "[kpc]")
# print("c_val:", c_val, "[]")
# print("cfg.h:", cfg.const_h, "[]")

# ------------------------------------ CHANGE UNITS ------------------------------------ #
def cal_t0(_rho_s, _r_s, _sigma_m):
    """
    calculate the 'scaling time' variable.
    Where:

        _rho_s: [M_sun/kpc^3] (float).
        _r_s: [kpc] (float).
        _sigma_m: DM cross-section [cm^2/g] (float).

    More in `auxiliaryFunctions.py` file.
    """
    # DM cross-section
    sigma_m_SI = _sigma_m * 10 ** (-4) * 10 ** (3)  # [m^2/kg]
    sigma_m_SU = sigma_m_SI * (cfg.kpc_SI) ** (-2) * cfg.M_solar_SI  # [kpc^2 * M_sun^(-1)]
    # cal nu0
    M0 = cfg.FourPi * _r_s ** 3 * _rho_s  # [M_sun]
    G_SU = cfg.const_G_starUnits  # [kpc^3 * Gyr^-2 * M_sun^-1]
    nu0 = np.sqrt(G_SU * M0 / _r_s)  # [kpc * Gyr^-1]

    return (cfg.FourOverRootPi * sigma_m_SU * nu0 * _rho_s) ** (-1)  # [Gyr]

def cal_time_tilda(_tage, _rho_s, _r_s, _sigma_m, log_time=False):
    """
    calculate the dimensionless `t^`.
    Where: selected evolution time of galaxy [Gyr] (float).

        _tage: evolution time of galaxy [Gyr or log(Gyr)] (float).
        _rho_s: parameter of NFW [M_sun/kpc^3] (float).
        _r_s: parameter of NFW [kpc] (float).
        _sigma_m: DM cross-section [cm^2/g] (float).
        log_time: which type of time you put. If you put log10(Gyr), set True. (bool)

    More in `auxiliaryFunctions.py` file.
    """
    t0 = cal_t0(_rho_s, _r_s, _sigma_m)  # [Gyr]

    # different units of time
    if log_time is False:
        return _tage / t0
    else:
        return 10 ** _tage / t0

def convert_time_tilda(_time_tilda, _rho_s, _r_s, _sigma_m, log_time=False):
    """
    Calculate the dimension full `time`, I mean the time.

                tage = time_tilda * t0 [Gyr]

        where:

            _time_tilda: evolution time of galaxy, the dimensionless value
                [dimensionless] (float).
            _rho_s: parameter of NFW
                [M_sun/kpc^3] (float).
            _r_s: parameter of NFW
                [kpc] (float).
            _sigma_m: DM cross-section.
                [cm^2/g] (float).
            log_time: which type of time you receive. If you want log10(Gyr) in output, set True.
                (bool)
        """
    t0 = cal_t0(_rho_s, _r_s, _sigma_m)  # [Gyr]
    if log_time is False:
        return _time_tilda * t0  # [Gyr]
    else:
        return np.log10(_time_tilda * t0)  # [log10(Gyr)]

# ###########################  EMPIRICAL MODEL FOR THE LMFP GRAVOTHERMAL ########################### #
class ModelGravothermal(object):
    """
    Class, which will describe LMFP grovothermal model, where we can predict density profile: \rho(r,t).
    --------------------------------------------------------------------------------------------------
    For now this class can work with our approach - but it can be changed in such a way to work parallel to
    both way!!!
    """

    def __init__(self, beta: float):
        """
        Because we will use results from publication 2205.02957, but unfortunately we will use different
        sets of units we have to differentiate those two set-ups. So, whenever we will use, for instances:
        `old_parameters` that will refer to their settings. Our set-up will be differentiated by using word `new`,
        for example: `new_parameters`.

        In general, we will use really similar sets of units:
            density profile: [rho_s]
            radius: [r_s]
            annihilation cross-section: [cm^2/g]

        The differences are apper, when it came to time
            old time: t^ = t / t_0, where t_0 = 1/sqrt(4 * pi * G * rho_s)
            new time: t^ = t / t_0, where t_0 = 1/sqrt(4 * pi * G * rho_s) *  1 / (sigma_m * r_s * rho_s) *
                                                * (4/sqrt(pi))**(-1)

            remember that t^ denotes to the `dimensionless time`. Thus, [t] = [t_0].

        beta: adjustable parameter to match the fluid numerical solutions with N-body
              or gravothermal simulations.
              (float) [dimensionless]
        """
        # --- basic constants
        # FourOverRootPi = 4 / np.sqrt(np.pi)
        logFourOverRootPi = np.log10(4 / np.sqrt(np.pi))

        self.beta = beta  # set the parameter beta
        # --------------------------------------- DEAL WITH OLD MODEL --------------------------------------- #
        self.old_param = dict()
        # --- behaviour of halo evolution
        self.old_param["E"] = 2.238  # moment of core collapse
        self.old_param["F"] = 1.341  # moment of maximum core - start core contraction
        self.old_param["s"] = 2.19  # power dependency in core contraction
        # --- parameters corresponding to growing core
        self.old_param["A_r_core_early"] = -0.1078
        self.old_param["B_r_core_early"] = 0.3737
        self.old_param["C_r_core_early"] = -0.7720
        # --- parameters corresponding to core contraction
        self.old_param["A_rho_core"] = 0.05771
        self.old_param["C_rho_core"] = -21.64
        self.old_param["D_rho_core"] = 21.11
        self.old_param["A_r_core"] = -0.04049
        self.old_param["C_r_core"] = 43.07
        self.old_param["D_r_core"] = -43.07
        self.old_param["A_r_out"] = 0.02403
        self.old_param["C_r_out"] = -4.724
        self.old_param["D_r_out"] = 5.011

        # --------------------------------------- DEAL WITH NEW MODEL --------------------------------------- #
        self.new_param = dict()
        # --- behaviour of halo evolution
        self.new_param["E"] = 2.238 + logFourOverRootPi  # moment of core collapse
        self.new_param["F"] = 1.341 + logFourOverRootPi  # moment of maximum core - start core contraction
        self.new_param["s"] = 2.19  # power dependency in core contraction
        # --- parameters corresponding to growing core
        self.new_param["A_r_core_early"] = self.old_param["A_r_core_early"]
        self.new_param["B_r_core_early"] = self.old_param["B_r_core_early"] \
            - 2 * self.old_param["A_r_core_early"] * logFourOverRootPi
        self.new_param["C_r_core_early"] = self.old_param["C_r_core_early"] \
            - self.old_param["A_r_core_early"] * logFourOverRootPi ** 2 \
            - (self.old_param["B_r_core_early"] - 2 * self.old_param["A_r_core_early"] * logFourOverRootPi) * logFourOverRootPi
        # --- parameters corresponding to core contraction
        self.new_param["A_rho_core"] = 0.05771
        self.new_param["C_rho_core"] = -21.64
        self.new_param["D_rho_core"] = 21.11

        self.new_param["A_r_core"] = -0.04049
        self.new_param["C_r_core"] = 43.07
        self.new_param["D_r_core"] = -43.07

        self.new_param["A_r_out"] = 0.02403
        self.new_param["C_r_out"] = -4.724
        self.new_param["D_r_out"] = 5.011

    # --------------------------------------- LATE HALO --------------------------------------- #
    def cof_log_rho_core(self, time):
        """
        Find the coefficient value: `log(rho_core)`.

        :param time: moment of halo evolution.
        (float) [dimensionless]
        """
        log_bt = np.log10(self.beta * time)
        log_rho_core = self.new_param["A_rho_core"] * (log_bt - (self.new_param["E"] + 3)) ** 2 \
            + self.new_param["C_rho_core"] \
            + self.new_param["D_rho_core"] / (self.new_param["E"] + 0.0001 - log_bt) ** 0.02
        return log_rho_core

    def cof_log_r_core(self, time):
        """
        Find the coefficient value: `log(r_core)`.

        :param time: moment of halo evolution.
        (float) [dimensionless]
        """
        log_bt = np.log10(self.beta * time)
        log_r_core = self.new_param["A_r_core"] * (log_bt - (self.new_param["E"] + 2)) ** 2 \
            + self.new_param["C_r_core"] \
            + self.new_param["D_r_core"] / (self.new_param["E"] + 0.0001 - log_bt) ** 0.005
        return log_r_core

    def cof_log_r_out(self, time):
        """
        Find the coefficient value: `log(r_out)`.

        :param time: moment of halo evolution.
        (float) [dimensionless]
        """
        log_bt = np.log10(self.beta * time)  # [dimensionless]
        log_r_core = self.new_param["A_r_out"] * (log_bt - (self.new_param["E"] + 2)) ** 2 \
            + self.new_param["C_r_out"] \
            + self.new_param["D_r_out"] / (self.new_param["E"] + 0.04 - log_bt) ** 0.005
        return log_r_core

    def rho_late(self, r, time):
        """
        Density profile in regime, when core starting to contracting.

        :param time: moment of halo evolution.
        (float) [dimensionless]
        :param r: radius.
        (float) [r_s]
        """
        # --- take constants
        rho_core = 10 ** self.cof_log_rho_core(time)
        r_core = 10 ** self.cof_log_r_core(time)
        r_out = 10 ** self.cof_log_r_out(time)
        # --- calculate
        rho = rho_core / (1 + (r / r_core) ** self.new_param["s"] * (1 + r / r_out) ** (3 - self.new_param["s"]))
        return rho

    # --------------------------------------- EARLY HALO --------------------------------------- #
    def cof_log_r_core_early(self, time):
        """
        Find the coefficient value: `log(r_core_early)`.

        :param time: moment of halo evolution.
        (float) [dimensionless]
        """
        log_bt = np.log10(self.beta * time)
        log_r_core = self.new_param["A_r_core_early"] * log_bt ** 2 + self.new_param["B_r_core_early"] * log_bt \
            + self.new_param["C_r_core_early"]
        return log_r_core

    def rho_early(self,  r, time):
        """
        Density profile in regime, when core is expanding.

        :param time: moment of halo evolution.
        (float) [dimensionless]
        :param r: radius.
        (float) [r_s]
        """
        # --- take constants
        r_core = 10 ** self.cof_log_r_core_early(time)
        # --- calculate
        rho = np.tanh(r / r_core) / (r * (1 + r) ** 2)
        return rho

    # --------------------------------------- FIND DENSITY --------------------------------------- #
    def rho(self, radi, time):
        """
        Find the density value with fixed time and radi.

        :param time: moment of halo evolution.
        (float) [dimensionless]
        :param radi: radius.
        (float or array) [r_s]
        """
        log_bt = np.log10(self.beta * time)
        if isinstance(radi, float):
            # check in which regime we have already been
            if log_bt < self.new_param["F"]:
                rho = self.rho_early(radi, time)
            else:
                rho = self.rho_late(radi, time)
        elif isinstance(radi, list):
            # check in which regime we have already been
            if log_bt < self.new_param["F"]:
                rho = [self.rho_early(r, time) for r in radi]
            else:
                rho = [self.rho_late(r, time) for r in radi]
        else:
            raise TypeError("Expected float or list of floats")
        return rho

    def rho_model(self, r, time):
        """
        Find the density profile.

        :param time: moment of halo evolution.
        (float or array) [dimensionless]
        :param r: radius.
        (float or array) [r_s]
        ----------------------------
        len(r) == len(time): they should have same length!
        """
        log_bt = np.log10(self.beta * time)
        return np.where(log_bt < self.new_param["F"], self.rho_early(r, time), self.rho_late(r, time))

    # --------------------------------------- RETURN --------------------------------------- #
    def return_coefficients(self, time):
        """
        Give me coefficients, which corresponds to the halo in fixed moment of time.

        :param time: moment of halo evolution.
        (float) [dimensionless]
        """
        log_bt = np.log10(self.beta * time)
        # check in which regime we have already been
        if log_bt < self.new_param["F"]:
            radius_core = 10 ** self.cof_log_r_core_early(time)
            return [True, radius_core]
        else:
            r_core = 10 ** self.cof_log_r_core(time)
            r_out = 10 ** self.cof_log_r_out(time)
            rho_core = 10 ** self.cof_log_rho_core(time)
            return [False, r_core, r_out, rho_core]

    def return_time_from_log_bt(self, log_bt_val):
        """
        Let assume that I fixed: `log(beta * t^)` and I want to get `t^`.

        :param log_bt_val: value of `log(beta * t^)`
        (float) [dimensionless]
        """
        return 1 / self.beta * 10 ** log_bt_val  # [dimensionless]

    def return_time_transition(self):
        """
        I was interesting, when we move on to the core contracting regime.
        """
        return 1 / self.beta * 10 ** self.new_param["F"]  # [dimensionless]


# --- test our class
# Model = ModelGravothermal()
# time = Model.return_time_transition()
# print(time)

# Model = ModelGravothermal(beta=0.75)
# print("parameter E", Model.new_param["E"])
# print("parameter F", Model.new_param["F"])
#
# print("parameter A_r_core_early", Model.new_param["A_r_core_early"])
# print("parameter B_r_core_early", Model.new_param["B_r_core_early"])
# print("parameter C_r_core_early", Model.new_param["C_r_core_early"])
