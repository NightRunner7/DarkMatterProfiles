"""
In the code / name we used  to use shortcut: `ISO`, `Iso`, `iso` etc. all of that refers to
Isothermal SIDM Model / Isothermal profile. In this code `ISO` profile will connect to `hDM`
(halo DM) without taking into account baryons.

This file contains implementation of the Isothermal profile: `ISO`. The part of the code is
copied from (https://github.com/JiangFangzhou/SIDM). I find the more proper implementation,
which has been defined in class: `IsoEvolution`.
Contraction to the `NFW` case, here we will watch the evolution of the profile, which will
change in time: that also was the main problem how to get proper value of density and cross-section.
"""
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Union
# ------------ IMPORT FROM FILES ------------ #
import config as cfg
import auxiliaryFunctions as aux  # helpful functions
import units as uni  # helpful functions
from IsoAndHalo import IsoAndHalo

# ######################### MINIMIZE: `delta^2` AND ISOTHERMAL PROFILE ######################### #
def rhs_ode_CDM_only(y, x, a):
    """
    Returns the right-hand-side functions of the ODEs for solving the
    dimensionless SIDM (Self-Interacting Dark Matter) potential:

    The equations are defined as:

        rho_dm(x) / rho_dm0 := rho_hat = g(x)
        g''(x) = (1 / g(x)) * (g'(x))^2 - (2 / x) * g'(x) - a * (g(x))^2

    where:

        rho_dm0 : Central density
        x       : Dimensionless radius, defined as r / r_s
        a       : Constant, defined as 4 * pi * G * rho_dm0 * r_s^2 / sigma_0^2

    with:
        - sigma_0: Velocity dispersion in the isothermal core
        - G      : Gravitational constant [kpc^3 Gyr^-2 M_sun^-1]

    Syntax:

        rhs_ode_CDM_only(y, x, a)

    Parameters:
        y (list or array): [g(x), g'(x)] - g(x) and its first derivative
        x (float)        : Dimensionless radius (r / r_s)
        a (float)        : Constant (4 * pi * G * rho_dm0 * r_s^2 / sigma_0^2)

    Returns:
        list: [g'(x), (1 / g(x)) * (g'(x))^2 - (2 / x) * g'(x) - a * (g(x))^2]

    Note:
        This function is intended to be used with scipy.integrate.odeint, e.g.,

        scipy.integrate.odeint(rhs_ode_CDM_only, y0, t, args=(a,))
    """
    rho_hat, d_rho_hat_dr = y
    d2_rho_hat_dr2 = - (2/x) * d_rho_hat_dr + (1/rho_hat) * d_rho_hat_dr ** 2 - a * rho_hat ** 2
    return [d_rho_hat_dr, d2_rho_hat_dr2]

def rho_iso_CDM_only(x: Union[np.ndarray, float], a: float) -> tuple:
    """
    Solve for the dimensionless SIDM (Self-Interacting Dark Matter) potential profile:

        rho_dm(x) / rho_dm0 = g(x)
        g''(x) = (1 / g(x)) * (g'(x))^2 - (2 / x) * g'(x) - a * (g(x))^2,

    where,

        rho_dm0 : Central density
        x       : Dimensionless radius, defined as r / r_s
        a       : Constant, defined as 4 * pi * G * rho_dm0 * r_s^2 / sigma_0^2

    with:
        - sigma_0: Velocity dispersion in the isothermal core
        - G      : Gravitational constant [kpc^3 Gyr^-2 M_sun^-1]

    Note:
        For odeint to work, the t variable (in this case, x) must be an array and
        must have the initial-value point as the first element of this array.
        Example usage with odeint:

            odeint(f, [0., -b/2.], t, args=(a, b))

    Syntax:

        rho_iso_CDM_only(x, a)

    Parameters:
        x (array or float): Radius in units of r_s in ascending order
        a (float)         : Constant (4 * pi * G * rho_dm0 * r_s^2 / sigma_0^2)

    Returns:
        array: The dimensionless density profile rho_dm(x) / rho0
    """
    if np.isscalar(x):
        x = np.array([1e-8, x])
    else:
        x = np.append(1e-8, x)

    # Set initial conditions
    rho_dm_initial = 1.0  # initial density in units [rho0]
    d_rho_dm_dr_initial = 0.0  # initial derivative of density
    initial_conditions = [rho_dm_initial, d_rho_dm_dr_initial]

    # Solve the ODE system
    solution = odeint(rhs_ode_CDM_only, initial_conditions, x, args=(a,))
    return solution[1:, 0]


def deltaSqare(p: np.ndarray, r_s, rhoCDM1, MCDM1, r: np.ndarray):
    """
    Evaluate the relative error in stitching the isothermal-core profile
    to the outer CDM-like profile.

    The relative error delta is defined as:

        delta = sqrt(delta_rho^2 + delta_M^2)
        deltaSqare = delta_rho^2 + delta_M^2

    where:

        delta_rho = | rho_iso(r_1) - rho_CDM(r_1) | / rho_CDM(r_1)
        delta_M = | M_iso(r_1) - M_CDM(r_1) | / M_CDM(r_1)

    Parameters:
        p (array or list): [log(rho_dm0), log(sigma_0)] in units [M_sun/kpc^3, kpc/Gyr]
        r_s (float)      : Scale radius [kpc]
        rhoCDM1 (float)  : CDM density to match at r_1 [M_sun/kpc^3]
        MCDM1 (float)    : CDM enclosed mass to match at r_1 [M_sun]
        r (array)        : Radii between 0 and r_1, where the SIDM profile is computed [kpc],
                           e.g., np.logspace(-3., np.log10(r1), 500)

    Returns:
        float: The squared relative error deltaSqare = delta_rho^2 + delta_M^2
    """
    # Convert log parameters to linear scale
    rhodm0 = 10. ** p[0]
    sigma0 = 10. ** p[1]

    # Solve differential equation for SIDM profile
    a = cfg.FourPiG * r_s ** 2 * rhodm0 / sigma0 ** 2
    x = np.divide(r, r_s)
    rho = rhodm0 * rho_iso_CDM_only(x, a)

    # Select function to calculate enclosed mass
    # M = aux.Miso_linspaceR(r, rho)
    # M = aux.Miso_simpson(r, rho)  # pretty low, so max(len(r)) = 500
    M = aux.Miso_logspaceR(r, rho)  # Should be sufficiently accurate

    # Calculate squared relative differences
    drho = (rho[-1] - rhoCDM1) / rhoCDM1
    dM = (M[-1] - MCDM1) / MCDM1
    return drho ** 2 + dM ** 2

# ######################### GET ISOTHERMAL PROFILE ######################### #
def stitchSIDM(r1, halo, N=500):
    """
    Find the isothermal SIDM core that is smoothly stitched to the
    CDM-like outskirt.

    Syntax:

        stitchSIDMcore(r1, halo, N=500)

    Parameters:
        r1 (float): The characteristic radius of a SIDM halo at which an average
                    particle is scattered once during the age of the halo [kpc].
        halo (object): The halo profile for the CDM-like outskirt (a density profile object).
                       Example in `NFWProfile.py`
        N (int, optional): Length of the isothermal-core profile to be returned. Default is 500.

    Returns:
        tuple: A tuple containing:
            - central DM density [M_sun/kpc^3] (float)
            - central DM velocity dispersion [kpc/Gyr] (float)
            - density profile out to r1 [M_sun/kpc^3] (array of length N)
            - circular velocity profile out to r1 [M_sun] (array of length N)
            - radii for the profile [kpc] (array of length N)
    """
    # Prepare a few quantities
    sigmaCDM1 = halo.sigma_accurate(r1)
    rhoCDM1 = halo.rho(r1)
    MCDM1 = halo.Mass(r1)
    rhoCDMres = halo.rho(cfg.Rres)
    r_s = halo.r_s

    # Set the options for the Powell method
    name_method = 'Powell'  # 'Powell', 'L-BFGS-B'
    options = {
        # 'ftol': 1e-6, 'maxiter': 10000, 'xtol': 1e-1
    }

    # Specify the initial guess and the searching range
    lgrhodm0_init = 0.5 * (np.log10(rhoCDM1) + np.log10(rhoCDMres))
    lgsigma0_init = np.log10(sigmaCDM1)
    lgrhodm0_lo = np.log10(rhoCDM1)
    lgrhodm0_hi = 0.5 * (np.log10(rhoCDM1) + np.log10(rhoCDMres) + np.log10(1e2))  # <<< test: upper bound
    lgsigma0_lo = np.log10(0.5 * sigmaCDM1)
    lgsigma0_hi = np.log10(2.0 * sigmaCDM1)

    # Define the radius array
    r = np.logspace(np.log10(r_s * 1e-3), np.log10(r1), N)

    # Minimize
    res = minimize(deltaSqare, np.array([lgrhodm0_init, lgsigma0_init]),
                   args=(r_s, rhoCDM1, MCDM1, r),
                   bounds=((lgrhodm0_lo, lgrhodm0_hi), (lgsigma0_lo, lgsigma0_hi)),
                   method=name_method,
                   options=options
                   )

    # Compute the profile to be returned
    rhodm0 = 10. ** res.x[0]
    sigma0 = 10. ** res.x[1]
    a = cfg.FourPiG * r_s ** 2 * rhodm0 / sigma0 ** 2
    rho = rhodm0 * rho_iso_CDM_only(r / r_s, a)
    M = aux.Miso_simpson(r, rho)
    Vc = np.sqrt(cfg.const_G_starUnits * M / r)

    return rhodm0, sigma0, rho, Vc, r, M


def findTwoDeltaMinimum(r1, halo, N=500, r_low=-3.0):
    """
    Find the two minimum (in two different region), which occurred in plot log10(delta). This function
    uses the isothermal SIDM core, which is stitched smoothly to the CDM-like outskirt.
    Important note!: this works only for case: CDM-only.

    Syntax:

       findTwoDeltaMinimum(r1, halo, N=500, r_low=-3.0)

    Parameters:
       r1 (float): The characteristic radius of a SIDM halo at which an average
                   particle is scattered once during the age of the halo [kpc].
       halo (object): The halo profile for the CDM-like outskirt
                      (a density profile object). We have in NFW_class!!!
       N (int, optional): Length of the isothermal-core profile to be returned.
                          Default is 500.
       r_low (float, optional): The starting radius for calculations.
                                Default is -3.0.

    Returns:
       list: A list containing two sets of data:
           - For low density region: [rhodm0_LoDens, sigma0_LoDens, rho_LoDens, Vc_LoDens, r, M]
           - For high density region: [rhodm0_HiDens, sigma0_HiDens, rho_HiDens, Vc_HiDens, r, M]
    """
    # evaluate vel dispersion, density, and enclosed mas at r_1
    sigmaCDM1 = halo.sigma_accurate(r1)
    rhoCDM1 = halo.rho(r1)
    MCDM1 = halo.Mass(r1)
    rhoCDMres = halo.rho(cfg.Rres)
    r_s = halo.r_s

    # radius series over which we perform the integration
    r = np.logspace(r_low, np.log10(r1), N)

    # -------------------- PHYSICAL: LOW DENSE (CENTRAL VALUES) -------------------- #
    lgrhodm0_init = 0.5 * (np.log10(rhoCDM1) + np.log10(rhoCDMres))
    lgsigma0_init = np.log10(sigmaCDM1)
    lgrhodm0_lo = np.log10(rhoCDM1)
    lgrhodm0_hi = np.log10(rhoCDM1 * 1e3)  # <<< test: upper bound
    lgsigma0_lo = np.log10(0.5 * sigmaCDM1)
    lgsigma0_hi = np.log10(2.0 * sigmaCDM1)
    res = minimize(deltaSqare, np.array([lgrhodm0_init, lgsigma0_init]),
                   args=(r_s, rhoCDM1, MCDM1, r),
                   bounds=((lgrhodm0_lo, lgrhodm0_hi), (lgsigma0_lo, lgsigma0_hi)),
                   )
    # Compute the profile to be returned
    rhodm0 = 10.**res.x[0]
    sigma0 = 10.**res.x[1]
    a = cfg.FourPiG * r_s ** 2 * rhodm0 / sigma0 ** 2
    rho = rhodm0 * rho_iso_CDM_only(r / r_s, a)
    M = aux.Miso_simpson(r, rho)
    Vc = np.sqrt(cfg.const_G_starUnits * M / r)
    # register
    rhodm0_LoDens = rhodm0
    sigma0_LoDens = sigma0
    rho_LoDens = rho
    Vc_LoDens = Vc
    # complete data
    data_loDens = [rhodm0_LoDens, sigma0_LoDens, rho_LoDens, Vc_LoDens, r, M]

    # -------------------- PHYSICAL: HIGH DENSE (CENTRAL VALUES) -------------------- #
    #   Also note that there could an even higher-density solution,
    #   which we have essentially excluded from the searching range
    lgrhodm0_init = np.log10(rhoCDMres)
    lgsigma0_init = np.log10(2. * sigmaCDM1)

    # --------------------------- LIMIT TO DEAL WITH --------------------------- #
    # lower limit for rho0
    lgrhodm0_lo = np.log10(1e2 * rhoCDM1)
    # lgrhodm0_lo = np.log10(1.06 * rhodm0_LoDens)
    # upper limit for rho0
    # lgrhodm0_hi = np.log10(1e4 * rhoCDMres)
    lgrhodm0_hi = np.log10(1e2 * rhoCDMres)

    # limit for velocity disparsion are good/hold
    lgsigma0_lo = np.log10(0.5 * sigmaCDM1)
    lgsigma0_hi = np.log10(2.0 * sigmaCDM1)

    res = minimize(deltaSqare, np.array([lgrhodm0_init, lgsigma0_init]),
                   args=(r_s, rhoCDM1, MCDM1, r),
                   bounds=((lgrhodm0_lo, lgrhodm0_hi), (lgsigma0_lo, lgsigma0_hi)),
                   method='Powell',  # <<< important !
                   )
    # Compute the profile to be returned
    rhodm0 = 10.**res.x[0]
    sigma0 = 10.**res.x[1]
    a = cfg.FourPiG * r_s ** 2 * rhodm0 / sigma0 ** 2
    rho = rhodm0 * rho_iso_CDM_only(r / r_s, a)
    M = aux.Miso_simpson(r, rho)
    Vc = np.sqrt(cfg.const_G_starUnits * M / r)
    # register
    rhodm0_HiDens = rhodm0
    sigma0_HiDens = sigma0
    rho_HiDens = rho
    Vc_HiDens = Vc
    # complete data
    data_HiDens = [rhodm0_HiDens, sigma0_HiDens, rho_HiDens, Vc_HiDens, r, M]

    # return both minimum
    return [data_loDens, data_HiDens]

# ######################### ISOTHERMAL CLASS ######################### #
class IsoEvolution(object):
    """
    Find the isothermal SIDM core that is stitched smoothly to the
    CDM-like outskirt. One important remark: this class dealing
    with `CDM-only` case. In the other words we do not include baryons.

    Our class has destined to describe the evolution of the `SIDM Isothermal core`
    case. So, firstly we have to input the initial CDM profile (we will use NFW,
    to more details go to file `NFWProfile.py`) - now I want to make one comment
    the stage of evolution is fixed by value of `r1`, computed using `NFW` halo.

        r1(t1) > r1(t2) if t2 < t1

    Also, `r1` separates the halo profile into two regions:

        r < r1: described by `SIDM Isothermal core` (sometimes named by: `Isothermal`).
        r > r1: described by `NFW` (halo dark matter).
    """
    def __init__(self, _halo, r1, nr=500, cff=0.001, rel_err_mergin=1.0 + 40e-2):
        """
        Set initial halo and evolution stage of `SIDM Isothermal core`.

        where,

            r1: the characteristic radius of a SIDM halo at which an average
                particle is scattered once during the age of the halo [kpc]
                (float)
            halo: the halo profile for the CDM-like outskirt
                (object).
            nr: number of radius's, which we use in the minimization procedure.
                (int)
            cff: value of coefficient, which we multiply by `r_s`. Default value is: `0.001`.
                More details in the `config.py`. Set the spatial resolution.
                (float)
            rel_err_mergin: the relative difference where merging should be occurred.
                (float)
        """
        # --- selecting merging: when `Low dense` become `High dense`
        self.merged_appear = False  # flag to find the first time when physical and unphysical have comparable values
        self.Is_merging = False  # flag to check that `Low dense` solution becomes the `high dense`.
        self.rel_err_mergin = rel_err_mergin

        # --- find Rres: spatial resolution
        self.r_s = _halo.r_s
        Rres = cfg.find_Rres(_halo, cff=cff)  # [kpc]
        # radi bin in the class: to somehow stick accuracy of calculation, one order of magnitude lower than resolution
        self.r_low = np.log10(cff) - 1.0
        self.nr = nr  # how many radius's in bin

        # --- Cold Dark Matter properties at r1
        self.halo = _halo
        self.sigmaCDM1 = self.halo.sigma_accurate(r1)  # [kpc/Gyr]
        self.rhoCDM1 = self.halo.rho(r1)  # [M_sun/kpc^3]
        self.MCDM1 = self.halo.Mass(r1)  # [M_sun]
        self.rhoCDMres = self.halo.rho(Rres)  # [M_sun/kpc^3]
        self.r1 = r1  # [kpc]

        # --- Set initial central value: physical value
        self.rho0 = 0.0  # [M_sun/kpc^3]
        self.sigma0 = 0.0  # [kpc/Gyr]
        # --- Central value from previous step: physical value
        self.rho0_pre = 0.0  # [M_sun/kpc^3]
        self.sigma0_pre = 0.0  # [kpc/Gyr]
        # --- Set initial central value: unphysical value
        self.rho0_HiDen = 0.0  # [M_sun/kpc^3]
        self.sigma0_HiDen = 0.0  # [kpc/Gyr]
        # --- Central value from previous step: unphysical value
        self.rho0_HiDen_pre = 0.0  # [M_sun/kpc^3]
        self.sigma0_HiDen_pre = 0.0  # [kpc/Gyr]

        # --- List which saves values of log delta of LoDen and HiDen during evolution of our Isothermal profile
        self.logDeltaLoDen_arr = np.array([])
        self.logDeltaHiDen_arr = np.array([])

        # --- we have to call the function it finds initial values of central parameters
        # Set the options for the Powell method
        self.options = {
            # 'ftol': 1e-5,
            # 'maxiter': 10000,
        }
        self.initial_guess()

        # --- Isothermal data, created during evolution
        self.IsoAndNFW_data = dict()
        self.IsoAndNFW_data['time'] = np.array([])  # radius [kpc]
        self.IsoAndNFW_data['r'] = np.array([])  # radius [kpc]
        self.IsoAndNFW_data["rho"] = np.array([])  # density [M_sun/kpc^3]
        self.IsoAndNFW_data["mass"] = np.array([])  # enclosed mass [M_sun]
        self.IsoAndNFW_data["velDis"] = np.array([])  # nu in [kpc/Gyr]
        self.IsoAndNFW_data["central-rho"] = np.array([])  # central density [M_sun/kpc^3]
        self.IsoAndNFW_data["central-velDis"] = np.array([])  # central velocity dispersion [kpc/Gyr]
        self.IsoAndNFW_data["r1-check-merge"] = None  # [kpc]
        self.params_at_tmerge = None  # all parameters at merge

    # ########################################### STEP I: INITIALIZATION ########################################### #
    # -------------------- SETTING CENTRAL VALUES -------------------- #
    def initial_guess(self, nr=500, r_low=-4.0, initSetUp=True):
        """
        Find initial value of central density and central velocity dispersion,
        `initial` means corresponding to the initial halo of dark matter and
        value of `r1`, which describes the evolution stage of the `Isothermal core`.

        where,

            nr: number of radius's, which we use in the minimization procedure.
                (int)
            r_low: 10^r_low is the smallest radi, which we consider during
                minimization procedure. Take care about units: [10^r_low] = [r_s]
                (float)
            initSetUp: the settings of `nr` and `r_low` comes from the init.
                (bool)

        The default values should be enough to find proper value, but just in case
        (or in need) is easy to change it and get better accuracy.
        ----------
        In order to obtain the central values of central density and central velocity
        dispersion, we have to set initial guesses and borders of searching. To do it,
        we will use:

            sigmaCDM1 = halo.sigma(r1): velocity dispersion from CDM halo at `r1`.
                [kpc/Gyr]
            rhoCDM1 = halo.rho(r1): density from CDM halo at `r1`.
                [M_sun/kpc^3]
            rhoCDMres = halo.rho(Rres): density from CDM halo at `Rres`. `Rres` usually
                denotes the smallest radi, which we take into account during our procedure.
                In the other words, we stick boundary of our accuracy.
                [M_sun/kpc^3]

        For both cases ('physical' and 'unphysical') we use the same boundaries
        (these boundaries are well-defined, so we do not have to change it):

            log10(sigma0) init guess: log10(sigmaCDM1)
            log10(sigma0) belongs to: [log10(sigmaCDM1) * 0.5, log10(sigmaCDM1) * 2.0]

        When it comes to set boundaries for central density it has a little more tricky.
        We set that 'physical' central density has to be between two values: rhoCDM1 and rhoCDMres,
        this time between means in the `logarithmic manner`. It means:

            log10(rho0) init guess: 0.5 * log10(rhoCDM1) + 0.5 * log10(rhoCDMres)
            log10(rho0) belongs to: [log10(rhoCDM1), log10(rhoCDM1 * 1e3)] <--- should be near the halo CDM.

        The 'unphysical' central density should occur near our accuracy of calculation, so the
        boundaries as follows:

            log10(rho0_HiDen) init guess: log10(rhoCDMres)
            log10(rho0_HiDen) belongs to: [log10(rhoCDM1 * 1e2), log10(rhoCDMres * 1e2)]
        """
        # radius series over which we perform the integration
        if initSetUp is True:
            r = np.logspace(np.log10(self.r_s * 10 ** self.r_low), np.log10(self.r1), self.nr)
        else:
            r = np.logspace(np.log10(self.r_s * 10 ** r_low), np.log10(self.r1), nr)

        # -------------------- PHYSICAL: LOW DENSE (CENTRAL VALUES) -------------------- #
        # initial guess (we can work)
        lg_rho0_init = 0.5 * (np.log10(self.rhoCDM1) + np.log10(self.rhoCDMres))
        lg_sigma0_init = np.log10(self.sigmaCDM1)
        # density region (we can work)
        lg_rho0_lo = np.log10(self.rhoCDM1)
        lg_rho0_hi = np.log10(self.rhoCDM1 * 1e3)  # <<< test: upper bound
        # velocity dispersion region (good)
        lg_sigma0_lo = np.log10(0.05 * self.sigmaCDM1)
        lg_sigma0_hi = np.log10(2.0 * self.sigmaCDM1)

        # minimize `deltaSquare` get central density and central velocity dispersion
        res = minimize(deltaSqare, np.array([lg_rho0_init, lg_sigma0_init]),
                       args=(self.r_s, self.rhoCDM1, self.MCDM1, r),
                       bounds=((lg_rho0_lo, lg_rho0_hi), (lg_sigma0_lo, lg_sigma0_hi)),
                       method='Powell',  # <<< important! Powell, Trust-Constr, L-BFGS-B, Nelder-Mead
                       options=self.options)
        rho0 = 10. ** res.x[0]  # [M_sun/kpc^3] central DM density
        sigma0 = 10. ** res.x[1]  # [kpc/Gyr] central DM velocity dispersion

        # --- update self data
        self.rho0 = rho0
        self.sigma0 = sigma0

        # -------------------- PHYSICAL: HIGH DENSE (CENTRAL VALUES) -------------------- #
        # initial guess (we can work)
        lg_rho0_init = np.log10(self.rhoCDMres)
        lg_sigma0_init = np.log10(2.0 * self.sigmaCDM1)
        # density region (we can work)
        lg_rho0_lo = np.log10(self.rhoCDM1 * 1e2)
        lg_rho0_hi = np.log10(self.rhoCDMres * 1e2)
        # velocity dispersion region (good)
        lg_sigma0_lo = np.log10(0.05 * self.sigmaCDM1)
        lg_sigma0_hi = np.log10(2.0 * self.sigmaCDM1)

        # minimize `deltaSquare` get central density and central velocity dispersion
        res = minimize(deltaSqare, np.array([lg_rho0_init, lg_sigma0_init]),
                       args=(self.r_s, self.rhoCDM1, self.MCDM1, r),
                       bounds=((lg_rho0_lo, lg_rho0_hi), (lg_sigma0_lo, lg_sigma0_hi)),
                       method='Powell',  # <<< important! Powell, Trust-Constr,L-BFGS-B, Nelder-Mead
                       options=self.options)
        rho0 = 10. ** res.x[0]  # [M_sun/kpc^3] central DM density
        sigma0 = 10. ** res.x[1]  # [kpc/Gyr] central DM velocity dispersion

        # --- Central value from the current step
        self.rho0_HiDen = rho0
        self.sigma0_HiDen = sigma0

    # ########################################### STEP II: EVOLUTION STEP ########################################### #
    def new_evolution_step(self, r1, nr=500, r_low=-4.0, initSetUp=True):
        """
        Now we interested in following the evolution of `SIMD Isothermal core`.
        If the value of `r1` has been changed that means we have new step (in time)
        in the evolution. So, it means that we have to update some variables, for
        e.g.: the density of halo CDM at `r1` might be changed.

        where,

            r1: the characteristic radius of a SIDM halo at which an average
                particle is scattered once during the age of the halo.
                (float) [kpc]
            nr: number of radius's, which we use in the minimization procedure.
                (int)
            r_low: 10^r_low is the smallest radi, which we consider during
                minimization procedure. Take care about units: [10^r_low] = [r_s]
                (float)
            initSetUp: the settings of `nr` and `r_low` comes from the init.
                (bool)

        The default values should be enough to find proper value, but just in case
        (or in need) is easy to change it and get better accuracy.
        ----------------
        When it comes to finding new position of central density and central velocity
        dispersion: we use previous values (of those parameters) as initial guesses.

        In our procedure we have one more tricky part: the setting high boundary and
        low boundary for 'physical' and 'unphysical' central density respectively. We
        set that this boundary is the same as defined as follows:

            boundary = 0.5 * log10(rho0_pre) + 0.5 * log10(rho0_HiDen_pre)

        for 'physical' in upper boundary, for 'unphysical' is lower boundary (remember
        that boundary is corresponding to `log10(rho)`).

        One at least but last: if during evolution you will see at some point,
        that `rho0_HiDen` has the value upper that `rhoCDM1` you do not bother
        about. This situation means, in general, that at this time we haven't
        right accuracy -> so above this line the 'unphysical' central density
        can behave relly wired.
        """
        # --- Central value from previous step: physical
        self.rho0_pre = self.rho0  # [M_sun/kpc^3]
        self.sigma0_pre = self.sigma0  # [kpc/Gyr]
        # --- Central value from previous step: physical
        self.rho0_HiDen_pre = self.rho0_HiDen  # [M_sun/kpc^3]
        self.sigma0_HiDen_pre = self.sigma0_HiDen  # [kpc/Gyr]
        # update values corresponding to change place of r1
        self.sigmaCDM1 = self.halo.sigma_accurate(r1)  # [kpc/Gyr]
        self.rhoCDM1 = self.halo.rho(r1)  # [M_sun/kpc^3]
        self.MCDM1 = self.halo.Mass(r1)  # [M_sun]
        self.r1 = r1  # [kpc]

        # radius series over which we perform the integration
        if initSetUp is True:
            r = np.logspace(np.log10(self.r_s * 10 ** self.r_low), np.log10(self.r1), self.nr)
        else:
            r = np.logspace(np.log10(self.r_s * 10 ** r_low), np.log10(self.r1), nr)

        # -------------------- CHECK THAT LOW DENSE IS HIGH DENSE -------------------- #
        self.Is_merging = aux.calRelErr_mergin(self.rho0_HiDen, self.rho0, self.rel_err_mergin)

        if self.Is_merging is False:
            """
            Physical situation, when `Low dense` can be distinguishable to the `High dense`
            solution. 
            """

            # -------------------- PHYSICAL: LOW DENSE (CENTRAL VALUES) -------------------- #
            # initial guess (we can work)
            lg_rho0_init = np.log10(self.rho0_pre)
            lg_sigma0_init = np.log10(self.sigma0_pre)
            # density region (we can work)
            lg_rho0_lo = np.log10(self.rho0_pre * 1e-1)
            if self.rho0_HiDen_pre > self.rhoCDMres:
                lg_rho0_hi = np.log10(self.rhoCDMres * 8 * 1e-1)
            else:
                lg_rho0_hi = 0.5 * (np.log10(self.rho0_pre) + np.log10(self.rho0_HiDen_pre))
            # velocity dispersion region (good)
            lg_sigma0_lo = np.log10(0.05 * self.sigmaCDM1)
            lg_sigma0_hi = np.log10(2.0 * self.sigmaCDM1)

            # minimize `deltaSquare` get central density and central velocity dispersion
            res = minimize(deltaSqare, np.array([lg_rho0_init, lg_sigma0_init]),
                           args=(self.r_s, self.rhoCDM1, self.MCDM1, r),
                           bounds=((lg_rho0_lo, lg_rho0_hi), (lg_sigma0_lo, lg_sigma0_hi)),
                           method='Powell',  # <<< important ! Powell, Trust-Constr, L-BFGS-B, Nelder-Mead
                           options=self.options)
            rho0 = 10. ** res.x[0]  # [M_sun/kpc^3] central DM density
            sigma0 = 10. ** res.x[1]  # [kpc/Gyr] central DM velocity dispersion
            logDelta = np.log10(res.fun) * 1 / 2  # log delta value after minimization procedure

            # --- Central value from the current step
            self.rho0 = rho0
            self.sigma0 = sigma0
            self.logDeltaLoDen_arr = np.append(self.logDeltaLoDen_arr, logDelta)

            # -------------------- PHYSICAL: HIGH DENSE (CENTRAL VALUES) -------------------- #
            # initial guess (we can work)
            lg_rho0_init = np.log10(self.rho0_HiDen_pre)
            lg_sigma0_init = np.log10(self.sigma0_HiDen_pre)
            # density region (we can work)
            lg_rho0_lo = 0.5 * (np.log10(self.rho0_HiDen_pre) + np.log10(self.rho0_pre))
            if self.rho0_HiDen_pre > self.rhoCDMres:
                # lg_rho0_hi = np.log10(self.rhoCDMres * 1e2)  # sometimes works, sometimes not
                lg_rho0_hi = np.log10(self.rho0_HiDen_pre)
            else:
                # print("GUESS: ", self.rho0_HiDen_pre, "[M_sun/kpc^3]")
                lg_rho0_hi = np.log10(self.rhoCDMres)
            # velocity dispersion region (good)
            lg_sigma0_lo = np.log10(0.05 * self.sigmaCDM1)
            lg_sigma0_hi = np.log10(2.0 * self.sigmaCDM1)

            # minimize `deltaSquare` get central density and central velocity dispersion
            res = minimize(deltaSqare, np.array([lg_rho0_init, lg_sigma0_init]),
                           args=(self.r_s, self.rhoCDM1, self.MCDM1, r),
                           bounds=((lg_rho0_lo, lg_rho0_hi), (lg_sigma0_lo, lg_sigma0_hi)),
                           method='Powell',  # <<< important ! Powell,  Trust-Constr, L-BFGS-B, Nelder-Mead
                           )
            rho0 = 10. ** res.x[0]  # [M_sun/kpc^3] central DM density
            sigma0 = 10. ** res.x[1]  # [kpc/Gyr] central DM velocity dispersion
            logDelta = np.log10(res.fun) * 1 / 2  # log delta value after minimization procedure

            # --- Central value from the current step
            self.rho0_HiDen = rho0
            self.sigma0_HiDen = sigma0
            self.logDeltaHiDen_arr = np.append(self.logDeltaHiDen_arr, logDelta)

            # --- save value of r1
            self.IsoAndNFW_data["r1-check-merge"] = r1
        else:
            """
            Unphysical situation, when `Low dense` can not be distinguishable to the `High dense`
            solution. 
            """
            # initial guess (we can work)
            lg_rho0_init = np.log10(self.rho0_HiDen_pre)
            lg_sigma0_init = np.log10(self.sigma0_HiDen_pre)
            # density region (we can work)
            lg_rho0_lo = np.log10(self.rhoCDM1)
            lg_rho0_hi = np.log10(self.rhoCDMres)
            # velocity dispersion region (good)
            lg_sigma0_lo = np.log10(0.05 * self.sigmaCDM1)
            lg_sigma0_hi = np.log10(2.0 * self.sigmaCDM1)

            # minimize `deltaSquare` get central density and central velocity dispersion
            res = minimize(deltaSqare, np.array([lg_rho0_init, lg_sigma0_init]),
                           args=(self.r_s, self.rhoCDM1, self.MCDM1, r),
                           bounds=((lg_rho0_lo, lg_rho0_hi), (lg_sigma0_lo, lg_sigma0_hi)),
                           method='Powell',  # <<< important ! Powell, Trust-Constr, L-BFGS-B, Nelder-Mead
                           options=self.options)
            rho0 = 10. ** res.x[0]  # [M_sun/kpc^3] central DM density
            sigma0 = 10. ** res.x[1]  # [kpc/Gyr] central DM velocity dispersion
            logDelta = np.log10(res.fun) * 1 / 2  # log delta value after minimization procedure

            # --- Central value from the current step: High dense
            self.rho0_HiDen = rho0
            self.sigma0_HiDen = sigma0
            self.logDeltaLoDen_arr = np.append(self.logDeltaLoDen_arr, logDelta)
            # --- Central value from the current step: Low dense
            self.rho0 = rho0
            self.sigma0 = sigma0
            self.logDeltaHiDen_arr = np.append(self.logDeltaHiDen_arr, logDelta)

    def new_evolution_step_naive(self, r1, nr=500, r_low=-4.0, initSetUp=True):
        """
        Evolution by educated guess.

        Now we interested in following the evolution of `SIMD Isothermal core`.
        If the value of `r1` has been changed that means we have new step (in time)
        in the evolution. So, it means that we have to update some variables, for
        e.g.: the density of halo CDM at `r1` might be changed. Version of guy

        where,

            r1: the characteristic radius of a SIDM halo at which an average
                particle is scattered once during the age of the halo.
                (float) [kpc]
            nr: number of radius's, which we use in the minimization procedure.
                (int)
            r_low: 10^r_low is the smallest radi, which we consider during
                minimization procedure. Take care about units: [10^r_low] = [r_s]
                (float)
            initSetUp: the settings of `nr` and `r_low` comes from the init.
                (bool)

        The default values should be enough to find proper value, but just in case
        (or in need) is easy to change it and get better accuracy.
        """
        # evaluate vel dispersion, density, and enclosed mas at r_1
        self.sigmaCDM1 = self.halo.sigma_accurate(r1)  # [kpc/Gyr]
        self.rhoCDM1 = self.halo.rho(r1)  # [M_sun/kpc^3]
        self.MCDM1 = self.halo.Mass(r1)  # [M_sun]
        self.r1 = r1  # [kpc]

        # radius series over which we perform the integration
        if initSetUp is True:
            r = np.logspace(np.log10(self.r_s * 10 ** self.r_low), np.log10(self.r1), self.nr)
        else:
            r = np.logspace(np.log10(self.r_s * 10 ** r_low), np.log10(self.r1), nr)

        # -------------------- PHYSICAL: LOW DENSE (CENTRAL VALUES) -------------------- #
        lgrhodm0_init = 0.5 * (np.log10(self.rhoCDM1) + np.log10(self.rhoCDMres))
        lgsigma0_init = np.log10(self.sigmaCDM1)
        lgrhodm0_lo = np.log10(self.rhoCDM1)
        lgrhodm0_hi = np.log10(self.rhoCDM1*10**2)  # <<< test: upper bound
        # lgrhodm0_hi = 0.5 * (np.log10(self.rhoCDM1) + np.log10(self.rhoCDMres) + np.log10(7e-1))  # <<< test: upper bound
        lgsigma0_lo = np.log10(0.05 * self.sigmaCDM1)
        lgsigma0_hi = np.log10(2.0 * self.sigmaCDM1)
        res = minimize(deltaSqare, np.array([lgrhodm0_init, lgsigma0_init]),
                       args=(self.r_s, self.rhoCDM1, self.MCDM1, r),
                       bounds=((lgrhodm0_lo, lgrhodm0_hi), (lgsigma0_lo, lgsigma0_hi)),
                       )
        rhodm0 = 10. ** res.x[0]
        sigma0 = 10. ** res.x[1]
        logDelta = np.log10(res.fun) * 1 / 2  # log delta value after minimization procedure
        # register
        self.rho0 = rhodm0
        self.sigma0 = sigma0
        self.logDeltaLoDen_arr = np.append(self.logDeltaLoDen_arr, logDelta)

        # -------------------- PHYSICAL: HIGH DENSE (CENTRAL VALUES) -------------------- #
        #   Also note that there could an even higher-density solution,
        #   which we have essentially excluded from the searching range
        lgrhodm0_init = np.log10(self.rhoCDMres)
        lgsigma0_init = np.log10(2. * self.sigmaCDM1)
        lgrhodm0_lo = np.log10(1e1 * self.rhoCDM1)
        lgrhodm0_hi = np.log10(1e1 * self.rhoCDMres)
        lgsigma0_lo = np.log10(0.05 * self.sigmaCDM1)
        lgsigma0_hi = np.log10(2.0 * self.sigmaCDM1)
        res = minimize(deltaSqare, np.array([lgrhodm0_init, lgsigma0_init]),
                       args=(self.r_s, self.rhoCDM1, self.MCDM1, r),
                       bounds=((lgrhodm0_lo, lgrhodm0_hi), (lgsigma0_lo, lgsigma0_hi)),
                       method='Powell',  # <<< important !
                       )
        rhodm0 = 10. ** res.x[0]
        sigma0 = 10. ** res.x[1]
        logDelta = np.log10(res.fun) * 1 / 2  # log delta value after minimization procedure
        # register
        self.rho0_HiDen = rhodm0
        self.sigma0_HiDen = sigma0
        self.logDeltaHiDen_arr = np.append(self.logDeltaHiDen_arr, logDelta)

    # ########################################### STEP III: GET DATA ########################################### #
    # ########################################### DURING EVOLUTION ########################################### #
    def prepare_ISO_data(self, _r, rho0, sigma0):
        """
        Prepare data, which describes the `SIDM Isothermal core` with
        fixed central density and central velocity dispersion.

        where,

            _r: radii between 0 and r_1, where we compute the SIDM profile
                [kpc] (array), e.g., np.logspace(-3.,np.log10(r1),500)
            rho0: central DM density.
                [M_sun/kpc^3] (float)
            sigma0: central DM velocity dispersion.
                [kpc/Gyr]
        ------------
        As you can see this is fully independent of value `r1`. It means you can
        firstly evolve `Isothermal` and save the values of `rho0` and `sigma0`
        """
        # prepare
        a = cfg.FourPiG * self.r_s ** 2 * rho0 / sigma0 ** 2
        x = np.divide(_r, self.r_s)
        rho = rho0 * rho_iso_CDM_only(x, a)
        # Select function to calculate enclosed mass
        M = aux.Miso_logspaceR(_r, rho)  # Should be sufficiently accurate
        Vc = np.sqrt(cfg.const_G_starUnits * M / _r)  # [kpc/Gyr] circular velocity

        data_ISO = [rho0, sigma0, rho, Vc, _r, M]
        return data_ISO

    def get_ISO_data_evolution(self, nr=500, r_low=-4.0, initSetUp=True):
        """
        Prepare data, which describes the `SIDM Isothermal core`, but during
        evolution. It means we will use the values of central density,
        velocity dispersion and `r1`, which are stored in class. In the other
        words those parameters, it describes actual step in `Isothermal` evolution.

        where,

            nr: number of radius's, which we use in the minimization procedure.
                (int)
            r_low: 10^r_low is the smallest radi, which we consider during
                minimization procedure. Take care about units: [10^r_low] = [r_s]
                (float)
            initSetUp: the settings of `nr` and `r_low` comes from the init.
                (bool)

        The default values should be enough to find proper value, but just in case
        (or in need) is easy to change it and get better accuracy.
        """
        # radius series over which we compute data
        if initSetUp is True:
            r = np.logspace(np.log10(self.r_s * 10 ** self.r_low), np.log10(self.r1), self.nr)
        else:
            r = np.logspace(np.log10(self.r_s * 10 ** r_low), np.log10(self.r1), nr)

        # ISO: `physical`
        ISO_LoDen = self.prepare_ISO_data(r, self.rho0, self.sigma0)
        # ISO: 'unphysical'
        ISO_HiDen = self.prepare_ISO_data(r, self.rho0_HiDen, self.sigma0_HiDen)

        return [ISO_LoDen, ISO_HiDen]

    def get_ISO_data(self, r, parms_LoDen, params_HiDen):
        """
         Prepare data, which describes the `SIDM Isothermal core`, with fixed
         values of central density, velocity dispersion and `r1`.

         where,
            r: radii between 0 and r_1, where we compute the SIDM profile
                [kpc] (array), e.g., np.logspace(-3.,np.log10(r1),500)

            parms_LoDen: parameters in the 'physical case' (low dense).
            parms_LoDen[0]: rho0: central DM density.
                [M_sun/kpc^3]
            parms_LoDen[1]: sigma0: central DM velocity dispersion.
                [kpc/Gyr]

            parms_LoDen: parameters in the 'unphysical case' (high dense).
            parms_LoDen[0]: rho0_HiDen: central DM density.
                [M_sun/kpc^3]
            parms_LoDen[1]: sigma0_HiDen: central DM velocity dispersion.
                [kpc/Gyr]
        """
        # ISO: `physical`
        rho0 = parms_LoDen[0]
        sigma0 = parms_LoDen[1]
        ISO_LoDen = self.prepare_ISO_data(r, rho0, sigma0)
        # ISO: 'unphysical'
        rho0_HiDen = params_HiDen[0]
        sigma0_HiDen = params_HiDen[1]
        ISO_HiDen = self.prepare_ISO_data(r, rho0_HiDen, sigma0_HiDen)

        return [ISO_LoDen, ISO_HiDen]

    # -------------------- RETURN VALUES -------------------- #
    def retrun_rho0_LoDen(self):
        """Return `physical` central density"""
        return self.rho0  # [M_sun/kpc^3]

    def return_sigma0_LoDen(self):
        """Return `physical` central velocity dispersion"""
        return self.sigma0  # [kpc/Gyr]

    def return_rho0_HiDen(self):
        """Return `unphysical` central density"""
        return self.rho0_HiDen  # [M_sun/kpc^3]

    def return_sigma0_HiDen(self):
        """Return `unphysical` central velocity dispersion"""
        return self.sigma0_HiDen  # [kpc/Gyr]

    def return_few_rho_LoDen(self, elements=2, nr=500, r_low=-4.0, initSetUp=True):
        """return first few values of density (Low Dense)"""
        # radius series over which we compute data
        if initSetUp is True:
            r = np.logspace(np.log10(self.r_s * 10 ** self.r_low), np.log10(self.r1), self.nr)
        else:
            r = np.logspace(np.log10(self.r_s * 10 ** r_low), np.log10(self.r1), nr)
        # ISO: `physical`
        ISO_LoDen = self.prepare_ISO_data(r, self.rho0, self.sigma0)
        rho_arr = ISO_LoDen[2][0:elements]  # [M_sun/kpc^3]
        r_arr = ISO_LoDen[4][0:elements]  # [kpc]

        return [rho_arr, r_arr]

    def return_few_rho_HiDen(self, elements=2, nr=500, r_low=-4.0, initSetUp=True):
        """return first few values of density (High Dense)"""
        # radius series over which we compute data
        if initSetUp is True:
            r = np.logspace(np.log10(self.r_s * 10 ** self.r_low), np.log10(self.r1), self.nr)
        else:
            r = np.logspace(np.log10(self.r_s * 10 ** r_low), np.log10(self.r1), nr)
        # ISO: `unphysical`
        ISO_HiDen = self.prepare_ISO_data(r, self.rho0_HiDen, self.sigma0_HiDen)
        rho_arr = ISO_HiDen[2][0:elements]  # [M_sun/kpc^3]
        r_arr = ISO_HiDen[4][0:elements]  # [kpc]

        return [rho_arr, r_arr]

    def return_rho_core_LoDen(self, elements=2, nr=500, r_low=-4.0, initSetUp=True):
        """calculate core density (Low dense)"""
        # radius series over which we compute data
        if initSetUp is True:
            r = np.logspace(np.log10(self.r_s * 10 ** self.r_low), np.log10(self.r1), self.nr)
        else:
            r = np.logspace(np.log10(self.r_s * 10 ** r_low), np.log10(self.r1), nr)
        # ISO: `physical`
        ISO_LoDen = self.prepare_ISO_data(r, self.rho0, self.sigma0)
        rho_arr = ISO_LoDen[2][0:elements]  # [M_sun/kpc^3]
        r_arr = ISO_LoDen[4][0:elements]  # [kpc]
        # calculate core density
        rho_core = aux.calculate_rho_core(r_arr, rho_arr, elements)  # [M_sun/kpc^3]
        return rho_core

    def return_rho_core_HiDen(self, elements=2, nr=500, r_low=-4.0, initSetUp=True):
        """calculate core density (Low dense)"""
        # radius series over which we compute data
        if initSetUp is True:
            r = np.logspace(np.log10(self.r_s * 10 ** self.r_low), np.log10(self.r1), self.nr)
        else:
            r = np.logspace(np.log10(self.r_s * 10 ** r_low), np.log10(self.r1), nr)
        # ISO: `unphysical`
        ISO_HiDen = self.prepare_ISO_data(r, self.rho0_HiDen, self.sigma0_HiDen)
        rho_arr = ISO_HiDen[2][0:elements]  # [M_sun/kpc^3]
        r_arr = ISO_HiDen[4][0:elements]  # [kpc]
        # calculate core density
        rho_core = aux.calculate_rho_core(r_arr, rho_arr, elements)  # [M_sun/kpc^3]
        return rho_core

    def return_logDelta_evolution(self):
        """return stored value of log delta during evolution"""
        return [self.logDeltaLoDen_arr, self.logDeltaHiDen_arr]

    # ########################################### STEP IV: UPDATE DATA ########################################### #
    # ########################################### DURING EVOLUTION ########################################### #
    def update_central_data(self, lowDens=True):
        """central data"""
        if lowDens is True:
            # --- update vale of central data
            rho0 = self.rho0
            self.IsoAndNFW_data["central-rho"] = np.append(self.IsoAndNFW_data["central-rho"],
                                                           rho0)  # [M_sun/kpc^3]
            sigma0 = self.sigma0  # [kpc/Gyr]
            self.IsoAndNFW_data["central-velDis"] = np.append(self.IsoAndNFW_data["central-velDis"],
                                                              sigma0)  # [kpc/Gyr]
        else:
            # --- update vale of central data
            rho0 = self.rho0_HiDen
            self.IsoAndNFW_data["central-rho"] = np.append(self.IsoAndNFW_data["central-rho"],
                                                           rho0)  # [M_sun/kpc^3]
            sigma0 = self.sigma0_HiDen  # [kpc/Gyr]
            self.IsoAndNFW_data["central-velDis"] = np.append(self.IsoAndNFW_data["central-velDis"],
                                                              sigma0)  # [kpc/Gyr]

    def update_before_merging(self, _tage, nr=400, r_low=-2.0, r_up=2, lowDens=True):
        """
        Before merging `low dense` and `high dense` can be distinguishable. Save some important data
        during evolution steps.

        Where,

            nr: number of radius's, which we use in the minimization procedure.
                (int)
            r_low: 10^r_low is the smallest radi, which you want to consider / save
                in txt file. Typically: `r_low=-2.0`. Take care about units: [10^r_low] = [r_s]
                (float)
            r_up: 10^r_low is the biggest radi, which you want to consider / save
                in txt file. Typically: `r_up=2.0`. Take care about units: [10^r_low] = [r_s]
                (float)
            lowDens: To distinguishable two possible results, `LoDense` and `HiDense`.
        """
        # radi logspace: to get sufficient accuracy - comes from init
        rToISO = np.logspace(np.log10(self.r_s * 10 ** self.r_low),
                             np.log10(self.r1),
                             self.nr)  # [kpc]

        if lowDens is True:
            # --- and take data from memory
            ISO_LoDen = self.prepare_ISO_data(rToISO, self.rho0, self.sigma0)
            ISO_data = dict()
            ISO_data["r"] = ISO_LoDen[4]  # radius [kpc]
            ISO_data["rho"] = ISO_LoDen[2]  # density [M_sun/kpc^3]
            ISO_data["mass"] = ISO_LoDen[5]  # enclosed mass [M_sun]

            # --- Create `Iso and halo class`: to find out how looks profile, where we have to stitch two profiles.
            ClassIsoAndHalo = IsoAndHalo(ISO_data, self.halo, self.r1)
            IsoAndNFW_dataToUpdate = ClassIsoAndHalo.create_data(r_low=r_low, r_up=r_up, nr=nr)

            # --- update vale of central data
            rho0 = self.rho0
            self.IsoAndNFW_data["central-rho"] = np.append(self.IsoAndNFW_data["central-rho"],
                                                           rho0)  # [M_sun/kpc^3]
            sigma0 = self.sigma0  # [kpc/Gyr]
            self.IsoAndNFW_data["central-velDis"] = np.append(self.IsoAndNFW_data["central-velDis"],
                                                              sigma0)  # [kpc/Gyr]

            # --- Here we stored data from `IsoAndHalo`
            self.IsoAndNFW_data["time"] = np.append(self.IsoAndNFW_data["time"],
                                                    nr*[_tage])  # time [Gyr]
            self.IsoAndNFW_data["r"] = np.append(self.IsoAndNFW_data["r"],
                                                 IsoAndNFW_dataToUpdate["r"])  # radius [kpc]
            self.IsoAndNFW_data["rho"] = np.append(self.IsoAndNFW_data["rho"],
                                                   IsoAndNFW_dataToUpdate["rho"])  # density [M_sun/kpc^3]
            self.IsoAndNFW_data["mass"] = np.append(self.IsoAndNFW_data["mass"],
                                                    IsoAndNFW_dataToUpdate["mass"])  # enclosed mass [M_sun]
            self.IsoAndNFW_data["velDis"] = np.append(self.IsoAndNFW_data["velDis"],
                                                      IsoAndNFW_dataToUpdate["velDis"])  # nu in [kpc/Gyr]
        else:
            # --- and take data from memory
            ISO_HiDen = self.prepare_ISO_data(rToISO, self.rho0_HiDen, self.sigma0_HiDen)
            ISO_data = dict()
            ISO_data["r"] = ISO_HiDen[4]  # radius [kpc]
            ISO_data["rho"] = ISO_HiDen[2]  # density [M_sun/kpc^3]
            ISO_data["mass"] = ISO_HiDen[5]  # enclosed mass [M_sun]

            # --- Create `Iso and halo class`: to find out how looks profile, where we have to stitch two profiles.
            ClassIsoAndHalo = IsoAndHalo(ISO_data, self.halo, self.r1)
            IsoAndNFW_dataToUpdate = ClassIsoAndHalo.create_data(r_low=r_low, r_up=r_up, nr=nr)

            # --- update vale of central data
            rho0 = self.rho0_HiDen
            self.IsoAndNFW_data["central-rho"] = np.append(self.IsoAndNFW_data["central-rho"],
                                                           rho0)  # [M_sun/kpc^3]
            sigma0 = self.sigma0_HiDen  # [kpc/Gyr]
            self.IsoAndNFW_data["central-velDis"] = np.append(self.IsoAndNFW_data["central-velDis"],
                                                              sigma0)  # [kpc/Gyr]

            # --- Here we stored data from `IsoAndHalo`
            self.IsoAndNFW_data["time"] = np.append(self.IsoAndNFW_data["time"],
                                                    nr*[_tage])  # time [Gyr]
            self.IsoAndNFW_data["r"] = np.append(self.IsoAndNFW_data["r"],
                                                 IsoAndNFW_dataToUpdate["r"])  # radius [kpc]
            self.IsoAndNFW_data["rho"] = np.append(self.IsoAndNFW_data["rho"],
                                                   IsoAndNFW_dataToUpdate["rho"])  # density [M_sun/kpc^3]
            self.IsoAndNFW_data["mass"] = np.append(self.IsoAndNFW_data["mass"],
                                                    IsoAndNFW_dataToUpdate["mass"])  # enclosed mass [M_sun]
            self.IsoAndNFW_data["velDis"] = np.append(self.IsoAndNFW_data["velDis"],
                                                      IsoAndNFW_dataToUpdate["velDis"])  # nu in [kpc/Gyr]

    def update_at_merging(self, _tage, _sigma_m, nr=400, r_low=-2.0, r_up=2.0, lowDens=True):
        """
        Before merging `low dense` and `high dense` can be distinguishable. Save some important data
        during evolution steps.

        Where,

            nr: number of radius's, which we use in the minimization procedure.
                (int)
            r_low: 10^r_low is the smallest radi, which you want to consider / save
                in txt file. Typically: `r_low=-2.0`. Take care about units: [10^r_low] = [r_s]
                (float)
            r_up: 10^r_low is the biggest radi, which you want to consider / save
                in txt file. Typically: `r_up=2.0`. Take care about units: [10^r_low] = [r_s]
                (float)
            lowDens: To distinguishable two possible results, `LoDense` and `HiDense`.

        """
        # radi logspace: to get sufficient accuracy - comes partially from init
        rToISO = np.logspace(np.log10(self.r_s * 10 ** self.r_low),
                             np.log10(self.r1),
                             self.nr)  # [kpc]

        # --- parameters at merging
        self.params_at_tmerge = dict()
        self.params_at_tmerge['r1'] = self.r1  # [kpc]
        self.params_at_tmerge['t'] = _tage  # [Gyr]
        self.params_at_tmerge['Mvir'] = self.halo.M_vir  # [M_sun]
        self.params_at_tmerge['c_const'] = self.halo.con  # [dimensionless]
        self.params_at_tmerge['cross_section'] = _sigma_m  # [cm^2/g]
        self.params_at_tmerge['t_tilda'] = uni.time_tilda(_tage, self.halo.rho_s,
                                                          self.halo.r_s, _sigma_m)  # [dimensionless]
        self.params_at_tmerge['rho0_LoDens'] = self.rho0  # [M_sun/kpc^3]
        self.params_at_tmerge['rho0_LoDens_tilda'] = uni.rho_tilda(self.rho0,
                                                                   self.halo.rho_s)  # [dimensionless]
        self.params_at_tmerge['sigma0_LoDens'] = self.sigma0  # [kpc/Gyr]
        sigma0_LoDens_tilda = uni.nu_tilda(self.sigma0, self.halo.r_s, self.halo.rho_s)
        self.params_at_tmerge['sigma0_LoDens_tilda'] = sigma0_LoDens_tilda  # [dimensionless]
        self.params_at_tmerge['rho0_HiDens'] = self.rho0_HiDen  # [M_sun/kpc^3]
        self.params_at_tmerge['rho0_HiDens_tilda'] = uni.rho_tilda(self.rho0_HiDen,
                                                                   self.halo.rho_s)  # [dimensionless]
        self.params_at_tmerge['sigma0_HiDens'] = self.sigma0_HiDen  # [kpc/Gyr]
        sigma0_HiDens_tilda = uni.nu_tilda(self.sigma0_HiDen, self.halo.r_s, self.halo.rho_s)
        self.params_at_tmerge['sigma0_HiDens_tilda'] = sigma0_HiDens_tilda  # [dimensionless]

        if lowDens is True:
            # --- and take data from memory
            ISO_LoDen = self.prepare_ISO_data(rToISO, self.rho0, self.sigma0)
            ISO_data = dict()
            ISO_data["r"] = ISO_LoDen[4]  # radius [kpc]
            ISO_data["rho"] = ISO_LoDen[2]  # density [M_sun/kpc^3]
            ISO_data["mass"] = ISO_LoDen[5]  # enclosed mass [M_sun]

            # --- Create `Iso and halo class`: to find out how looks profile, where we have to stitch two profiles.
            ClassIsoAndHalo = IsoAndHalo(ISO_data, self.halo, self.r1)
            IsoAndNFW_dataToUpdate = ClassIsoAndHalo.create_data(r_low=r_low, r_up=r_up, nr=nr)

            # --- update vale of central data
            rho0 = self.rho0
            self.IsoAndNFW_data["central-rho"] = np.append(self.IsoAndNFW_data["central-rho"],
                                                           rho0)  # [M_sun/kpc^3]
            sigma0 = self.sigma0  # [kpc/Gyr]
            self.IsoAndNFW_data["central-velDis"] = np.append(self.IsoAndNFW_data["central-velDis"],
                                                              sigma0)  # [kpc/Gyr]

            # --- Here we stored data from `IsoAndHalo`
            self.IsoAndNFW_data["time"] = np.append(self.IsoAndNFW_data["time"],
                                                    nr*[_tage])  # time [Gyr]
            self.IsoAndNFW_data["r"] = np.append(self.IsoAndNFW_data["r"],
                                                 IsoAndNFW_dataToUpdate["r"])  # radius [kpc]
            self.IsoAndNFW_data["rho"] = np.append(self.IsoAndNFW_data["rho"],
                                                   IsoAndNFW_dataToUpdate["rho"])  # density [M_sun/kpc^3]
            self.IsoAndNFW_data["mass"] = np.append(self.IsoAndNFW_data["mass"],
                                                    IsoAndNFW_dataToUpdate["mass"])  # enclosed mass [M_sun]
            self.IsoAndNFW_data["velDis"] = np.append(self.IsoAndNFW_data["velDis"],
                                                      IsoAndNFW_dataToUpdate["velDis"])  # nu in [kpc/Gyr]
        else:
            # --- and take data from memory
            ISO_HiDen = self.prepare_ISO_data(rToISO, self.rho0_HiDen, self.sigma0_HiDen)
            ISO_data = dict()
            ISO_data["r"] = ISO_HiDen[4]  # radius [kpc]
            ISO_data["rho"] = ISO_HiDen[2]  # density [M_sun/kpc^3]
            ISO_data["mass"] = ISO_HiDen[5]  # enclosed mass [M_sun]

            # --- Create `Iso and halo class`: to find out how looks profile, where we have to stitch two profiles.
            ClassIsoAndHalo = IsoAndHalo(ISO_data, self.halo, self.r1)
            IsoAndNFW_dataToUpdate = ClassIsoAndHalo.create_data(r_low=r_low, r_up=r_up, nr=nr)

            # --- update vale of central data
            rho0 = self.rho0_HiDen
            self.IsoAndNFW_data["central-rho"] = np.append(self.IsoAndNFW_data["central-rho"],
                                                           rho0)  # [M_sun/kpc^3]
            sigma0 = self.sigma0_HiDen  # [kpc/Gyr]
            self.IsoAndNFW_data["central-velDis"] = np.append(self.IsoAndNFW_data["central-velDis"],
                                                              sigma0)  # [kpc/Gyr]

            # --- Here we stored data from `IsoAndHalo`
            self.IsoAndNFW_data["time"] = np.append(self.IsoAndNFW_data["time"],
                                                    nr*[_tage])  # time [Gyr]
            self.IsoAndNFW_data["r"] = np.append(self.IsoAndNFW_data["r"],
                                                 IsoAndNFW_dataToUpdate["r"])  # radius [kpc]
            self.IsoAndNFW_data["rho"] = np.append(self.IsoAndNFW_data["rho"],
                                                   IsoAndNFW_dataToUpdate["rho"])  # density [M_sun/kpc^3]
            self.IsoAndNFW_data["mass"] = np.append(self.IsoAndNFW_data["mass"],
                                                    IsoAndNFW_dataToUpdate["mass"])  # enclosed mass [M_sun]
            self.IsoAndNFW_data["velDis"] = np.append(self.IsoAndNFW_data["velDis"],
                                                      IsoAndNFW_dataToUpdate["velDis"])  # nu in [kpc/Gyr]

    # -------------------- RETURN VALUES -------------------- #
    def return_IsoAndNFW_data(self):
        """return prepared data"""
        return self.IsoAndNFW_data

    def return_params_at_tmerge(self):
        """return parameters at merging"""
        return self.params_at_tmerge
