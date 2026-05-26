#########################################################################
#
# global variables
#
# import config as cfg in all related modules, and use a global variable
# x defined here in the other modules as cfg.x

#########################################################################
import os
import numpy as np
import math
import scipy.constants

# ############################################ LOCALIZATION OF IMPORTANT DIRECTORIES ################################# #
# Get the base directory of the project (two levels up from this file's directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the data directories relative to the base directory
DATA_GRAVOTHERMAL_DIR = os.path.join(BASE_DIR, 'data', 'Gravothermal')
DATA_ISOTHERMAL_NFW = os.path.join(BASE_DIR, 'data', 'IsothermalAndNFW')

# DATA_ISOTHERMAL_NFW_LoDen_DIR = os.path.join(BASE_DIR, 'data', 'IsothermalAndNFW', 'LowDenSolution')
# DATA_ISOTHERMAL_NFW_HiDen_DIR = os.path.join(BASE_DIR, 'data', 'IsothermalAndNFW', 'HighDenSolution')
# print("base dir:", BASE_DIR)
# print("DATA_GRAVOTHERMAL_DIR:", DATA_GRAVOTHERMAL_DIR)
# print("DATA_ISOTHERMAL_AND_NFW_DIR:", DATA_ISOTHERMAL_AND_NFW_DIR)

# ############################################ COSMOLOGICAL SETTINGS ################################################# #
# ------------- PHYSICAL CONSTANTS ------------- #
# --- setting constants value
# setting, which will be corresponding to gravothermal simulation
const_h = 0.671  # [dimensionless]
rho_c = 1.8791 * 1.4771 * 10**2 * const_h**2  # critical density [M_solar / kpc^3]
# setting in paper 2206
const_h_old = 0.71025  # [dimensionless]
rho_c_old = 2.7754 * 10**2 * const_h_old**2  # critical density [M_solar / kpc^3]


# const_h = 0.71025  # [dimensionless]
# rho_c = 2.7754 * 10**2 * const_h_old**2  # critical density [M_solar / kpc^3]


year = 365 * 24 * 60 * 60  # [s]
Gyr = 10**9 * year  # [s]
kpc_SI = 3.08567758 * 10**19  # [m]
M_solar_SI = 1.98847 * 10**30  # [kg]
const_G = scipy.constants.G  # [m^3 * kg^-1 * s^-2]
const_G_starUnits = const_G * kpc_SI**(-3) * M_solar_SI**1 * Gyr**2  # gravitational constant [kpc^3 Gyr^-2 M_sun^-1]
# print(gravitational constant [kpc^3 Gyr^-2 M_sun^-1] = {const_G_starUnits}')

# ---------------- Unit conversions ----------------
# velocity: [kpc/Gyr] -> [km/s]
kpcGyr_to_kms = (kpc_SI / Gyr) / 1.0e3  # = 0.977792... km/s

# and the inverse if you ever need it:
kms_to_kpcGyr = 1.0 / kpcGyr_to_kms

# ------------- FOR SATELLITE EVOLUTION ------------- #
Mres = 1e4  # [M_sun] mass resolution
Rres = 0.01  # [kpc] spatial resolution, original: 0.01, 0.0033

def find_Rres(halo, cff=0.001):
    """
    Set the value of the spatial resolution. In my opinion the most accurate way of set
    the spatial resolution is to correspond that one to the `r_s` coming the halo profile
    for the CDM-like outskirt.

    halo: the halo profile for the CDM-like outskirt
        (object).
    cff: value of coefficient, which we multiply by `r_s`. Default value is: `0.001`.
        (float)
    """
    r_s = halo.r_s  # [kpc]
    Rres = cff * r_s  # [kpc], do not care about 1/6 - just to get better results!
    return Rres

# ------------- CONSTANTS ------------- #
ln10 = np.log(10.)
Root2 = np.sqrt(2.)
RootPi = np.sqrt(np.pi)
Root2OverPi = np.sqrt(2./np.pi)
Root1Over2Pi = np.sqrt(0.5/np.pi)
TwoOverRootPi = 2./np.sqrt(np.pi)
FourOverRootPi = 4./np.sqrt(np.pi)
FourPiOverThree = 4.*np.pi/3.
TwoPi = 2.*np.pi
TwoPiG = 2.*np.pi*const_G_starUnits
TwoPisqr = 2.*np.pi**2
ThreePi = 3.*np.pi
FourPi = 4.*np.pi
FourPiG = 4.*np.pi*const_G_starUnits
FourPiGsqr = 4.*np.pi * const_G_starUnits**2.  # useful for dynamical friction
ThreePiOverSixteenG = 3.*np.pi / (16.*const_G_starUnits)  # useful for dynamical time

eps = 0.001  # an infinitesimal for various purposes: e.g., if the
# fractional difference of a quantity between two consecutive steps
# is smaller than cfg.eps, jump out of a loop; and e.g., for
# computing derivatives

# ------------- CONSTANTS FROM PAPER: 2206.12425 ------------- #
M_vir = 10**11  # [M_solar]
const_c = 15  # [dimensionless]
time_evolution = 5 * 10**9 * year  # [s]
sigma_m = 1  # [cm^2 / g]
sigma_m_SI = sigma_m * 10**(-4) * 10**3  # [m^2/kg]

# --------- HELPFUL FUNCTIONS --------- #
def rounded_number(_number, _significant_digits):
    """
    Taking the value to rounded and takes the amount of significant digits you specify.
    """
    return round(_number, _significant_digits - int(math.floor(math.log10(abs(_number)))) - 1)


# --------- COSMOLOGY FUNCTIONS --------- #
# WMAP7
h = 0.71
Om = 0.266
Ob = 0.0465
OL = 0.734
s8 = 0.801
ns = 0.963

def rhoc(z, h=0.7, Om=0.3, OL=0.7):
    """
    Critical density [M_sun kpc^-3] at redshift z.

    Syntax:

        rhoc(z,h=0.7,Om=0.3,OL=0.7)

    where

        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1
                = h/10 km s^-1 kpc^-1
                = h/9.778 Gyr^-1
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3)
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7)
    """
    return rho_c * h ** 2 * (Om * (1. + z) ** 3 + OL)

# think about it !
# print("my rho_c:", rho_c)
# print("function rho_c:", rhoc(0, h, Om, OL))
