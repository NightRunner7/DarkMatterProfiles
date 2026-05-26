"""
Units Conversion Functions

This module contains functions for converting between "our" units (discussed in `units.py`) and "their" units (used
in publication 2305.05067).

Functions:
- time_units_paper: Converts time from our units to the units used in the discussed publication.
- vrms_units: Scales the velocity dispersion to publication units when provided in [r_s/Gyr].
- vrms_units_NFW: Scales the velocity dispersion to publication units when provided in [kpc/Gyr].

These functions are designed to provide a convenient way to handle unit conversions in various parts of the codebase.

Author: Krzysztof Szafrański
"""
import numpy as np
# --- IMPORT FROM FILES
import config as cfg

# ################################################# PAPER UNITS ###################################################### #
def time_units_paper(_time, _rho_s, _r_s, _sigma_m):
    """
    Convert time from our units to the units used in the discussed publication.

    Parameters:
        _time (float or ndarray): Evolution time of the galaxy [Gyr].
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].
        _r_s (float): Scale radius of NFW [kpc].
        _sigma_m (float): Dark Matter cross-section [cm^2/g].

    Returns:
        float or ndarray: Time in publication units.
    """
    # Convert dark matter cross-section from [cm^2/g] to [m^2/kg]
    sigma_m_SI = _sigma_m * 10 ** (-4) * 10 ** 3  # [m^2/kg]

    # Convert dark matter cross-section to star units [kpc^2/M_sun]
    sigma_m_SU = sigma_m_SI * cfg.kpc_SI ** (-2) * cfg.M_solar_SI  # [kpc^2/M_sun]

    # Calculate time in publication units
    time_pub_units = sigma_m_SU * _time * _rho_s * _r_s * np.sqrt(4 * np.pi * cfg.const_G_starUnits * _rho_s)

    return time_pub_units

def vrms_units(vel_dis, _rho_s, _r_s):
    """
    Scale the velocity dispersion to publication units.

    Parameters:
        vel_dis (float, list, or ndarray): Velocity dispersion of Dark Matter [r_s/Gyr].
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].
        _r_s (float): Scale radius of NFW [kpc].

    Returns:
        float or ndarray: Scaled velocity dispersion in publication units.
    """
    def scale_velocity(vel):
        """
        Helper function to scale a single velocity dispersion value.

        Parameters:
            vel (float): Velocity dispersion of Dark Matter [r_s/Gyr].

        Returns:
            float: Scaled velocity dispersion [dimensionless].
        """
        vel_dis_SU = vel * _r_s  # Convert velocity dispersion to [kpc/Gyr]
        vel_dis_hat = vel_dis_SU / (_r_s * np.sqrt(4 * np.pi * cfg.const_G_starUnits * _rho_s))  # Scale to publication units
        return vel_dis_hat

    if isinstance(vel_dis, (list, np.ndarray)):
        # Convert list or ndarray of velocity dispersions
        vel_dis_arr = np.array([scale_velocity(vel) for vel in vel_dis])
        return vel_dis_arr
    else:
        # Convert single velocity dispersion value
        return scale_velocity(vel_dis)

def vrms_units_NFW(vel_dis, _rho_s, _r_s):
    """
    Scale the velocity dispersion to publication units.

    Parameters:
        vel_dis (float, list, or ndarray): Velocity dispersion of Dark Matter [kpc/Gyr].
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].
        _r_s (float): Scale radius of NFW [kpc].

    Returns:
        float or ndarray: Scaled velocity dispersion in publication units.
    """
    def scale_velocity(vel):
        """
        Helper function to scale a single velocity dispersion value.

        Parameters:
            vel (float): Velocity dispersion of Dark Matter [kpc/Gyr].

        Returns:
            float: Scaled velocity dispersion [dimensionless].
        """
        return vel / (_r_s * np.sqrt(4 * np.pi * cfg.const_G_starUnits * _rho_s))

    if isinstance(vel_dis, (list, np.ndarray)):
        # Convert list or ndarray of velocity dispersions
        vel_dis_arr = np.array([scale_velocity(vel) for vel in vel_dis])
        return vel_dis_arr
    else:
        # Convert single velocity dispersion value
        return scale_velocity(vel_dis)
