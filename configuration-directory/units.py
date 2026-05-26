"""
Units Conversion Functions

This module contains functions for converting between different units commonly used in the code.

Functions:
- convert_time_txt_to_full: Convert time from log10 units back to the original units.
- convert_nu_txt_to_full: Convert velocity dispersion from transformed units back to the original units.
- [Add descriptions for other functions if needed]

These functions are designed to provide a convenient way to handle unit conversions in various parts of the codebase.

Author: Krzysztof Szafrański
"""
import numpy as np
# --- IMPORT FROM FILES
import config as cfg

# ################################################# UNITS IN ISOTHERMAL ############################################## #
# ------------------------------------- DIMENSIONLESS UNITS ------------------------------------- #
"""
If we set evolution time of galaxy (tage) we can get the rho0.

    rho0: best value of rho, which minimize the variable delta [M_sun/kpc^3] (float).
    tage: selected evolution time of galaxy [Gyr] (float).

Now we're gonna to introduce function, which we will use to calculate
rho_tilda and time_tilda (see fig.6 in paper 2206). We can write explicit form as:

    rho_tilda = rho/rho_s [dimensionless] = [rho_s]
    r_tilda = r/r_s [dimensionless] = [r_s]
    Mass_tilda =Mass/(rho_s * r_s^3) [dimensionless] = [rho_s * r_s^3]
    time_tilda = tage/t0 [dimensionless]
    nu_tilda = nu/nu0 [dimensionless]

where:
    t0 = a * sigma_m_SU * nu0 * rho_s [Gyr]
    nu0 = sqrt(G_SU * M0 / r_s) [kpc * Gyr^(-1)]
    M0 = 4 * pi * r_s^3 * rho_s [M_sun]
    
    G_SU: Gravitational constant in 'star units' [kpc^3 * Gyr^(-2) * M_sun^(-1)] (float).
    sigma_m_SU: Cross-section of DM in 'star units' [kpc^2/M_sun] (float).
    rho_s: Parameter of NFW (Dark matter halo) [M_sun/kpc^3].
    r_s: parameter of NFW (Dark matter halo) [kpc].
"""
def cal_t0(_rho_s, _r_s, _sigma_m):
    """
    Calculate the 'scaling time' variable t0.

    Parameters:
        _rho_s (float): Density parameter [M_sun/kpc^3].
        _r_s (float): Scale radius [kpc].
        _sigma_m (float): DM cross-section [cm^2/g].

    Returns:
        float: Scaling time t0 [Gyr].

    Steps:
        1. Convert DM cross-section from [cm^2/g] to [kpc^2 * M_sun^-1].
        2. Calculate the characteristic mass M0 [M_sun].
        3. Calculate the gravitational constant in star units G_SU [kpc^3 * Gyr^-2 * M_sun^-1].
        4. Compute the characteristic velocity nu0 [kpc * Gyr^-1].
        5. Calculate and return t0 [Gyr].
    """
    # Step 1: Convert DM cross-section to [kpc^2 * M_sun^-1]
    sigma_m_SI = _sigma_m * 1e-4 * 1e3  # Convert from [cm^2/g] to [m^2/kg]
    sigma_m_SU = sigma_m_SI * cfg.kpc_SI ** -2 * cfg.M_solar_SI  # Convert to [kpc^2 * M_sun^-1]

    # Step 2: Calculate the characteristic mass M0 [M_sun]
    M0 = cfg.FourPi * _r_s ** 3 * _rho_s  # [M_sun]

    # Step 3: Gravitational constant in star units [kpc^3 * Gyr^-2 * M_sun^-1]
    G_SU = cfg.const_G_starUnits
    # print("G_SU:", G_SU)

    # Step 4: Calculate the characteristic velocity nu0 [kpc * Gyr^-1]
    nu0 = np.sqrt(G_SU * M0 / _r_s)  # [kpc * Gyr^-1]

    # print("sigma_m*rho_s*r_s", sigma_m_SU*_rho_s*_r_s)

    # Step 5: Calculate and return the scaling time t0 [Gyr]
    t0 = (cfg.FourOverRootPi * sigma_m_SU * nu0 * _rho_s) ** -1  # [Gyr]

    return t0

def time_tilda(_tage, _rho_s, _r_s, _sigma_m):
    """
    Calculate the dimensionless `\tilde{time}`.

    Parameters:
        _tage (float or ndarray): Evolution time of the galaxy [Gyr].
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].
        _r_s (float): Scale radius of NFW [kpc].
        _sigma_m (float): Dark Matter cross-section [cm^2/g].

    Returns:
        float or ndarray: Dimensionless time `\tilde{time}`.

    Steps:
        1. Calculate the scaling time t0 [Gyr] using the cal_t0 function.
        2. Calculate and return the dimensionless time by dividing _tage by t0.
    """
    # Step 1: Calculate the scaling time t0 [Gyr]
    t0 = cal_t0(_rho_s, _r_s, _sigma_m)  # [Gyr]

    # Step 2: Calculate and return the dimensionless time
    return _tage / t0

def rho_tilda(_rho, _rho_s):
    """
    Calculate the dimensionless `\tilde{rho}`.

    Parameters:
        _rho (float or ndarray): Density [M_sun/kpc^3].
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].

    Returns:
        float or ndarray: Dimensionless rho `\tilde{rho} = [rho_s]`.

    Steps:
        1. Divide _rho0 by _rho_s to obtain the dimensionless rho.
    """
    return _rho / _rho_s

def r_tilda(_r, _r_s):
    """
    Calculate the dimensionless `\tilde{r}`.

    Parameters:
        _r (float or ndarray): Radius [kpc].
        _r_s (float): Scale radius of NFW [kpc].

    Returns:
        float or ndarray: Dimensionless radius `\tilde{r} = [r_s]`.

    Steps:
        1. Divide _r by _r_s to obtain the dimensionless radius.
    """
    return _r / _r_s

def nu_tilda(_nu, _r_s, _rho_s):
    """
    Calculate the dimensionless `\tilde{nu}`, representing the velocity dispersion.

    Parameters:
        _nu (float or ndarray): Velocity dispersion [kpc/Gyr].
        _r_s (float): Scale radius of NFW [kpc].
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].

    Returns:
        float or ndarray: Dimensionless velocity dispersion `\tilde{nu}`.

    Steps:
        1. Calculate the gravitational constant G in star units.
        2. Compute the characteristic mass M0 [M_sun].
        3. Calculate the characteristic velocity dispersion nu0 [kpc/Gyr].
        4. Divide _nu by nu0 to obtain the dimensionless velocity dispersion.
    """
    # Step 1: Gravitational constant in star units [kpc^3 Gyr^-2 M_sun^-1]
    G = cfg.const_G_starUnits

    # Step 2: Calculate the characteristic mass M0 [M_sun]
    M0 = 4 * np.pi * _r_s**3 * _rho_s  # [M_sun]

    # Step 3: Calculate the characteristic velocity dispersion nu0 [kpc/Gyr]
    nu0 = np.sqrt(G * M0 / _r_s)  # [kpc/Gyr]

    # Step 4: Calculate and return the dimensionless velocity dispersion
    return _nu / nu0

def mass_tilde(_mass, _rho_s, _r_s):
    """
    Convert enclosed mass to dimensionless units suitable for storage or calculations.

    Parameters:
        _mass (float or ndarray): Enclosed mass [M_sun].
        _rho_s (float): Parameter of NFW [M_sun/kpc^3].
        _r_s (float): Parameter of NFW [kpc].

    Returns:
        float or ndarray: Dimensionless mass `\tilde{mass} = [rho_s * r_s^3]`.

    Steps:
        1. Divide _mass by (_r_s^3 * _rho_s) to obtain the dimensionless mass.
    """
    # Step 1: Divide _mass by (_r_s^3 * _rho_s) to obtain the dimensionless mass
    return _mass / (_r_s ** 3 * _rho_s)

# ------------------------------------- DIMENSION FULL UNITS ------------------------------------- #
"""
Here we write necessary function which converts dimensionless variables and
convert them to the dimensional (full) variable.
"""
def convert_time_tilda(_time_tilda, _rho_s, _r_s, _sigma_m):
    """
    Convert dimensionless time to dimension full time.

    Parameters:
        _time_tilda (float or ndarray): Dimensionless evolution time of the galaxy.
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].
        _r_s (float): Scale radius of NFW [kpc].
        _sigma_m (float): Dark Matter cross-section [cm^2/g].

    Returns:
        float or ndarray: Dimension full time [Gyr].

    Steps:
        1. Calculate the scaling time t0 [Gyr] using the cal_t0 function.
        2. Multiply _time_tilda by t0 to obtain the dimension full time.
    """
    # Step 1: Calculate the scaling time t0 [Gyr]
    t0 = cal_t0(_rho_s, _r_s, _sigma_m)  # [Gyr]
    # print("t0", t0)
    # Step 2: Multiply _time_tilda by t0 to obtain the dimension full time [Gyr]
    return _time_tilda * t0

def convert_rho_tilda(_rho, _rho_s):
    """
    Convert dimensionless density to dimension full density.

    Parameters:
        _rho (float or ndarray): Dimensionless density [M_sun/kpc^3].
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].

    Returns:
        float or ndarray: Dimension full density [M_sun/kpc^3].

    Steps:
        1. Multiply _rho by _rho_s to obtain the dimension full density.
    """
    # Step 1: Multiply _rho by _rho_s to obtain the dimension full density [M_sun/kpc^3]
    return _rho * _rho_s

def convert_r_tilda(_r_tilda, _r_s):
    """
    Convert dimensionless radius to dimension full radius.

    Parameters:
        _r_tilda (float or ndarray): Dimensionless radius.
        _r_s (float): Parameter of NFW [kpc].

    Returns:
        float or ndarray: Dimension full radius [kpc].

    Steps:
        1. Multiply _r_tilda by _r_s to obtain the dimension full radius.
    """
    # Step 1: Multiply _r_tilda by _r_s to obtain the dimension full radius [kpc]
    return _r_tilda * _r_s

def convert_nu_tilda(_nu_tilda, _r_s, _rho_s):
    """
    Convert dimensionless velocity dispersion to dimension full velocity dispersion.

    Parameters:
        _nu_tilda (float or ndarray): Dimensionless velocity dispersion `\tilde{nu}`.
        _r_s (float): Scale radius of NFW [kpc].
        _rho_s (float): Density parameter of NFW [M_sun/kpc^3].

    Returns:
        float or ndarray: Dimension full velocity dispersion [kpc/Gyr].

    Steps:
        1. Calculate the gravitational constant G in star units.
        2. Compute the characteristic mass M0 [M_sun].
        3. Calculate the characteristic velocity dispersion nu0 [kpc/Gyr].
        4. Multiply _nu_tilda by nu0 to obtain the dimension full velocity dispersion.
    """
    # Step 1: Gravitational constant in star units [kpc^3 Gyr^-2 M_sun^-1]
    G = cfg.const_G_starUnits

    # Step 2: Calculate the characteristic mass M0 [M_sun]
    M0 = 4 * np.pi * _r_s**3 * _rho_s  # [M_sun]

    # Step 3: Calculate the characteristic velocity dispersion nu0 [kpc/Gyr]
    nu0 = np.sqrt(G * M0 / _r_s)  # [kpc/Gyr]

    # Step 4: Multiply _nu_tilda by nu0 to obtain the dimension full velocity dispersion
    return _nu_tilda * nu0

def convert_mass_tilde(_mass_tilde, _rho_s, _r_s):
    """
    Convert dimensionless mass to dimension full mass.

    Parameters:
        _mass_tilde (float or ndarray): Dimensionless mass `\tilde{mass}`.
        _rho_s (float): Parameter of NFW [M_sun/kpc^3].
        _r_s (float): Parameter of NFW [kpc].

    Returns:
        float or ndarray: Dimension full mass [M_sun].

    Steps:
        1. Multiply mass_tilde by (r_s^3 * rho_s) to obtain the dimension full mass.
    """
    return _mass_tilde * (_r_s ** 3 * _rho_s)

# ################################################# STORED DATA IN .csv FILE ######################################### #
# ------------------------------------- IN .csv FILE ------------------------------------- #
"""
This comment describes the units of data to be stored in a .csv file and the approach 
for handling different quantities.

For NFW Profile:
    [Density] = [rho_s] [M_sun/kpc^3]
    [Radius] = [r_s] [kpc]
    [Mass] = [rho_s * r_s^3] [M_sun]

Different from the Isothermal Profile:
    [Time] = log10(Gyr)
    [Velocity Dispersion] = [nu] = [r_s / Gyr] [kpc/Gyr]
"""

def time_txt(_time):
    """
    Convert time (evolution of galaxy) into the same units as used in
    the gravothermal simulation by taking the base-10 logarithm.

    Parameters:
        _time (float or ndarray): Evolution time of the galaxy [Gyr].

    Returns:
        float or ndarray: Transformed time in log10 units.

    Notes:
        The logarithm transformation is applied to maintain consistency
        with the units used in the gravothermal simulation.
    """
    return np.log10(_time)


def nu_txt(_nu, _r_s):
    """
    Convert velocity dispersion into the same units as used in
    the gravothermal simulation.

    Parameters:
        _nu (float or ndarray): Velocity dispersion [kpc/Gyr].
        _r_s (float): Parameter of the NFW profile [kpc].

    Returns:
        float or ndarray: Transformed velocity dispersion in units [r_s/Gyr].

    Notes:
        The velocity dispersion is divided by the scale radius (_r_s) to maintain
        consistency with the units used in the gravothermal simulation.
    """
    return _nu / _r_s

# ------------------------------------- CODE DIMENSION FULL UNITS ------------------------------------- #
def convert_time_txt(_time_txt):
    """
    Convert time from log10 units back to the 'code' dimension full units.

    Parameters:
        _time_txt (float or ndarray): Transformed time in log10 units.

    Returns:
        float or ndarray: Original time in [Gyr] units.

    Notes:
        The inverse operation of taking the base-10 logarithm is applied to
        revert the transformed time back to 'code' dimension full units.
    """
    return 10 ** _time_txt

def convert_nu_txt(_nu_txt, _r_s):
    """
    Convert velocity dispersion from the transformed units back to the original units used in the code.

    Parameters:
        _nu_txt (float): Transformed velocity dispersion in units [r_s/Gyr].
        _r_s (float): Parameter of the NFW profile [kpc].

    Returns:
        float: Original velocity dispersion in [kpc/Gyr] units.

    Notes:
        The inverse operation of dividing by the scale radius (_r_s) is applied to
        revert the transformed velocity dispersion back to the original units used in the code.
    """
    return _nu_txt * _r_s
