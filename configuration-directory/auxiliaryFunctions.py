import numpy as np
import pandas as pd
import math
import os
import shutil
import glob
from scipy.interpolate import interp1d
from scipy.integrate import simps
from PIL import Image
# --- IMPORT FROM FILES
import config as cfg

# ################################################# GENERAL FUNCTION ################################################# #
# --------- ROUNDING FUNCTION --------- #
def rounded_number(_number, _significant_digits):
    """
    Taking the value to rounded and takes the amount of significant digits you specify.
    """
    return round(_number, _significant_digits - int(math.floor(math.log10(abs(_number)))) - 1)

# ################################################# PLAY WITH CREATING FOLDERS ####################################### #
def make_directory(_path_name):
    """ Create directory with fixed name"""
    try:
        # Create target Directory
        os.mkdir(_path_name)
        print("Directory ", _path_name, " Created ")
    except FileExistsError:
        print("Directory ", _path_name, " already exists")

def clear_and_make_directory(_path_name):
    """Create directory with fixed name. And make sure that directory will be empty"""
    try:
        # Create target Directory
        os.mkdir(_path_name)
        print("Directory ", _path_name, " Created ")
    except FileExistsError:
        # directory exist
        print("Directory ", _path_name, " already exists")
        # delete all files in directory
        for filename in os.listdir(_path_name):
            file_path = os.path.join(_path_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def make_gif(frame_folder, name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    frame_one = frames[0]
    frame_one.save(f"{frame_folder}/{name}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=50, loop=0)

# ################################################# PLAY WITH LOCALIZATION OF FOLDERS ################################ #
def load_csv_from_dir(directory, file_name, sep=',', names=None, dtype=None, low_memory=True):
    """
    Function to load a CSV file from a given directory with optional parameters for separator and column names.

    :param directory: The directory where the CSV file is located.
    :param file_name: The name of the CSV file.
    :param sep: The delimiter to use (default is ',').
    :param names: List of column names to use (default is None, inferring from the first row).
    :param dtype: Data type(s) to apply to either the whole dataset or individual columns
    :param low_memory: (see) https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    :return: A pandas DataFrame if the file exists, otherwise None.
    """
    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        return pd.read_csv(file_path, sep=sep, names=names, dtype=dtype, low_memory=low_memory)
    else:
        print(f"The file {file_name} does not exist in the directory {directory}.")
        return None
# ################################################# PLAY WITH ARRAYS AND LISTS ####################################### #
def find_nearest_and_index(arr, x):
    """
    Put array or list and one value. This function will return the nearest value
    to the specified.

    :param arr: list of values
        (array)
    :param x: specified value
        (float)
    """
    # calculate the difference array
    difference_array = np.absolute(arr - x)

    index = difference_array.argmin()  # find the index of minimum element from the array
    nearest_val = arr[index]  # nearest element to the given values

    return [nearest_val, index]

# ################################################# INTERPOLATION #################################################### #
def lnrho_interp_simple(logold_r, logold_rho, lognew_r, extrapolate=True):
    """
    Function which does interpolation and (or) extrapolation.

    where:
        logold_r: base radius to do interpolation. Notice that we take: `ln(r)`.
            (array) [ln(kpc)]
        logold_rho: base densities to do interpolation. Notice that we take: `ln(\rho(r))`.
            (array) [ln(M_sun/kpc^3)]
        lognew_r: the value of radi for which we want to know the value of density.
            Notice that we take: `ln(new_r)`.
            (array or float) [ln(kpc)]

    return: log(\rho(new_r)): interpolated or extrapolated value of density at `new_r`.
        (array or float) [ln(M_sun/kpc^3)]
    """
    if extrapolate is True:
        density_interp_func = interp1d(logold_r, logold_rho, kind='linear', fill_value="extrapolate")
    else:
        density_interp_func = interp1d(logold_r, logold_rho, kind='linear', fill_value=0)  # if something out of prepared
                                                                                           # set zero
    density_interp = density_interp_func(lognew_r)
    return density_interp

def prepare_lnrho_interp_simple(old_r, old_rho, new_beg, new_end, num):
    """
    Function, which uses in interpolation and (or) extrapolation function:

        lnrho_interp_simple(logold_r, logold_rho, lognew_r, extrapolate=True) .

    To get interpolate array of radius's and corresponding to them densities: `\rho(r)`.
    One important remark: we use that interpolation in cases, when we have known
    that density will `be increasing` or `be decreasing` exponentially. Then we
    should use it.
    -------------
    where,
        old_r: array of radi, where successive r's are in exponential intervals (r = np.logspace())
            (array) [kpc]
        old_rho: array of densities, which are connected to array of radius's: `\rho(r)`.
            (array) [M_sun/kpc^3]
        new_beg: where extrapolation radi should start (usually we use: `new_beg = old_r[0]`).
            (float) [kpc]
        new_end: where extrapolation radi should end (usually we use: `new_end = old_r[-1]`)
            (float) [kpc]
        num: how many items should the new list contain (usually we use: `num = 10-50 * len(old_r)`)
            (float)
    -------------
    return: [rinterp_list,rhointerp]
        rinterp_list: array of radi after interpolation.
            (array) [kpc]
        rhointerp: array of densities after interpolation.
            (array) [M_sun/kpc^3]
    """
    # radius
    rinterp_list = np.logspace(np.log10(new_beg), np.log10(new_end), num, endpoint=True)  # [kpc]
    # density
    rhointerp = np.exp(lnrho_interp_simple(np.log(old_r), np.log(old_rho), np.log(rinterp_list)))  # [M_sun/kpc^3]
    return [rinterp_list, rhointerp]  # [M_sun/kpc^3]

# ################################################# ENCLOSED MASS #################################################### #
def Miso_simpson(r, rho):
    """
    Enclosed mass profile of the SIDM isothermal core using Simpson's rule.
    More accurate method of finding enclosed mass. Can be time-consuming calculation.
    The radii can be provided in any spaced bins.

    Parameters:
    - r: array-like, radii at which the density profile is evaluated [kpc]
    - rho: array-like, density profile [M_sun/kpc^3]

    Returns:
    - array-like, mass profile corresponding to the input density profile [M_sun]
    """
    # Ensure inputs are numpy arrays
    r = np.asarray(r)
    rho = np.asarray(rho)

    # Compute the enclosed mass using Simpson's rule
    M_enc = np.zeros_like(r)
    for i in range(1, len(r)):
        # Use Simpson's rule on [0, r[i]]
        M_enc[i] = 4 * np.pi * simps(rho[:i+1] * r[:i+1]**2, r[:i+1])

    return M_enc

def Miso_logspaceR(r, rho):
    """
    Calculate the enclosed mass profile of the SIDM isothermal core using the method of spherical shells.
    The radii should be provided in logarithmically spaced bins.

    Parameters:
    - r: array-like, radii at which the density profile is evaluated [kpc]
    - rho: array-like, density profile [M_sun/kpc^3]

    Returns:
    - array-like, mass profile corresponding to the input density profile [M_sun]
    """
    rhoave = np.append(rho[0], 0.5 * (rho[1:] + rho[:-1]))
    dM = np.append(cfg.FourPiOverThree * r[0] ** 3 * rhoave[0],
                   cfg.FourPiOverThree * (r[1:]**3 - r[:-1]**3) * rhoave[1:])
    return dM.cumsum()

def Miso_linspaceR(r, rho):
    """
    Enclosed mass profile of the SIDM isothermal core, given the radii and density profile array.
    The radii should be provided in linearly spaced bins.
    ----------------
    We can consider:
        r = np.logspace(np.log10(r_s * 10 ** -4.), np.log10(r1), nr)
    only when we take nr = 50000.
    ----------------
    Parameters:
    - r: array-like, radii at which the density profile is evaluated [kpc]
    - rho: array-like, density profile [M_sun/kpc^3]

    Return:
    - array-like, mass profile corresponding to the input density profile [M_sun]
    """
    rtmp = np.append(0., r[:-1])
    dr = r - rtmp
    rhoave = np.append(rho[0], 0.5 * (rho[1:] + rho[:-1]))
    dM = cfg.FourPi * r**2 * dr * rhoave
    return dM.cumsum()

# ################################################# RHO AND VelDIS CORE ############################################## #
# def central_density_first_cell(r, rho):
def central_density_first_cell(r, rho):
    """
    Estimate central density as the volume-averaged density
    in the innermost cell, assuming:
      - r[0] = 0 is the center
      - rho[0] is the central density value
      - rho[i] are profile values at radii r[i]

    We define the first cell edge as midway between r[1] and r[2].
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if len(r) < 3:
        raise ValueError("Need at least 3 radial points.")
    if r[0] != 0.0:
        raise ValueError("Expected r[0] = 0 for this definition.")

    # outer edge of first cell
    r_edge = 0.5 * (r[1] + r[2])

    # piecewise-linear interpolation in rho(r)
    r_sub = r[:3]
    rho_sub = rho[:3]

    rr = np.linspace(0.0, r_edge, 200)
    rho_interp = np.interp(rr, r_sub, rho_sub)

    integral = np.trapz(rho_interp * rr**2, rr)
    rho_c = 3.0 * integral / (r_edge**3)

    return rho_c

# def mean_density_within_radius(r, rho, R_core):
def mean_density_within_radius(r, rho, _elements):
    """
    Volume-averaged density within fixed radius R_core:
        rho_bar = 3/R_core^3 * ∫ rho(r) r^2 dr

    Works for profiles that include r=0.
    Uses linear interpolation in rho(r).
    """
    R_core=1e-2
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if r[0] != 0.0:
        raise ValueError("Expected r[0] = 0.")
    if R_core <= 0:
        raise ValueError("R_core must be positive.")
    if R_core > r[-1]:
        raise ValueError("R_core exceeds available radial range.")

    # keep points up to R_core
    mask = r <= R_core
    r_use = r[mask]
    rho_use = rho[mask]

    # ensure exact endpoint at R_core
    if r_use[-1] < R_core:
        rho_R = np.interp(R_core, r, rho)
        r_use = np.append(r_use, R_core)
        rho_use = np.append(rho_use, rho_R)

    rr = np.linspace(0.0, R_core, 400)
    rho_interp = np.interp(rr, r_use, rho_use)

    integral = np.trapz(rho_interp * rr**2, rr)
    return 3.0 * integral / (R_core**3)

#calculate_rho_core_test
def calculate_rho_core(r, rho, R_core=1e-2, skip_first_nonzero=1, n_int=400):
    """
    Volume-averaged density within fixed radius R_core:
        rho_bar = 3/R_core^3 * ∫ rho(r) r^2 dr

    Parameters
    ----------
    r, rho : array-like
        Radial grid and density profile. Assumes r[0] = 0.
    R_core : float
        Core averaging radius.
    skip_first_nonzero : int
        Number of first nonzero radial bins to remove from the interpolation.
        0 = use all points
        1 = drop the first nonzero bin
        2 = drop the first two nonzero bins
    n_int : int
        Number of integration points for the final uniform grid.
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if r[0] != 0.0:
        raise ValueError("Expected r[0] = 0.")
    if R_core <= 0:
        raise ValueError("R_core must be positive.")
    if R_core > r[-1]:
        raise ValueError("R_core exceeds available radial range.")

    # points up to R_core
    mask = r <= R_core
    r_use = r[mask].copy()
    rho_use = rho[mask].copy()

    if len(r_use) < 2:
        raise ValueError("Not enough points inside R_core.")

    # identify nonzero bins
    nonzero_idx = np.where(r_use > 0.0)[0]
    if skip_first_nonzero > len(nonzero_idx):
        raise ValueError("skip_first_nonzero is too large for available grid.")

    # keep r=0 and remove chosen early nonzero bins
    if skip_first_nonzero > 0:
        remove_idx = nonzero_idx[:skip_first_nonzero]
        keep_mask = np.ones(len(r_use), dtype=bool)
        keep_mask[remove_idx] = False
        r_use = r_use[keep_mask]
        rho_use = rho_use[keep_mask]

    # ensure exact endpoint at R_core
    if r_use[-1] < R_core:
        rho_R = np.interp(R_core, r, rho)
        r_use = np.append(r_use, R_core)
        rho_use = np.append(rho_use, rho_R)

    rr = np.linspace(0.0, R_core, n_int)
    rho_interp = np.interp(rr, r_use, rho_use)

    integral = np.trapz(rho_interp * rr**2, rr)
    return 3.0 * integral / (R_core**3)

# def calculate_rho_core(_r, _rho, _elements):
#     """
#     Core density for Rprocedure case. We can estimate
#     this by using data stored in .txt file.
#
#     :param _rho: density at fixed radi, `rho(r)`.
#         (array or list)
#     :param _r: fixed radi.
#         (array or list)
#     :param _elements: how many first radius's we treat as belonging to a core.
#         (int)
#     :return: core density
#     -------
#     Important note: data coming from gravothermal simulation have density value
#     at `r = 0.0 kpc`. That is reason, why we set parameter: `_elements`.
#     """
#     rho_core = _rho[0] * (_r[0] / _r[_elements - 1]) ** 3
#
#     for i in range(1, _elements):
#         part_volume = (_r[i] / _r[_elements - 1]) ** 3 - (_r[i - 1] / _r[_elements - 1]) ** 3
#         rho_core = rho_core + part_volume * _rho[i]
#
#     return rho_core
def calculate_veldis_core(_r, _veldis, _elements):
    """
    Core velocity dispersion for gravothermal case. We can estimate
    this by using data stored in .txt file. Now I take algebraical sum/

    :param _veldis: velocity dispersion at fixed radi, `nu(r)`.
        (array or list)
    :param _r: fixed radi.
        (array or list)
    :param _elements: how many first radius's we treat as belonging to a core.
        (int)
    :return: velocity dispersion at core.
    -------
    Important note: data coming from gravothermal simulation have density value
    at `r = 0.0 kpc`. That is reason, why we set parameter: `_elements`.
    """
    veldis_core = 0
    for i in range(0, _elements):
        veldis_core = veldis_core + _veldis[i] / _elements

    return veldis_core

# ################################################# SELECTION MERGING (ISOTHERMAL) ################################### #
def calRelErr_mergin(rho_high, rho_low, rel_err_margin=1.0, default_selection=True):
    """
    Calculate the relative differences between central DM density, for
    physical and unphysical case. And check that merging is happening.

    Update: Now when I'm thinking the better way of finding minimum
    is input the `dimensionless` value of density parameters: in that case
    we are somehow independent of the viral mass and so on.

    where,
        rho_high: the unphysical value of density: central DM density.
            (float) [M_sun/kpc^3] or [dimensionless]
        rho_low: the physical value of density: central DM density.
            (float) [M_sun/kpc^3] or [dimensionless]
        rel_err_margin: the relative difference where merging should be occurred.
            (float)
        default_selection: type of selection that you preferred. The default selection
            is given by eq. (1) .
            (True or False)
    ----------
    One important remark: the density value, which we get from:

            ISO_CDM.retrun_rho0_LoDen() -> to more details go to file: `IsothermalSIDMModel.py`
            ISO_CDM.retrun_rho0_HiDen() -> to where we implement the `ISO_CDM` class.

    I mean the: `central DM density [M_sun/kpc^3] (float)`, usually is enormous
    large number. So if we're looking for merging time, we want them to be comparable.
    To do so we can use fixed value of relative error:

        (1) rel_err = |rho_high - rho_low| / rho_low,
        (2) rel_err = |2*rho_high - rho_low| / rho_low,

    for sets of parameters that I use, I always get that: rho_high > rho_low.
    In the (1) method to detect, where the merging occurs we for example search
    when the:

        (1) rel_err < 1e-5,

    or other very small number (much smaller than 1). For the case (2)
    we have:

        (2) rel_err < 1.0,

    Then we have merging (or we can use different value around 1)
    """
    if default_selection is True:
        '''
        approximately:  rel_err_margin = 1.0
        '''
        rel_err = abs(2*rho_high - rho_low) / rho_low

        if rel_err_margin > rel_err:
            return True
        else:
            return False
    else:
        '''
        approximately:  rel_err_margin = 1e-5
        '''
        rel_err = abs(rho_high - rho_low) / rho_low

        if rel_err_margin > rel_err:
            return True
        else:
            return False

def calErr_mergin(rho_high, rho_low, err_margin=1e-3):
    """
    Calculate the differences between central DM density, for
    physical and unphysical case. And check that merging is happening.

    where,
        rho_high: the unphysical value of density: central DM density [M_sun/kpc^3].
            (float)
        rho_low: the physical value of density: central DM density [M_sun/kpc^3].
            (float)
        err_margin: the difference where merging should be occurred.
            (float)
    ----------
    """
    err = abs(rho_high - rho_low)
    if err_margin > err:
        return True
    else:
        return False

# ################################################# GRAVOTHERMAL ##################################################### #
"""
In this part we will write some function, which will try connect
our develop of R-procedure with gravothermal.
"""
def fun_a(_z):
    return 0.520 + (0.905 - 0.520)*np.exp(-0.617 * _z**1.21)

def fun_b(_z):
    return -0.101 + 0.026 * _z

def DMc_gravo(_Mvir, _z, _h=cfg.const_h):
    """
    Compute DM concentration in the same way as in the mathematica file,
    which we implement the computing gravothermal.

    where,

        Mvir: value of viral mass.
            (float) [M_sun]
    """
    c = 10**(fun_a(_z)) * (_Mvir / (10**12 * _h**-1))**(fun_b(_z))
    return c

def cal_r_s(_mvir, _z):
    """
    Find `r_s`.

    z: redshift, usually `z=0`. (float)
    Mvir: value of viral mass.
    (float) [M_sun]
    """
    return (200 * 4 / 3 * np.pi * cfg.rho_c * 10**-9) ** (-1/3) * _mvir ** (1/3) * \
           (DMc_gravo(_mvir, _z))**(-1) * 10 ** (-3)  # [kpc]

def cal_rho_s(_mvir, _z):
    """
    Find `rho_s`.

    z: redshift usually `z=0`. (float)
    Mvir: value of viral mass.
    (float) [M_sun]
    """
    return _mvir / (4 * np.pi * (cal_r_s(_mvir, _z)) ** 3) \
        * 1 / (np.log(1 + DMc_gravo(_mvir, _z)) - DMc_gravo(_mvir, _z) / (1 + DMc_gravo(_mvir, _z)))
