"""
Here we have implementation of method to join the `halo DM`(`hDM`) and `ISO` case in one. If our system evolve
it will change the Isothermal profile, so `hDM` it will somehow fit to the ISO. In general, we will use:

    `halo DM` == `NFW`.

This file contains a class, which can find out how should look velocity dispersion after sticking
procedure with density profiles of `ISO` and `hDM`. In the past it have been problematic for us,
to develop those calculation in not-time consuming way. Fortunately at now we have solved it
- so, getting velocity dispersion should be pretty fast.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
# ------------ IMPORT FROM FILES ------------ #
import config as cfg

# ------------ INTERPOLATION FUNCTION ------------ #
def do_cubic_interpolation(r_arr, y_rr, r_new_arr, extrapolate=True):
    """
    Interpolates y_lis data using cubic splines.

    Parameters:
    r_arr (array): array of radi, where successive r's are in exponential intervals (r = np.logspace())
        [kpc]
    y_rr (array):  array of densities, enclosed mass or velocity dispersion, which are connected
        to array of radius's: `\rho(r)`, `M(<r)`, `velDis(r)`.
        [M_sun/kpc^3], [M_sun], [kpc/Gyr]
    r_new_arr (float): array of radi, where successive r's are in exponential intervals (r = np.logspace()).
        This array corresponds to grid of radi, which we want to have (or we're looking for `rho(r)`, where
        `r` belongs to `r_new_arr`).
        [kpc]

    Returns: `\rho(r)`, `M(<r)` or `velDis(r)`, where `r` belongs to `rnew_arr`.
    """
    # Get the interpolation function
    yFunction = CubicSpline(r_arr, y_rr, extrapolate=extrapolate)

    # Evaluating the interpolation at new points
    ynew_arr = yFunction(r_new_arr)
    return ynew_arr

def rho_interpolation1d(old_r, old_rho, new_r, extrapolate=True):
    """
    Function which does interpolation and (or) extrapolation. Keep in mind that
    radi we stored in class is in logarithmic manner (we used `np.logspace`).
    We will use to get density in `ISO` regime.

    where:
        new_r: the value of radi for which we want to know the value of density.
            (array or float) [kpc]
        extrapolate: if is set in `True`, the extrapolation may be possible. In another case
            the extrapolation will be forbidden.
    ----------
    To use interpolation we have to have stored data (will be stored in class):

        old_r: base radius to do interpolation. Notice that we take: `r`.
            (array) [kpc]
        old_rho: base densities to do interpolation. Notice that we take: `\rho_ISO(r)`.
            (array) [M_sun/kpc^3]
    ---------
    return: \rho_ISO(new_r): interpolated or extrapolated value of density at `new_r`.
        (array or float) [M_sun/kpc^3]
    """
    # do interpolation or extrapolation
    if extrapolate is True:
        density_interp_func = interp1d(old_r, old_rho, kind='linear', fill_value="extrapolate")
    else:
        density_interp_func = interp1d(old_r, old_rho, kind='linear',
                                       fill_value=0)  # if something out of prepared set zero
    density_interp = density_interp_func(new_r)
    return density_interp  # [M_sun/kpc^3]

def M_interpolation1d(old_r, old_mass, new_r, extrapolate=True):
    """
    Function which does interpolation and (or) extrapolation. Keep in mind that
    radi we stored in class is in logarithmic manner (we used `np.logspace`).
    We will use to get mass in `ISO` regime.

    where:
        new_r: the value of radi for which we want to know the value of density.
            (array or float) [kpc]
        extrapolate: if is set in `True`, the extrapolation may be possible. In another case
            the extrapolation will be forbidden.
    ----------
    To use interpolation we have to have stored data (will be stored in class):

        old_r: base radius to do interpolation. Notice that we take: `r`.
            (array) [kpc]
        old_mass: base mass to do interpolation. Notice that we take: `M_ISO(r)`.
            (array) [M_sun]
    ---------
    return: M_ISO(new_r): interpolated or extrapolated value of mass at `new_r`.
        (array or float) [M_sun]
    """
    # do interpolation or extrapolation
    if extrapolate is True:
        mass_interp_func = interp1d(old_r, old_mass, kind='linear', fill_value="extrapolate")
    else:
        mass_interp_func = interp1d(old_r, old_mass, kind='linear',
                                    fill_value=0)  # if something out of prepared set zero
    mass_interp = mass_interp_func(new_r)
    return mass_interp  # [M_sun]

# ####################################### NEW VERSION ####################################### #
class IsoAndHalo(object):
    """
    This class is used to describe the galaxy profile. which contains `ISO` and `halo DM`(`hDM`) part.
    Firstly we have to set the initial profile (density and so on) of galaxy - in our case
    it will be `NFW` (in general). Then we used that to estimate how can will change in time
    - now appears the Isothermal part. with later evolution time, more and more of the entire
    galaxy's profile is governed by `Isothermal` part.

    More details in file: `IsothermalSIDMModel.py` and in the paper: 2206.12425 .

    This class will be used to described and collect data at fixed time of galaxy evolution.
    The whole profile of galaxy is separated in two parts and the border is `r1`.

        (1) `r < r1`: governed by `ISO`.
        (2) 'r > r1': governed by 'hDM'.

    One important remark - the example who this parting occurs:
        \rho(r) = \rho_hDM(r) if r > r1
        \rho(r) = \rho_ISO(r) if r < r1
    """
    def __init__(self, _ISO_data, halo, r1):
        """
        Initialize Velocity dispersion. Important remark: this class has constructed
        assuming that we have Isothermal region (for r < r1) and hDM region.

        where:


            r1:  The characteristic radius of a SIDM halo at which an average particle
                 is scattered once during the age of the halo, defined as the solution
                 [kpc] (float).
            halo: the halo profile for the CDM-like outskirt
                (object).

            _ISO_data: is list contains necessary data coming from the `ISO` profile.

            _ISO_data["r"]: _r_ISO: the array contains all radius, which is connected to the isothermal region.
                Important remark: have been the same as in all _r.
                [kpc] (array)

            _ISO_data["rho"]: _rho_ISO: is array contains the densities, which has been connected to the radius
                (_rho[i] is the value of density in distance: _r[i]). So, it stores information
                about: \rho(tmarge, r), where time is fixed.
                [M_sun/kpc^3] (array)

            _ISO_data["mass"]: _M_ISO: enclosed mass up to `r` radi. So, it stores information
                about: mass(tmarge, <r), where time is fixed.
                [M_sun] (array)
        """
        # --- Base variables
        self.r1 = r1  # [kpc]
        self.halo = halo

        # --- Constants
        self.G = cfg.const_G_starUnits  # [kpc^3 Gyr^-2 M_sun^-1]

        # --- ISO profile
        self.r_ISO_data = _ISO_data["r"]  # [kpc]
        self.rho_ISO_data = _ISO_data["rho"]  # [M_sun/kpc^3]
        self.M_ISO_data = _ISO_data["mass"]  # [M_sun]
        self.Nr_ISO = len(self.r_ISO_data)

        # --- Precompute interpolation functions
        self.rho_ISO_interp = interp1d(_ISO_data["r"], _ISO_data["rho"], kind='linear', fill_value="extrapolate")
        self.M_ISO_interp = interp1d(_ISO_data["r"], _ISO_data["mass"], kind='linear', fill_value="extrapolate")
        """
        `ISO` regime is up to `r = r1`. Beside that we have to consider the
        `hDM` profile. So we can introduce a variable, which describes
        the value of enclosed mass up to `r1`.
        """
        self.MassTrans = self.M_ISO(self.r1)  # [M_sun], transmission mass

    # -------------------------------- halo: MASS AND DENSITY -------------------------------- #
    def rho_hDM(self, _r):
        """
        Density [M_sun / kpc^3] at radius _r. Connected to halo profile.

        where:
            _r: r-coordinate. Keep in mind that `hDM` profile is valid for: `r > r1`.
                [kpc] (float or array)
        ---------
        return: Density [M_sun / kpc^3] at radius _r. Connected to halo profile.
            [M_sun] (float or array)
        """
        return self.halo.rho(_r)  # [M_sun / kpc^3]

    def M_hDM(self, _r, _r1):
        """
        Mass connected to halo profile. In our model we want to compute the enclosed
        mass, but we have to remember that for r < r1 is the isothermal region. So,
        this function will compute only the enclosed mass above the isothermal region.
        We can think ab out it that we want to calculate the mass, which was contains
        in big ball with radius: `_r` with hollow center of the sphere, where the
        hollow has radius: `_r1`.

        Mass [M_sun] of dark matter enclosed up to radius: `_r` with a hollow radius: `_r1`.

            M_halo(<r) = \int^r \rho_halo(r) dV, where: dV  = 4 * pi * r**2 ,

        where we integrate by shell's from `r1` up to `r` and taking into account `M_ISO(<r1)`.
        M_halo(<r) is enclosed mass up to r. Doing some algebra we ends in:

            M_halo(r,r1) == M_halo(r1<r) = M_ISO(<r1) + (M_halo(r) - M_halo(r1)),

        where:
            _r: r-coordinate, upper radius.
                [kpc] (float or array)
            _r1: r-coordinate, hollow radius.
                [kpc] (float or array)
            M_ISO(<r1): mass corresponding to `ISO` regime (up to `r1`). This is constant!
                [M_sun] (float)
        ---------
        return: Mass [M_sun] of dark matter enclosed up to radius: '_r', taking into account `ISO` profile
            as a hollow up to radi: `r1`.
            [M_sun] (float or array)
        """
        return self.MassTrans + self.halo.Mass(_r) - self.halo.Mass(_r1)  # [M_sun]

    # -------------------------------- ISO: MASS AND DENSITY -------------------------------- #
    def rho_ISO(self, r):
        """
        Density [M_sun / kpc^3] at radius _r. Connected to ISO profile.
        Using stored data we will interpolate / extrapolate this in fixed radius.

        where:
            _r: r-coordinate. Keep in mind that ISO profile is valid for: `r < r1`.
                [kpc] (float or array)
        ---------
        return: Density [M_sun / kpc^3] at radius _r. Connected to ISO profile.
             (array or float) [M_sun/kpc^3]
        """
        return self.rho_ISO_interp(r)  # [M_sun/kpc^3]

    def M_ISO(self, r):
        """
        Enclosed mass in the `ISO` regime. Using the stored data we will interpolate
        or extrapolate this value (in a fixed radi).

        where:
            _r: r-coordinate. Keep in mind that ISO profile is valid for: `r < r1`.
                [kpc] (float or array)
        ---------
        return: Enclosed mass [M_sun] at radius _r. Connected to ISO profile.
             (array or float) [M_sun]
        """
        return self.M_ISO_interp(r)  # [M_sun]

    # -------------------------------- ISO + hDM: RETURN DATA -------------------------------- #
    def M_enclosed_return(self, r_list):
        """
        Compute enclosed mass in `fixed profile` at given radius's.

        where,

            r_list: r-coordinate. [kpc] (array)
        ---------
        return: Enclosed mass [M_sun] for some array `_r`. Here we consider whole galaxy
            profile, so `ISO` regime and `hDM` regime. It means array `_r` is arbitrary.
            (array) [M_sun]
        """
        Mass = np.array([])
        N = len(r_list)
        index_ISO = 0  # index of last radi, which belongs to `ISO` profile
        for i in range(0, N):
            r = r_list[i]
            if r > self.r1:
                index_ISO = i
                break

        # `ISO`
        for i in range(0, index_ISO + 1):
            mass = self.M_ISO(r_list[i])
            # appending
            Mass = np.append(Mass, mass)

        # `hDM`
        for i in range(index_ISO + 1, N):
            mass = self.M_hDM(r_list[i], self.r1)
            # appending
            Mass = np.append(Mass, mass)

        return Mass  # [M_sun]

    def rho_return(self, r_list):
        """
        Compute density in `fixed profile` at given radius's.

        where,

            r_list: r-coordinate. [kpc] (array)
        ---------
        return: Density [M_sun/kpc^3] for some array `_r`. Here we consider whole galaxy
            profile, so `ISO` regime and `hDM` regime. It means array `_r` is arbitrary.
            (array) [M_sun/kpc^3]
        """
        rho_list = np.array([])
        N = len(r_list)
        index_ISO = 0  # index of last radi, which belongs to `ISO` profile
        for i in range(0, N):
            r = r_list[i]
            if r > self.r1:
                index_ISO = i
                break

        # `ISO`
        for i in range(0, index_ISO + 1):
            rho = self.rho_ISO(r_list[i])
            # appending
            rho_list = np.append(rho_list, rho)

        # `hDM`
        for i in range(index_ISO + 1, N):
            rho = self.rho_hDM(r_list[i])
            # appending
            rho_list = np.append(rho_list, rho)

        return rho_list  # [M_sun/kpc^3]

    # -------------------------- VELOCITY DISPERSION --------------------------#
    def velDis_return(self, r_list):
        """
        Compute a velocity dispersion. To do that we will use two additional
        function. One will be focused on `ISO` + `hDM` profile and second
        will gonna interested only in `hDM` regime. Do to that we have to
        solve Jeans equation. IN general, we have:

            (\nu(r))^2 = 1 / rho(r) * int^{\infty}_{r} dr * rho(r') * G * M(<r') / (r')^2

        We consider or distinguish two cases: (1) when `r < r1` and (2) when `r > r1`. Let's focus
        on the case (1)

            (\nu(r))^2 = 1 / rho(r) * int^{\infty}_{r} dr * rho(r') * G * M(<r') / (r')^2 =
                       = 1 / rho(r) * int^{r1}_{r} dr * rho(r') * G * M(<r') / (r')^2 +
                       + 1 / rho(r) * int^{\infty}_{r1} dr * rho(r') * G * M(<r') / (r')^2 =
                       = 1 / rho_ISO(r) * int^{r1}_{r} dr * rho_ISO(r') * G * M_ISO(<r') / (r')^2 +
                       + 1 / rho_ISO(r) * int^{\infty}_{r1} dr * rho_hDM(r') * G * M_hDM(<r') / (r')^2

        where, we used that we consider two separates regions, which have border at `r1`. For case (2)
        we have:

            (\nu(r))^2 = 1 / rho(r) * int^{\infty}_{r} dr * rho(r') * G * M(<r') / (r')^2 =
                       = 1 / rho_hDM(r) * int^{\infty}_{r} dr * rho_hDM(r') * G * M_hDM(<r') / (r')^2

        --------------
        To sum up:

            (1) when `r < r1`, (\nu(r))^2 = 1 / rho_ISO(r) * int^{r1}_{r} dr * rho_ISO(r') * G * M_ISO(<r') / (r')^2 +
                                          + 1 / rho_ISO(r) * int^{\infty}_{r1} dr * rho_hDM(r') * G * M_hDM(<r') / (r')^2

            (2) when `r > r1`, (\nu(r))^2 = 1 / rho_hDM(r) * int^{\infty}_{r} dr * rho_hDM(r') * G * M_hDM(<r') / (r')^2

        -------------
        Arguments:

            r_list: list of radi, at which we want to compute the velocity dispersion.
                [kpc] (array or list)
        -------------
        return: velocity dispersion [kpc/Gyr] for some array `_r`. Here we consider whole galaxy
            profile, so `ISO` regime and `hDM` regime. It means array `_r` is arbitrary.
            (array) [kpc/Gyr]
        """
        Nr = len(r_list)
        velDis = np.array([])

        for i in range(0, Nr):
            r = r_list[i]

            if r < self.r1:
                rho = self.rho_ISO(r)
                velDis_i = self.compute_velDis_ISO(r, rho)
            else:
                rho = self.rho_hDM(r)
                velDis_i = self.compute_velDis_hDM(r, rho)

            velDis = np.append(velDis, velDis_i)
            # print('step:', i)

        return velDis  # [kpc/Gyr]

    def compute_velDis_ISO(self, r, rho):
        """
        Realization of (1) in calculation velocity dispersion.

            (1) when `r < r1`, (\nu(r))^2 = 1 / rho_ISO(r) * int^{r1}_{r} dr * rho_ISO(r') * G * M_ISO(<r') / (r')^2 +
                                          + 1 / rho_ISO(r) * int^{\infty}_{r1} dr * rho_hDM(r') * G * M_hDM(<r') / (r')^2
                                          = \nu_square_1a + \nu_square_1b
        ----------
        where,

            r: radi in the `ISO` regime (`r < r1`) at we're now calculating velocity dispersion.
                [kpc] (float)
            rho: density corresponding to fixed radi.
                [M_sun/kpc^3] (float)
        -------------
        return: velocity dispersion [kpc/Gyr] for some array `_r`. Here we consider `ISO` regime,
            so when: `r < r1`.
            (array) [kpc/Gyr]
        """
        # --- `ISO` part
        # Create an array of points between r and self.r1 for integration
        r_points = np.linspace(r, self.r1, 1000)
        integrand = self.rho_ISO(r_points) * self.G * self.M_ISO(r_points) / r_points ** 2

        # Perform the integration using the trapezoidal rule
        I = np.trapz(integrand, r_points)
        nu_square_1a = 1 / rho * I

        # --- `hDM` part
        g = lambda r: self.rho_hDM(r) * self.G * self.M_hDM(r, self.r1) / r ** 2
        # II = quad(g, self.r1, np.inf, epsabs=1.e-7, epsrel=1.e-6, limit=10000)[0]
        II = quad(g, self.r1, np.inf)[0]
        nu_square_1b = 1 / rho * II

        return np.sqrt(nu_square_1a + nu_square_1b)  # [kpc/Gyr]

    def compute_velDis_hDM(self, r, rho):
        """
        Realization of (2) in calculation velocity dispersion.

            (2) when `r > r1`, (\nu(r))^2 = 1 / rho_hDM(r) * int^{\infty}_{r} dr * rho_hDM(r') * G * M_hDM(<r') / (r')^2
                                          = \nu_square
        -------------
        where,

            r: radi in the `hDM` regime (`r > r1`) at we're now calculating velocity dispersion.
                [kpc] (float)
            rho: density corresponding to fixed radi.
                [M_sun/kpc^3] (float)
        -------------
        return: velocity dispersion [kpc/Gyr] for some array `_r`. Here we consider `hDM` regime,
            so when: `r > r1`.
            (array) [kpc/Gyr]
        """
        G = cfg.const_G_starUnits  # [kpc^3 Gyr^-2 M_sun^-1]
        f = lambda r: self.rho_hDM(r) * G * self.M_hDM(r, self.r1) / r ** 2
        # I = quad(f, r, np.inf, epsabs=1.e-7, epsrel=1.e-6, limit=10000)[0]  # DOES NOT ALLOW LOGSPACE
        I = quad(f, r, np.inf)[0]  # DOES NOT ALLOW LOGSPACE
        nu_sqare = 1 / rho * I

        return np.sqrt(nu_sqare)  # [kpc/Gyr]

    # -------------------------- CREATE DATA --------------------------#
    def create_data(self, r_low=-2.0, r_up=2.0, nr=400):
        """
        This function will produce data for fixed radius. We have to use the galaxy profile,
        which have been already stored in class, so `ISO` profile (`ISO_data`) and `hDM`
        profile (corresponding to outskirt profile of CDM). Then we will find how looks
        velocity dispersion for galaxy (the accuracy of that calculation comes from `ISO_data`).
        Next, we will find out interpolation function, of our interest. In other words, we have
        to find:

            density interpolation: rho(r),
            enclosed mass interpolation: M(<r),
            velocity dispersion interpolation: nu(r).

        In the end, we will create data list, based on the radius's we want (in code: `rNew_arr`).

        Where,

            r_low (float): lower limit for radi bin (output / after interpolation). Typically: `r_low = -2.0`.
            r_up (float): upper limit for radi bin (output / after interpolation). Typically: `r_up = 2.0`.
            nr (int): how many point we want in bin (output / after interpolation). Typically: `nr = 400`.
        -------------
        return: map[r, rho(r), Mass(<r), nu(r)] # [M_sun/kpc^3], [M_sun], [kpc/Gyr]
        """
        # --- Do interpolation
        IsoAndHalo_data = dict()
        # prepare `r` array
        IsoAndHalo_data['r'] = np.logspace(np.log10(self.halo.r_s * 10 ** r_low),
                                           np.log10(self.halo.r_s * 10 ** r_up),
                                           nr, endpoint=True)  # [kpc]

        # calculate: `rho(r)`
        IsoAndHalo_data['rho'] = self.rho_return(IsoAndHalo_data['r'])  # [M_sun/kpc^3]
        # calculate: `M(<r)`
        IsoAndHalo_data['mass'] = self.M_enclosed_return(IsoAndHalo_data['r'])  # [M_sun]
        # calculate: `velDis(r)`
        IsoAndHalo_data['velDis'] = self.velDis_return(IsoAndHalo_data['r'])  # [kpc/Gyr]
        return IsoAndHalo_data
