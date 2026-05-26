"""
!!!!!!!!!!!!!!!!!!!! BUGS: WORKING PROGRES !!!!!!!!!!!!!!!!!!!!

Here we implement the method: core NFW, which describes the evolution of the dark matter
during Rprocedure.

Plus: calculate really quickly the velocity dispersion through the R-procedure simulation.
Minus: the velocity dispersion, which we obtained from `coreNFWProfile` is not accurate to put
    into gravothermal simulation (because of dimple for small radius).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.optimize import curve_fit
# ------------ IMPORT FROM FILES ------------ #
import config as cfg

# ############################ CREATE CLASS ############################ #
class CoreNFWProfile(object):
    """
    Class that implements the "coreNFW" profile (Read+2016):
        M(r) = M_NFW(r) g(y)
        rho(r) = rho_NFW(r) g(y) + [1-g(y)^2] M_NFW(r) / (4 pi r^2 r_c)

    in a cylindrical frame (R,phi,z), where

        r = sqrt(R^2 + z^2)
        y = r / r_c with r_c a core radius, usually smaller than r_s
        g(y) = tanh(y)

    Syntax:

        halo = coreNFW(M,c,rc,Delta=200.,z=0.)

    where

        M: halo mass [M_sun], where halo is defined as spherical
            overdensity of Delta times critical density (float)
        c: NFW halo concentration (float)
        rc: core radius [kpc]
        Delta: average overdensity of the halo, in multiples of the
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)

    Attributes:

        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rc: core radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .Vmax: maximum circular velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius

    Methods:

        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z)
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)

    HISTORY: Arthur Fangzhou Jiang (2021-03-11, Caltech)
    """

    def __init__(self, M, c, rc, Delta=200., z=0.):
        """
        Initialize coreNFW profile.

        Syntax:

            halo = coreNFW(M,c,rc,Delta=200.,z=0.)

        where

            M: halo mass [M_sun] (float),
            c: halo concentration (float),
            rc: core radius [kpc]
            Delta: spherical overdensity with respect to the critical
                density of the universe (default is 200.)
            z: redshift (float)
        """
        # input attributes
        self.Mh = M
        self.ch = c
        self.rc = rc
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = cfg.rhoc(z, cfg.h, cfg.Om, cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3. * self.Mh / (cfg.FourPi * self.rhoh)) ** (1. / 3.)
        self.rs = self.rh / self.ch
        self.xc = self.rc / self.rs
        self.rmax = self.rs * 2.163  # accurate only if r_c < r_s
        self.rho0 = self.rhoc * self.Deltah / 3. * self.ch ** 3. / self.f(self.ch)
        self.Phi0 = -cfg.FourPiG * self.rho0 * self.rs ** 2.
        self.Vmax = self.Vcirc(self.rmax)  # accurate only if r_c < r_s
        self.s001 = self.s(0.01 * self.rh)

    def f(self, x):
        """
        Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)

        Syntax:

            .f(x)

        where

            x: dimensionless radius r/r_s (float or array)
        """
        return np.log(1. + x) - x / (1. + x)

    def g(self, y):
        """
        Auxiliary method for coreNFW profile: f(y) = tanh(y)

        Syntax:

            .g(y)

        where

            y: dimensionless radius r/r_c (float or array)
        """
        return np.tanh(y)

    def rho(self, R, z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2).

        Syntax:

            .rho(R,z=0.)

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.rs
        y = r / self.rc
        f = self.f(x)
        g = self.g(y)
        return self.rho0 * (g / (x * (1. + x) ** 2) + (1. - g ** 2) * f / (self.xc * x ** 2.))

    def s(self, R, z=0.):
        """
        Logarithmic density slope

            - d ln rho / d ln r

        at radius r = sqrt(R^2 + z^2).

        Syntax:

            .s(R,z=0.)
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        r1 = r * (1. + cfg.eps)
        r2 = r * (1. - cfg.eps)
        rho1 = self.rho(r1)
        rho2 = self.rho(r2)
        return - np.log(rho1 / rho2) / np.log(r1 / r2)

    def Mass(self, R, z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).

        Syntax:

            .M(R,z=0.)

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.rs
        y = r / self.rc
        return cfg.FourPi * self.rho0 * self.rs ** 3. * self.f(x) * self.g(y)

    def rhobar(self, R, z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2).

        Syntax:

            .rhobar(R,z=0.)

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        return self.M(r) / (cfg.FourPiOverThree * r ** 3)

    def tdyn(self, R, z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
        Syntax:

            .tdyn(R,z=0.)

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R, z))

    def Phi_accurate(self, R, z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).

        Syntax:

            .Phi_accurate(R,z=0.)
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.rs
        Phi1 = - cfg.const_G_starUnits * self.M(r) / r
        if isinstance(x, list) or isinstance(x, np.ndarray):
            if len(x.shape) == 1:  # i.e., if the input R array is 1D
                I = []
                for xx in x:
                    # II = quad(self.dIdx_Phi, xx, self.ch,)[0]
                    II = quad(self.dIdx_Phi, xx, np.inf, )[0]
                    I.append(II)
                I = np.array(I)
            elif len(x.shape) == 2:  # i.e., if the input R array is 2D
                I = np.empty(x.shape)
                for i, xx in enumerate(x):
                    for j, xxx in enumerate(xx):
                        # II = quad(self.dIdx_Phi, xxx, self.ch,)[0]
                        II = quad(self.dIdx_Phi, xxx, np.inf, )[0]
                        I[i, j] = II
        else:
            I = quad(self.dIdx_Phi, x, self.ch, )[0]
        Phi2 = self.Phi0 * I
        return Phi1 + Phi2

    def dIdx_Phi(self, x):
        """
        Integrand for the second-term of the potential of coreNFW.
        """
        f = self.f(x)
        g = self.g(x / self.xc)
        return g / (1. + x) ** 2 + (1. - g ** 2) * f / (x * self.xc)

    def Phi(self, R, z=0.):
        """
        Approximation expression for gravitational potential
        [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2):

            Phi(x) ~ [1+s(x)] Phi_core + s(x) Phi_NFW(x)

        where

            x = r/r_s
            Phi_core ~ Phi_NFW(0.8 x_c) is the flat potential in the core
            Phi_NFW(x) is the NFW potential

        For exact (but slower evaluation of the) potential, use
        .Phi_accurate

        Syntax:

            .Phi(R,z=0.)
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.rs
        xtrans = 0.8 * self.xc  # an empirical transition scale
        s = 0.5 + 0.5 * np.tanh((x - xtrans) / xtrans)  # transition function
        Phic = self.Phi0 * np.log(1. + xtrans) / xtrans
        PhiNFW = self.Phi0 * np.log(1. + x) / x
        return (1. - s) * Phic + s * PhiNFW

    def fgrav(self, R, z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)

            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]

        Syntax:

            .fgrav(R,z)

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)

        Note that unlike the other methods, where z is optional with a
        default of 0, here z must be specified.

        Return:

            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.rs
        y = r / self.rc
        fac = self.Phi0 * (self.g(y) * self.f(x) / x) / r ** 2.
        return fac * R, fac * 0., fac * z

    def Vcirc(self, R, z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).

        Syntax:

            .Vcirc(R,z=0.)

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        return np.sqrt(r * -self.fgrav(r, 0.)[0])

    def rmax_accurate(self):
        """
        Radius [kpc] at which maximum circular velocity is reached, which
        is given by the root of:

            g(y)/(1+x)^2 - f(x)g(y)/x^2 + [1-g(y)^2]f(x)/(x x_c) = 0

        where

            x = r/r_s
            x_c = r_c / r_s
            y = r/r_c = x/x_c
            g(y) = tanh(y)
        """
        xmax = brentq(self.Findxmax, 0.1, 10., args=(),
                      xtol=0.001, rtol=1e-5, maxiter=1000)
        return xmax * self.rs

    def Findxmax(self, x):
        """
        The left-hand-side function for finding x_max = r_max / r_s.
        """
        f = self.f(x)
        g = self.g(x / self.xc)
        return g / (1. + x) ** 2 - f * g / x ** 2 + (1. - g ** 2) * f / x / self.xc

    def sigma_accurate(self, R, z=0., beta=0.):
        """
        Velocity dispersion [kpc/Gyr].

        Syntax:

            .sigma(R,z=0.)

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
            beta: anisotropy parameter (default=0., i.e., isotropic)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.rs
        y = r / self.rc
        if isinstance(x, list) or isinstance(x, np.ndarray):
            I = []
            for xx in x:
                II = quad(self.dIdx_sigma, xx, np.inf, args=(beta,))[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf, args=(beta,))[0]
        f = self.f(x)
        g = self.g(y)
        A = g / (x * (1. + x) ** 2) + (1. - g ** 2) * f / (self.xc * x ** 2)
        sigmasqr = -self.Phi0 / x ** (2. * beta) / A * I
        return np.sqrt(sigmasqr)

    def dIdx_sigma(self, x, beta):
        """
        Integrand for the integral in the velocity dispersion of Burkert.
        """
        f = self.f(x)
        g = self.g(x / self.xc)
        return (g / (x * (1. + x) ** 2) + (1. - g ** 2) * f / (self.xc * x ** 2)) * \
               f * g * x ** (2. * beta - 2.)

    def sigma(self, R, z=0.):
        """
        Approximation expression for velocity dispersion [kpc/Gyr] at
        radius r = sqrt(R^2 + z^2), assuming isotropic velicity.

        For exact (but slower evaluation of the) dispersion, use
        .sigma_accurate

        Syntax:

            .sigma(R,z=0.)

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.rs
        xtrans = 0.8 * self.xc  # an empirical transition scale
        s = 0.5 + 0.5 * np.tanh((x - xtrans) / xtrans)  # transition function
        sigmac = self.Vmax * 1.4393 * xtrans ** 0.354 / (1. + 1.1756 * xtrans ** 0.725)
        sigmaNFW = self.Vmax * 1.4393 * x ** 0.354 / (1. + 1.1756 * x ** 0.725)
        return (1.-s)*sigmac + s*sigmaNFW

# ####################### FIT AND PREPARE coreNFW ####################### #
def f_NFW(x):
    """
    Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)
    where,

        x: dimensionless radius r/r_s (float or array)
    """
    return np.log(1. + x) - x / (1. + x)

def g_coreNFW(y):
    """
    Auxiliary method for coreNFW profile: f(y) = tanh(y)
    where

        y: dimensionless radius r/r_c (float or array)
    """
    return np.tanh(y)

def coreNFW_mass(r, r_s, rho_s, r_c):
    """
    fitting function in model: `core NFW`.

    where:
        rho_s: density from NFW one of fixed parameter in that model.
            (fla=oat) [M_sin/kpc^3]
         r_s: radi from NFW one of fixed parameter in that model.
            (float) [kpc]
        r: radi at we doing calculation.
            (array or float) [kpc]
        r_c: parameter, which we can get from fitting to our data (e.g. after tmerge).
            (float) [kpc]
    """
    x = r / r_s
    y = r / r_c
    return cfg.FourPi * rho_s * r_s ** 3. * f_NFW(x) * g_coreNFW(y)

def find_fitting_parm(NFW_and_ISO, r_list):
    """
    We use this function to find parameter: `r_c` in formula:

        Mass(r) = Mass_NFW(r) * tanh(r / r_c) ,

    To do that we will use our data, radius: `r_ISO` and corresponding values of
    densities: `rho_ISO`. Which was created from simulation of Isothermal.
    ---------------
    where,
        NFW_class: is prepared NFW class (with fixed viral mass and DM concentration) after
            taking into account the Isothermal evolution! More details you will get in the
            file `IsoAndHalo.py`.
            (object)
        r_list: array of radi, where we want to find approximate formula
            (array) [kpc]
    ---------------
    return:
        c_r: parameter fitting to data.
    """
    # ----- in the beginning we want to find the mass of the galaxy (during time evolution)
    mass_ISO_NFW = NFW_and_ISO.M_enclosed_return(r_list)

    # ----- Find parameter r_c
    r_s = NFW_and_ISO.r_s
    rho_s =NFW_and_ISO.rho_s

    def mass_fit(r, r_c):
        """
        function to find parameter r_c. More detail in the function: `coreNFW_mass`.
        Model is `core NFW`
        """
        x = r / r_s
        y = r / r_c
        return cfg.FourPi * rho_s * r_s ** 3. * f_NFW(x) * g_coreNFW(y)

    fitting_parameters = curve_fit(mass_fit, r_list, mass_ISO_NFW, method='lm')[0]
    c_r = fitting_parameters

    return c_r

