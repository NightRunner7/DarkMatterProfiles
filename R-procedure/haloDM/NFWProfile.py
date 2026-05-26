"""
This file contains a class object: `NFWProfile`, which describes the profile of
dark matter in basic approach in this case the whole dark matter can be described
using viral mass (_M_vir) and dark matter concentration (_con).

This file also gives a scheme, how should we name classes, function and variables
related to dark matter halo.
"""
import numpy as np
import math
from scipy.integrate import quad  # odeint
from scipy.optimize import brentq  # fsolve
# ------------ IMPORT FROM FILES ------------ #
import config as cfg
import units as uni


# ######################################## CREATE CLASS ######################################## #
class NFWProfile(object):
    def __init__(self, _M_vir, _con):
        """
        Initialize NFW profile.

        Where:
            _M_vir: halo mass of dark matter [M_sol] (float)
            _con: halo concentration of dark matter (float)

        Physical interpretation:
            x == _con
            rho_vir = rho(r_vir)

            rho_vir = rho_x = x * rho_critical
            M_vir = M_x = 4/3 * pi * rho_vir * r_vir^3
        """
        # input attributes
        self.M_vir = _M_vir  # [M_sun]
        self.con = _con  # [dimensionless]

        # basic variables: rho_s
        coeff = 200 / 3 * self.con ** 3 / self.f(self.con)
        rho_cry = cfg.rho_c  # critical density # [M_solar / kpc^3]
        self.rho_s = (coeff * rho_cry)  # [M_sun / kpc^3]

        # basic variables: r_s
        rho_s = self.rho_s
        denominator = 4 * math.pi * rho_s * self.f(self.con)
        self.r_s = (self.M_vir / denominator) ** (1 / 3)  # [kpc]

        # Constants see original code in Jiang GitHub
        self.rmax = self.r_s * 2.163
        self.Vmax = self.Vcirc(self.rmax)

    # -------------------------------- BASIC VARIABLES -------------------------------- #
    def f(self, x):
        """
        Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)

        where:
            x: dimensionless radius r/r_s (float or array)

        Syntax:
            .f(x)
        """
        return np.log(1. + x) - x / (1. + x)

    # -------------------------------- MASS, DENSITY AND PHI NFW -------------------------------- #

    def rho(self, _r):
        """
        Density [M_sun / kpc^3] at radius _r.

        where:
            _r: r-coordinate [kpc] (float or array)

        Syntax:
            .rho(_r)
        """
        # dimensionless distance
        x = _r / self.r_s

        denominator = x * (1 + x) ** 2

        return self.rho_s / denominator  # [M_sun / kpc^3]

    def Mass(self, _r):
        """
        Mass [M_sun] of dark matter enclosed up to radius _r.

        where:
            _r: r-coordinate [kpc] (float or array)

        Syntax:
            .Mass(_r)
        """
        # dimensionless distance
        x = _r / self.r_s
        cff = 4 * math.pi

        return cff * self.rho_s * self.r_s ** 3 * self.f(x)  # [M_sun]

    def Phi(self, _r):
        """
        Potential [(kpc/Gyr)^2] at radius _r.

         where
            _r: r-coordinate [kpc] (float or array)

        Syntax:
            .Phi(_r)
        """
        # dimensionless distance
        x = _r / self.r_s
        # constants
        G = cfg.const_G_starUnits  # gravitational constant [kpc^3 Gyr^-2 M_sun^-1]
        fourPiG = 4 * math.pi * G

        return -fourPiG * self.rho_s * self.r_s * np.log(1 + x) / x  # [(kpc/Gyr)^2]

    # -------------------------------- DIFFERENT VELOCITY -------------------------------- #

    def Vcirc(self, _r):
        """
        Circular velocity [kpc/Gyr] at radius _r. In different words it is a
        1D dispersion velocity.

        Where:
             _r: r-coordinate [kpc] (float or array)

        Syntax:
            .Vcirc(_r)
        """
        # constants
        G = cfg.const_G_starUnits  # gravitational constant [kpc^3 Gyr^-2 M_sun^-1]
        Mass = self.Mass(_r)  # [M_sun]

        return np.sqrt(G * Mass / _r)  # [kpc/Gyr]

    def Vavg(self, _r):
        """
        Averaged velocity [kpc/Gyr] for DM particles in Maxwell distribution up to radius _r.

        Where:
             _r: r-coordinate [kpc] (float or array)

        Syntax:
            .Vavg(_r)
        """
        # 1D dispersion velocity
        Vcircular = self.Vcirc(_r)  # [kpc/Gyr]
        # constants
        FourOverRootPi = 4. / np.sqrt(np.pi)

        return FourOverRootPi * Vcircular

    # -------------------------------- DIFFERENT VELOCITY -------------------------------- #
    def sigma(self, R):
        """
        Velocity dispersion [kpc/Gyr] at radius r = sqrt(R^2),
        assuming isotropic velicity dispersion tensor, and following the
        Zentner & Bullock (2003) fitting function:

            sigma(x) = V_max 1.4393 x^0.345 / (1 + 1.1756 x^0.725)

        where x = r/r_s.

        Syntax:

            .sigma(R,z=0.)

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2.)
        x = r / self.r_s
        return self.Vmax * 1.4393 * x ** 0.354 / (1. + 1.1756 * x ** 0.725)

    # -------------------------------- VELOCITY DISPERSION -------------------------------- #

    def sigma_accurate(self, _r):
        """
        Velocity dispersion [kpc/Gyr]. (nu)

        Where:
            _r: r-coordinate [kpc] (float or array)

        Syntax:
            .sigma_accurate(R,z=0.)
        """
        # constants
        G = cfg.const_G_starUnits  # gravitational constant [kpc^3 Gyr^-2 M_sun^-1]
        cff = - 4 * np.pi * G * self.rho_s * self.r_s ** 2
        # dimensionless distance
        x = _r / self.r_s

        if isinstance(x, list) or isinstance(x, np.ndarray):
            I = []
            for different_x in x:
                II = quad(self.dIdx_sigma, different_x, np.inf)[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf)[0]

        sigmaSqare = -cff / x ** (- 1) * (1. + x) ** 2 * I
        return np.sqrt(sigmaSqare)

    def dIdx_sigma(self, x):
        """
        Integrand for the integral in the velocity dispersion.

        Where:
            x = r / self.r_s, dimensionless distance

        Syntax:
            .dIdx_sigma(x)
        """
        f = self.f(x)
        return x ** (- 3.) * f / (1. + x) ** 2


# --- functions dealing with composite potential (i.e., potential list) ---
def rho(potential, R):
    """
    Density [M_sun/kpc^3], at location (R,z) in an axisymmetric potential
    which consists of either a single component or multiple components.

    Syntax:

        rho(potential,R,z=0.)

    where

        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the
            first argument R is also the halo-centric radius r)

    Example: we have a potential consisting of an NFW halo and a MN disk,

        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)

    i.e., potential = [halo,disk], and we want to get the density at
    (R,z) in this combined halo+disk host, we use:

        rho([halo,disk],R,z)
    """
    if not isinstance(potential, list):  # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential]
    sum = 0.
    for p in potential:
        sum += p.rho(R)
    return sum


def sigma(potential, R):
    """
    1D velocity dispersion [kpc/Gyr] at (R,z=0), in an axisymmetric
    potential which consists of either a single component or multiple
    components. For composite potential, the velocity dispersion is the
    quadratic sum of that of individual components

        sigma^2 = Sum sigma_i^2

    Syntax:

        sigma(potential,R,z=0):

    where

        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the
            first argument R is also the halo-centric radius r)

    Example: we have a potential consisting of an NFW halo and a MN disk,

        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)

    i.e., potential = [halo,disk], and we want to get the circular
    velocity at (R,z) in this combined halo+disk host, we use:

        sigma([halo,disk],R,z)
    """
    if not isinstance(potential, list):  # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential]
    sum = 0.
    for p in potential:
        sum += p.sigma_accurate(R) ** 2
    return np.sqrt(sum)

# ######################################## FINDING R1 ######################################## #
# It occurs that boy in paper use different way to look for the r1 and I write
# all of his method
def r1(potential, sigmamx=1., tage=1., cff=0.001):
    """
    The characteristic radius of a SIDM halo at which an average particle
    is scattered once during the age of the halo, defined as the solution
    to (e.g., Kaplinghat+16 eq.1)

        (4/sqrt{pi}) rho(r) sigma(r) (sigma/m_x) t_age = 1

    where  Gamma(r) := rho(r) sigma(r) (sigma/m_x) is the scattering rate
    with rho(r) the density profile, sigma(r) the velocity dispersion,
    and sigma/m_x the self-interaction cross-section per particle mass.

    Syntax:

        r1(potential,sigmamx=1.,tage=1.)

    where

        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        sigmamx: self-interaction cross-section per particle mass
            [cm^2/g] or [2.08889e-10 kpc^2/M_sun] (default=1.)
        tage: halo age [Gyr], somewhat arbitrary, e.g., lookback time to
            the formation epoch of the halo, where "formation" can be
            defined as the most recent major merger or the time of
            reaching half of the current mass.
        cff: value of coefficient, which we multiply by `r_s`. Default value is: `0.1`.
            More details in the `config.py`.
            (float)
    """
    Rres = cfg.find_Rres(potential, cff=cff)  # [kpc]
    a = Rres
    b = 2000.
    fa = Findr1(a, potential, sigmamx, tage)
    fb = Findr1(b, potential, sigmamx, tage)
    if fa * fb > 0.:
        r = Rres
    else:
        r = brentq(Findr1, a, b, args=(potential, sigmamx, tage),
                   xtol=0.001, rtol=1e-5, maxiter=1000)
    return r


def Findr1(r, potential, sigmamx, tage):
    """
    Auxiliary function for the function "r1", which returns the

        left-hand side  -  right-hand side

    of the equation

        4/sqrt{pi} rho(r) sigma(r) (sigma/m_x) t_age = 1
    """
    return cfg.FourOverRootPi * rho(potential, r) * sigma(potential, r) * (sigmamx * 2.08889e-10) * tage - 1.


def r1_new(potential, sigma0, sigmamx=1., tage=10., cff=0.001):
    """
    A variant version of the function "r1", where we use the central
    velocity dispersion sigma_0 in place of the CDM velocity dispersion
    profile sigma(r).

    The characteristic radius of a SIDM halo at which an average particle
    is scattered once during the age of the halo, defined as the solution
    to (e.g., Kaplinghat+16 eq.1)

        (4/sqrt{pi}) rho(r) sigma_0 (sigma/m_x) t_age = 1

    where  Gamma(r) := rho(r) sigma_0 (sigma/m_x) is the scattering rate
    with rho(r) the density profile, sigma_0 the central vel dispersion,
    and sigma/m_x the self-interaction cross-section per particle mass.

    Syntax:

        r1_new(potential,sigma0,sigmamx=1.,tage=10.)

    where

        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        sigma0: the central velocity dispersion [kpc/Gyr] (float)
        sigmamx: self-interaction cross-section per particle mass
            [cm^2/g] or [2.08889e-10 kpc^2/M_sun] (default=1.)
        tage: halo age [Gyr], somewhat arbitrary, e.g., lookback time to
            the formation epoch of the halo, where "formation" can be
            defined as the most recent major merger or the time of
            reaching half of the current mass. (default=1.)
        cff: value of coefficient, which we multiply by `r_s`. Default value is: `0.1`.
            More details in the `config.py`.
            (float)

    """
    Rres = cfg.find_Rres(potential, cff=cff)  # [kpc]
    a = Rres
    b = 2000.
    fa = Findr1_new(a, potential, sigma0, sigmamx, tage)
    fb = Findr1_new(b, potential, sigma0, sigmamx, tage)
    if fa * fb > 0.:
        r = Rres
    else:
        r = brentq(Findr1_new, a, b, args=(potential, sigma0, sigmamx, tage),
                   xtol=0.001, rtol=1e-5, maxiter=1000)
    return r


def Findr1_new(r, potential, sigma0, sigmamx, tage):
    """
    Auxiliary function for the function "r1", which returns the

        left-hand side  -  right-hand side

    of the equation

        (4/sqrt{pi}) rho(r) sigma_0 (sigma/m_x) t_age = 1
    """
    return cfg.FourOverRootPi * rho(potential, r) * sigma0 * (sigmamx * 2.08889e-10) * tage - 1.

# ######################################## RSIDM: SELF-INTERATCIONG CROSS_SECTION ##################### #
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

GEV_M2_TO_CM2 = 0.389379e-27          # cm^2 per GeV^-2
GEV_TO_G      = 1.78266192e-24        # g per GeV
GEV_M3_TO_CM2_PER_G = GEV_M2_TO_CM2 / GEV_TO_G  # cm^2/g per GeV^-3

# ---------- Maxwellian relative-speed PDF ----------
def maxwell_rel_pdf(v, nu):
    """
    Relative-speed Maxwellian:
    f(v) dv = (4π v^2 / (4π ν^2)^(3/2)) exp[-v^2/(4ν^2)] dv
    v, nu dimensionless (v/c).
    """
    return (4.0 * math.pi * v*v *
            math.exp(-v*v/(4.0*nu*nu))) / ((4.0*math.pi*nu*nu)**1.5)


# ---------- Resonant SIDM kernel ----------
def sigma_res_kernel(v, L, Gamma, vR, n=5):
    """
    Resonant SIDM kernel ∝ σ(v) without velocity weighting.
    """
    num = (v**(4*L + n)) * (Gamma**2)
    den = (v*v - vR*vR)**2 + 16.0 * (v**(2 + 4*L)) * (Gamma**2)
    return num / den

def sigma_m_eff_p1(
    nu_kms,
    sigma_m0,
    m_GeV,
    L,
    Gamma,
    vR_kms,
):
    """
    Effective <sigma/m>_{p=1}(nu) in [cm^2/g].

    nu_kms : 1D velocity dispersion [km/s]
    """
    # convert to dimensionless v/c
    nu = nu_kms / 3e5
    vR = vR_kms / 3e5

    # ⟨v⟩ for relative Maxwellian
    vmean = 4.0 * nu / math.sqrt(math.pi)

    def integrand(v):
        return v * sigma_res_kernel(v, L, Gamma, vR) * maxwell_rel_pdf(v, nu)

    # integrate (truncate tail safely)
    vmax = max(50.0 * nu, 5.0 * vR)
    I = quad(integrand, 0.0, vmax, epsrel=1e-8, limit=500)[0]

    pref = (256.0 * np.pi) * (GEV_M3_TO_CM2_PER_G) / (m_GeV ** 3)

    return sigma_m0 + pref * I / vmean

def build_sigma_m_eff_p1_interpolator(
    sigma_m0,
    m_GeV,
    L,
    Gamma,
    vR_kms,
    nu_kms_min=1.0,
    nu_kms_max=3000.0,
    N=300,
):
    nu_grid = np.logspace(
        np.log10(nu_kms_min),
        np.log10(nu_kms_max),
        N
    )

    vals = np.array([
        sigma_m_eff_p1(
            nu, sigma_m0, m_GeV, L, Gamma, vR_kms
        ) for nu in nu_grid
    ])

    interp = PchipInterpolator(
        np.log10(nu_grid),
        np.log10(vals),
        extrapolate=True
    )
    # print("nu_grid:", nu_grid)
    print("sigma_m_eff_p1", sigma_m_eff_p1(20, 0.008, 0.020, 0, 6e-12, 85.0))
    def sigma_m_eff_nu(nu_kpcGyr):
        nu_kms = nu_kpcGyr * cfg.kpcGyr_to_kms
        return 10.0**interp(np.log10(nu_kms))

    return sigma_m_eff_nu

# ---- PLotting function
def r1_vs_time_plot(
    halo,
    sigma_m_eff_nu,
    tmin=0.1, tmax=100.0, Nt=80,
    pick="outermost",
    K_n=1.0,
    cff=0.001,
    rmax=2000.0,
    ngrid=800
):
    """
    Compute r1(t) and make diagnostic plots:
      - r1_outer(t) and r1_inner(t)
      - number of roots vs time (branch structure)

    All times in Gyr, radii in kpc.
    """
    ts = np.logspace(np.log10(tmin), np.log10(tmax), Nt)

    r1_outer = np.full_like(ts, np.nan, dtype=float)
    r1_inner = np.full_like(ts, np.nan, dtype=float)
    nroots   = np.zeros_like(ts, dtype=int)

    for i, t in enumerate(ts):
        # all roots
        r1o, roots, dbg = find_all_r1_roots_rsidm(
            halo,
            tage=t,
            sigma_m_eff_nu=sigma_m_eff_nu,
            K_n=K_n,
            cff=cff,
            rmax=rmax,
            ngrid=ngrid,
            pick="outermost"
        )

        nroots[i] = len(roots)
        if len(roots) > 0:
            r1_outer[i] = roots[-1]
            r1_inner[i] = roots[0]
        else:
            # if no roots, your function returns rmin; we mark NaN for clarity
            r1_outer[i] = np.nan
            r1_inner[i] = np.nan

    rho_s = 6390000.0
    r_s = 9.33
    sigma_m = 70.0
    time_tilda = uni.convert_time_tilda(ts, rho_s, r_s, sigma_m)  # [Gyr]

    # ---- Plot r1(t) ----
    plt.figure(figsize=(7.0, 4.8))
    plt.loglog(time_tilda, r1_outer/9.33, marker="o", markersize=3, linewidth=1.2, label="r1")
    # plt.loglog(ts, r1_inner, marker="o", markersize=3, linewidth=1.2, label="r1 innermost", alpha=0.8)
    plt.xlabel(r"$t$ [Dimesionless]")
    plt.ylabel(r"$r_1$ [_s]")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot number of roots ----
    plt.figure(figsize=(7.0, 3.8))
    plt.semilogx(ts, nroots, marker="o", markersize=3, linewidth=1.2)
    plt.xlabel(r"$t_{\rm age}$ [Gyr]")
    plt.ylabel("number of r1 roots")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()

    return ts, r1_outer, r1_inner, nroots

def plot_F_r_for_times(
    halo,
    sigma_m_eff_nu,
    times_gyr=(0.3, 0.7, 1.0, 3.0, 10.0, 30.0),
    K_n=1.0,
    cff=0.001,
    rmax=2000.0,
    ngrid=800,
):
    """
    Plot F(r) for several halo ages to see how/when roots appear,
    and whether multiple branches exist.

    F(r) = (4/sqrt(pi)) rho(r) nu(r) <sigma/m>_eff(nu(r)) t_age - K_n
    """
    rmin = cfg.find_Rres(halo, cff=cff)
    if rmin <= 0.0:
        raise ValueError("rmin <= 0 from cfg.find_Rres; cannot use log grid.")

    rs = np.logspace(np.log10(rmin), np.log10(rmax), int(ngrid))

    plt.figure(figsize=(7.5, 5.0))
    for t in times_gyr:
        Fs = np.array([F_r1_rsidm(r, halo, t, K_n, sigma_m_eff_nu) for r in rs])
        plt.loglog(rs, Fs, marker=None, linewidth=1.4, label=rf"$t={t:g}$ Gyr")

        # plt.semilogx(rs, Fs, marker=None, linewidth=1.4, label=rf"$t={t:g}$ Gyr")

    # plt.axhline(0.0, color="k", linewidth=1.0)
    plt.ylim(1e-3, 1e4)
    plt.xlabel(r"$r$ [kpc]")
    plt.ylabel(r"$F(r)$")
    plt.title(r"RSIDM $r_1$ root function $F(r)$ at different $t_{\rm age}$")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_nu_r(
    halo,
    cff=0.001,
    rmax=2000.0,
    ngrid=600,
):
    """
    Plot the 1D velocity dispersion nu(r) in km/s to see which velocity range
    the halo probes across radii.
    """
    rmin = cfg.find_Rres(halo, cff=cff)
    if rmin <= 0.0:
        raise ValueError("rmin <= 0 from cfg.find_Rres; cannot use log grid.")

    rs = np.logspace(np.log10(rmin), np.log10(rmax), int(ngrid))
    nu_kms = np.array([sigma(halo, r) for r in rs]) * cfg.kpcGyr_to_kms

    plt.figure(figsize=(7.5, 4.6))
    plt.loglog(rs, nu_kms, linewidth=1.6)
    plt.xlabel(r"$r$ [kpc]")
    plt.ylabel(r"$\nu(r)$ [km/s]")
    plt.title(r"Halo 1D velocity dispersion profile $\nu(r)$")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()

# ######################################## FINDING R1: GENERAL ######################################## #
# ---------- REQUIRED: your wrappers already exist ----------
# from your file:
#   rho(potential, r)    -> [M_sun/kpc^3]
#   sigma(potential, r)  -> [kpc/Gyr]  (1D dispersion)
# and config:
#   cfg.FourOverRootPi = 4/sqrt(pi)
#   cfg.find_Rres(potential, cff) -> [kpc]

# ---------- Unit conversion you already use ----------
SIGMAMX_CGS_TO_KPC2_PER_MSUN = 2.08889e-10  # (cm^2/g) -> (kpc^2/M_sun)


def F_r1_rsidm(r, potential, tage, K_n, sigma_m_eff_nu):
    """
    Root function for RSIDM r1:
        F(r) = (4/sqrt(pi)) rho(r) nu(r) <sigma/m>_eff(nu(r)) t_age - K_n

    Inputs
    ------
    r : float [kpc]
    potential : NFWProfile or list of profiles
    tage : float [Gyr]
    K_n : float (usually 1.0)
    sigma_m_eff_nu : callable
        sigma_m_eff_nu(nu_kpcGyr) -> [cm^2/g]

    Returns
    -------
    float
        F(r) (dimensionless)
    """
    nu = sigma(potential, r)                    # [kpc/Gyr]
    sigmamx_cgs = sigma_m_eff_nu(nu)            # [cm^2/g]
    sigmamx = sigmamx_cgs * SIGMAMX_CGS_TO_KPC2_PER_MSUN  # [kpc^2/M_sun]

    print("sigma_m_eff_nu:", sigma_m_eff_nu(20)/cfg.kpcGyr_to_kms)
    # lhs = cfg.FourOverRootPi * rho(potential, r) * nu * sigmamx * tage
    lhs = nu * cfg.kpcGyr_to_kms
    return lhs


def find_all_r1_roots_rsidm(
    potential,
    tage,
    sigma_m_eff_nu,
    K_n=1.0,
    cff=0.001,
    rmax=2000.0,
    ngrid=800,
    pick="outermost",
    xtol=1e-3,
    rtol=1e-6,
):
    """
    Find r1 for RSIDM allowing multiple branches.

    Procedure
    ---------
    1) Build a log-spaced radius grid [rmin, rmax]
    2) Find all sign changes in F(r)
    3) Refine each bracket with brentq
    4) Select a root based on `pick` ("outermost" or "innermost")

    Returns
    -------
    r1_selected : float [kpc]
        Selected r1 (by rule). If no roots found, returns rmin.
    r1_roots : list of float [kpc]
        All roots found (sorted ascending).
    debug : dict
        Useful diagnostics: radii grid and F values.
    """
    rmin = cfg.find_Rres(potential, cff=cff)  # [kpc]
    if rmin <= 0.0:
        raise ValueError("rmin <= 0 from cfg.find_Rres; cannot use log grid.")

    # Log grid for robust bracketing (cheap)
    rs = np.logspace(np.log10(rmin), np.log10(rmax), int(ngrid))
    Fs = np.array([F_r1_rsidm(r, potential, tage, K_n, sigma_m_eff_nu) for r in rs])

    # Identify all sign changes (excluding NaNs)
    valid = np.isfinite(Fs)
    rs_v = rs[valid]
    Fs_v = Fs[valid]

    if len(rs_v) < 2:
        return rmin, [], {"rs": rs, "Fs": Fs}

    idx = np.where(Fs_v[:-1] * Fs_v[1:] < 0.0)[0]

    roots = []
    for i in idx:
        a, b = rs_v[i], rs_v[i + 1]
        try:
            root = brentq(
                F_r1_rsidm, a, b,
                args=(potential, tage, K_n, sigma_m_eff_nu),
                xtol=xtol, rtol=rtol, maxiter=1000
            )
            roots.append(root)
        except ValueError:
            # In rare cases F may be ill-behaved inside bracket due to numerical noise.
            # We skip the bracket; diagnostics allow you to inspect.
            pass

    roots = sorted(set([float(r) for r in roots]))  # unique + sorted

    if len(roots) == 0:
        # No crossing found: keep old behaviour (return rmin)
        return rmin, [], {"rs": rs, "Fs": Fs}

    if pick == "innermost":
        r1_selected = roots[0]
    elif pick == "outermost":
        r1_selected = roots[-1]
    else:
        raise ValueError("pick must be 'outermost' or 'innermost'.")

    debug = {"rs": rs, "Fs": Fs, "roots": roots, "rmin": rmin, "rmax": rmax}
    return r1_selected, roots, debug


# ------------------------------
# Example usage
# ------------------------------
# if __name__ == "__main__":
#     # 1) Build a halo (your class)
#     halo = NFWProfile(_M_vir=1e10, _con=12.0)
#
#     # 2) Provide an RSIDM effective sigma/m function (p=1) in [cm^2/g]
#     #    Here is a placeholder power-law just so the script runs.
#     #    Replace with your precomputed RSIDM interpolator sigma_m_eff_p1(nu).
#
#     sigma_m_eff_nu = build_sigma_m_eff_p1_interpolator(
#         sigma_m0=0.1,  # cm^2/g
#         m_GeV=0.02,
#         L=0,
#         Gamma=6e-12,
#         vR_kms=85.0
#     )
#
#     # 3) Find r1 (keep branch switching; choose outermost by default)
#     r1, roots, dbg = find_all_r1_roots_rsidm(
#         halo,
#         tage=100.0,
#         sigma_m_eff_nu=sigma_m_eff_nu,
#         pick="outermost"
#     )
#
#     print("r1 =", r1, "kpc")
#     print("roots =", roots)


# if __name__ == "__main__":
#     halo = NFWProfile(_M_vir=1e11, _con=12.09746609075212)
#
#     sigma_m_eff_nu = build_sigma_m_eff_p1_interpolator(
#         sigma_m0=0.008,
#         m_GeV=0.02,
#         L=0,
#         Gamma=6e-12,
#         vR_kms=85.0
#     )
#
#     ts, r1o, r1i, nroots = r1_vs_time_plot(
#         halo,
#         sigma_m_eff_nu,
#         tmin=0.1, tmax=10000.0, Nt=70,
#         cff=0.001, rmax=2000.0, ngrid=800
#     )
#     def diagnose_root_existence(halo, t, sigma_m_eff_nu, K_n=1.0, cff=0.001, rmax=2000.0):
#         rmin = cfg.find_Rres(halo, cff=cff)
#         Fmin = F_r1_rsidm(rmin, halo, t, K_n, sigma_m_eff_nu)
#         Fmax = F_r1_rsidm(rmax, halo, t, K_n, sigma_m_eff_nu)
#         print(f"t={t:8.3g} Gyr  rmin={rmin:.3e} kpc  F(rmin)={Fmin:+.3e}  F(rmax)={Fmax:+.3e}")
#
#     diagnose_root_existence(halo, 2e-1, sigma_m_eff_nu)

if __name__ == "__main__":
    halo = NFWProfile(_M_vir=1e10, _con=13.3)

    sigma_m_eff_nu = build_sigma_m_eff_p1_interpolator(
        sigma_m0=0.008,  # cm^2/g
        m_GeV=0.02,
        L=0,
        Gamma=6e-12,
        vR_kms=85.0
    )

    # Plot nu(r) first (to understand velocity regime)
    plot_nu_r(halo, cff=0.001, rmax=2000.0, ngrid=600)

    # Then plot F(r) at several ages (to see roots / branches)
    plot_F_r_for_times(
        halo,
        sigma_m_eff_nu,
        times_gyr=(0.2, 0.4, 0.6, 0.8, 1.0, 3.0, 10.0, 30.0),
        K_n=1.0,
        cff=0.001,
        rmax=2000000.0,
        ngrid=900,
    )
