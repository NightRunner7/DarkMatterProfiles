"""
Functions dealing with composite potential (i.e., potential list). Also, here we
will use `potential` and `halo` / `halos` interchangeably.
"""
from scipy.optimize import brentq  # fsolve
import numpy as np
# ------------ IMPORT FROM FILES ------------ #
import config as cfg

# ######################################## rho(r), nu(r), ... ######################################## #
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
