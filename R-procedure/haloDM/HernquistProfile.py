import numpy as np
# ------------ IMPORT FROM FILES ------------ #
import config as cfg

class HernquistProfile(object):
    """
    Class that implements the Hernquist (1990) profile:

        rho(r) = M / (2 pi a^3) / [x (1+x)^3], x = r/a

    in a cylindrical frame (R,phi,z), where

        M: total mass
        a: scale radius

    Syntax:

        baryon = Hernquist(M,a)

    where

        M: baryon mass [M_sun] (float)
        a: scalelength [kpc] (float)

    Attributes:

        .Mb: baryon mass [M_sun]
        .Mh: the same as .Md, but for the purpose of keeping the notation
            for "host" mass consistent with the other profile classes
        .a: scalelength [kpc]
        .r0: the same as .a
        .rho0: characteristic density, M/(2 pi a^3) [M_sun/kpc^3]
        .rhalf: half-mass radius [kpc]

    Methods:

        .rho(R,z=0.): density [M_sun kpc^-3] at (R,z)
        .M(R,z=0.): mass [M_sun] within radius r=sqrt(R^2+z^2),
            defined as M(r) = r Vcirc(r,z=0)^2 / G
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at (R,z)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z)
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at (R,z=0), defined as
            sqrt(R d Phi(R,z=0.)/ d R)
        .sigma(R,z=0.): velocity dispersion [kpc/Gyr] at (R,z)

    HISTORY: Arthur Fangzhou Jiang (2020-09-09, Caltech)
    """

    def __init__(self, M, a):
        """
        Initialize Hernquist profile

        Syntax:

            baryon = Hernquist(M,a)

        where

            M: baryon mass [M_sun],
            a: scale radius [kpc]
        """
        # input attributes
        self.Mb = M
        self.Mh = self.Mb
        self.a = a
        self.r0 = a
        #
        # derived attributes
        self.rho0 = M / (cfg.TwoPi * a ** 3)
        self.rhalf = 2.414213562373095 * a
        #
        # supportive attributes repeatedly used by following methods
        self.GMb = cfg.const_G_starUnits * M

    def rho(self, R, z=0.):
        """
        Density [M_sun kpc^-3] at (R,z).

        Syntax:

            .rho(R,z)

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.a
        return self.rho0 / (x * (1. + x) ** 3)

    def Mass(self, R, z=0.):
        """
        Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2).

        Syntax:

            .M(R,z=0):

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        return self.Mb * r ** 2 / (r + self.a) ** 2

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
        return 3. / (cfg.FourPi * r ** 3.) * self.Mass(R, z)

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

    def Phi(self, R, z=0.):
        """
        Potential [(kpc/Gyr)^2] at (R,z).

        Syntax:

            .Phi(R,z=0.)

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        return -self.GMb / (r + self.a)

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
        pass

    def Vcirc(self, R, z=0.):
        """
        Circular velocity [kpc/Gyr] at (R,z=0.), defined as

            V_circ(R,z=0.) = sqrt(R d Phi(R,z=0.)/ d R)

        Syntax:

            .Vcirc(R,z=0.)

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.)

        Note that only z=0 is meaningful. Because circular velocity is
        the speed of a satellite on a circular orbit, and for a disk
        potential, a circular orbit is only possible at z=0.
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        return np.sqrt(cfg.const_G_starUnits * self.Mass(r) / r)

    def sigma(self, R, z=0.):
        """
        Velocity dispersion [kpc/Gyr] assuming isotropic velicity
        dispersion tensor ...

        Syntax:

            .sigma(R,z=0.)

        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.)

        Note that this is at the same time the R-direction and the
        z-direction velocity dispersion, as we implicitly assumed
        that the distribution function of the disk potential depends only
        on the isolating integrals E and L_z. If we further assume
        isotropy, then it is also the phi-direction velocity dispersion.
        (See CP96 eqs 11-17 for more.)
        """
        r = np.sqrt(R ** 2. + z ** 2.)
        x = r / self.a
        sigmasqr = self.GMb / (12. * self.a) * (12. * x * (1. + x) ** 3 * np.log(1. + 1. / x)
                                                - (x / (1. + x)) * (25. + 52. * x + 42. * x ** 2 + 12. * x ** 3))
        return np.sqrt(sigmasqr)

# --- publication 2206.12425
Mb = 10 ** 9  # [M_sun]
r_1_over_2 = 1.9  # [kpc]
r0 = r_1_over_2 / (1 + np.sqrt(2))  # [kpc]
print("r0:", r0, "kpc")

Hernquist_class = HernquistProfile(Mb, r0)
rho_0 = Hernquist_class.rho0

print("rho_0:", rho_0, "[M_sun * kpc^-3]")


