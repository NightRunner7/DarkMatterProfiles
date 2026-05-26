import numpy as np
import math
from scipy.integrate import quad
from scipy.special import gammainc, gamma

import config as cfg
from NFWProfile import NFWProfile


class TruncatedNFWProfile(object):
    def __init__(self, _M_vir, _con, _r200=None, _log_rho_s=None, _r_d=None):
        """
        Truncated NFW profile.

        Two initialization modes:

        1. Standard NFW mode:
            TruncatedNFWProfile(M_vir, con)

            Then r200, r_s, rho_s are computed from M_vir and con.

        2. Explicit mode:
            TruncatedNFWProfile(M_vir, con, r200, log_rho_s)

            Then r200 and rho_s are taken directly from input.
            This is useful when reproducing benchmark/table values from a paper.

        Important convention:
            M_vir_input is the mass supplied by the user.
            M200_inner is the actual mass implied by the density profile inside r200.
        """

        # ------------------------------------------------------------------
        # Store user input
        # ------------------------------------------------------------------
        self.M_vir_input = _M_vir  # [M_sun], value supplied by user/table
        self.con = _con  # dimensionless concentration c = r200 / r_s

        # ------------------------------------------------------------------
        # Case 1: standard NFW initialization from M_vir and concentration
        # ------------------------------------------------------------------
        if _r200 is None and _log_rho_s is None:

            rho_crit = cfg.rho_c  # [M_sun / kpc^3]

            self.r200 = (
                                3.0 * self.M_vir_input
                                / (4.0 * math.pi * 200.0 * rho_crit)
                        ) ** (1.0 / 3.0)

            self.r_vir = self.r200
            self.r_s = self.r200 / self.con

            self.rho_s = (
                    self.M_vir_input
                    / (4.0 * math.pi * self.r_s ** 3 * self.f(self.con))
            )

            self.initialization_mode = "standard"

        # ------------------------------------------------------------------
        # Case 2: explicit initialization from r200 and log10(rho_s)
        # ------------------------------------------------------------------
        elif _r200 is not None and _log_rho_s is not None:

            self.r200 = _r200  # [kpc]
            self.r_vir = self.r200  # [kpc]

            self.r_s = self.r200 / self.con
            self.rho_s = 10.0 ** _log_rho_s  # [M_sun / kpc^3]

            self.initialization_mode = "explicit"

        # ------------------------------------------------------------------
        # Bad mixed input
        # ------------------------------------------------------------------
        else:
            raise ValueError(
                "You must either provide both _r200 and _log_rho_s, "
                "or provide neither of them."
            )

        # ------------------------------------------------------------------
        # Actual NFW mass inside r200 implied by rho_s and r_s
        # ------------------------------------------------------------------
        self.M200_inner = self.Mass_NFW_inner(self.r200)

        # Optional alias: use this only if you want M_vir to mean
        # "actual profile mass inside r200"
        self.M_vir = self.M200_inner

        # Relative mismatch between supplied mass and profile-implied mass
        self.mass_mismatch = (self.M200_inner - self.M_vir_input) / self.M_vir_input

        # ------------------------------------------------------------------
        # Local NFW density at r200
        # Avoid name rho_200 if you want to avoid confusion with 200*rho_crit.
        # ------------------------------------------------------------------
        self.rho_r200 = self.rho_s / (self.con * (1.0 + self.con) ** 2)

        # For compatibility with your earlier notation
        self.rho_200 = self.rho_r200

        # ------------------------------------------------------------------
        # Truncation scale
        # ------------------------------------------------------------------
        if _r_d is None:
            self.r_d = self.r200
        else:
            self.r_d = _r_d

        # ------------------------------------------------------------------
        # Exponential decay index from logarithmic slope continuity at r200
        # ------------------------------------------------------------------
        self.eps_d = (
                self.r200 / self.r_d
                - (1.0 + 3.0 * self.con) / (1.0 + self.con)
        )

        # ------------------------------------------------------------------
        # Characteristic NFW radius and velocity
        # ------------------------------------------------------------------
        self.rmax = self.r_s * 2.163
        self.Vmax = self.Vcirc(self.rmax)

        # ------------------------------------------------------------------
        # Warn if explicit parameters are inconsistent
        # ------------------------------------------------------------------
        if abs(self.mass_mismatch) > 1e-2:
            print(
                "Warning: supplied M_vir and profile parameters are not fully consistent."
            )
            print(f"Initialization mode = {self.initialization_mode}")
            print(f"M_vir input         = {self.M_vir_input:.6e} M_sun")
            print(f"M_NFW(<r200)        = {self.M200_inner:.6e} M_sun")
            print(f"relative mismatch   = {self.mass_mismatch:.3e}")


    # def __init__(self, _M_vir, _con, _r_d=None):
    #     """
    #     Gilman/Tran-style truncated NFW profile.
    #
    #     Inside r200:
    #         rho(r) = rho_s / [x (1+x)^2]
    #         x = r / r_s
    #
    #     Outside r200:
    #         rho(r) = rho(r200) * (r/r200)^eps_d * exp[-(r-r200)/r_d]
    #
    #     where:
    #         r200 = c * r_s
    #         r_d  = r200 by default
    #         eps_d = r200/r_d - (1 + 3c)/(1 + c)
    #
    #     Arguments:
    #         _M_vir : M200 halo mass [M_sun]
    #         _con   : concentration c = r200 / r_s
    #         _r_d   : exponential decay scale [kpc], optional.
    #                  If None, use r_d = r200.
    #     """
    #
    #     # First initialize ordinary NFW structure
    #     super().__init__(_M_vir, _con)
    #
    #     # input attributes
    #     self.M_vir = _M_vir  # [M_sun]
    #     self.M200_inner = self.M_vir
    #     self.con = _con  # [dimensionless]
    #
    #     # Virial radius
    #     rho_cry = cfg.rho_c  # critical density # [M_solar / kpc^3]
    #     self.r_vir = ((3 * self.M_vir)/(4 * math.pi * 200 * rho_cry))**(1/3)
    #     self.r200 = self.r_vir
    #
    #     # basic variables: r_s
    #     self.r_s = self.r_vir/self.con
    #
    #     # basic variables: rho_s
    #     coeff = 200 / 3 * self.con ** 3 / self.f(self.con)
    #     self.rho_s = (coeff * rho_cry)  # [M_sun / kpc^3]
    #     self.rho_200 = self.rho_s/(self.con * (1.0 + self.con) ** 2)
    #
    #     # Decay scale
    #     if _r_d is None:
    #         self.r_d = self.r200
    #     else:
    #         self.r_d = _r_d
    #
    #     # Exponential decay index from slope continuity
    #     self.eps_d = self.r200 / self.r_d - (1.0 + 3.0 * self.con) / (1.0 + self.con)
    #
    #     # Update rmax and Vmax using the total profile.
    #     # For NFW it remains close to 2.163 r_s, and this is inside r200
    #     # for usual c > 2.163.
    #     self.rmax = self.r_s * 2.163
    #     self.Vmax = self.Vcirc(self.rmax)


    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------
    def f(self, x):
        return np.log(1.0 + x) - x / (1.0 + x)

    def rho(self, _r):
        """
        Density [M_sun / kpc^3] at radius _r.

        Same syntax as NFWProfile.rho(_r).
        """

        r = np.asarray(_r, dtype=float)

        x = r / self.r_s

        rho_inner = self.rho_s / (x * (1.0 + x) ** 2)

        rho_outer = (
            self.rho_200
            * (r / self.r200) ** self.eps_d
            * np.exp(-(r - self.r200) / self.r_d)
        )

        rho = np.where(r <= self.r200, rho_inner, rho_outer)

        if np.isscalar(_r):
            return float(rho)
        return rho

    # ------------------------------------------------------------------
    # Mass
    # ------------------------------------------------------------------
    def Mass_NFW_inner(self, _r):
        """
        NFW enclosed mass [M_sun] for the inner profile.

        Valid especially for r <= r200, but mathematically can be evaluated
        for any r.
        """
        x = _r / self.r_s
        return 4.0 * math.pi * self.rho_s * self.r_s ** 3 * self.f(x)

    def _outer_mass_integral_numeric(self, r):
        """
        Numerical outer mass integral from r200 to r.
        Used as robust fallback.
        """
        if r <= self.r200:
            return 0.0

        integrand = lambda rr: 4.0 * math.pi * rr ** 2 * self.rho(rr)
        return quad(integrand, self.r200, r, epsabs=0.0, epsrel=1e-6)[0]

    def _outer_mass_integral_analytic(self, r):
        """
        Analytic mass contribution outside r200.

        Integral:
            4 pi rho_200 exp(r200/rd) r200^{-eps}
            int rr^{2+eps} exp(-rr/rd) drr

        This uses the lower incomplete gamma function.
        For unusual eps_d values this can be less stable, so the public
        Mass method can fall back to the numerical version if needed.
        """

        if r <= self.r200:
            return 0.0

        a = 3.0 + self.eps_d

        prefactor = (
            4.0
            * math.pi
            * self.rho_200
            * np.exp(self.r200 / self.r_d)
            * self.r200 ** (-self.eps_d)
            * self.r_d ** a
        )

        x1 = self.r200 / self.r_d
        x2 = r / self.r_d

        lower_gamma_diff = gamma(a) * (gammainc(a, x2) - gammainc(a, x1))

        return prefactor * lower_gamma_diff

    def Mass(self, _r, analytic_outer=True):
        """
        Enclosed mass [M_sun] up to radius _r.

        For r <= r200:
            ordinary NFW mass.

        For r > r200:
            NFW mass inside r200 + mass from the exponential tail.
        """

        r = np.asarray(_r, dtype=float)

        def mass_scalar(rr):
            if rr <= self.r200:
                return self.Mass_NFW_inner(rr)

            try:
                if analytic_outer:
                    outer = self._outer_mass_integral_analytic(rr)
                else:
                    outer = self._outer_mass_integral_numeric(rr)
            except Exception:
                outer = self._outer_mass_integral_numeric(rr)

            return self.M200_inner + outer

        if np.isscalar(_r):
            return float(mass_scalar(float(r)))

        return np.array([mass_scalar(float(rr)) for rr in r])
    # ------------------------------------------------------------------
    # Potential
    # ------------------------------------------------------------------

    def Phi(self, _r):
        """
        Gravitational potential [(kpc/Gyr)^2].

        For the truncated profile, the exact potential is not the same as
        the infinite NFW expression. This computes

            Phi(r) = -G [ M(<r)/r + integral_r^inf 4 pi r' rho(r') dr' ]

        with Phi(infinity) = 0.
        """

        G = cfg.const_G_starUnits

        r = np.asarray(_r, dtype=float)

        def phi_scalar(rr):
            mass_term = self.Mass(rr) / rr

            shell_integrand = lambda rp: 4.0 * math.pi * rp * self.rho(rp)

            # Integral to infinity converges because of exponential cutoff.
            shell_term = quad(shell_integrand, rr, np.inf, epsabs=0.0, epsrel=1e-6)[0]

            return -G * (mass_term + shell_term)

        if np.isscalar(_r):
            return float(phi_scalar(float(r)))

        return np.array([phi_scalar(float(rr)) for rr in r])

    # ------------------------------------------------------------------
    # Velocities: same names as in NFWProfile
    # ------------------------------------------------------------------

    def Vcirc(self, _r):
        """
        Circular velocity [kpc/Gyr].
        """
        G = cfg.const_G_starUnits
        return np.sqrt(G * self.Mass(_r) / _r)

    def Vavg(self, _r):
        """
        Averaged velocity [kpc/Gyr] for Maxwell distribution.
        """
        return 4.0 / np.sqrt(np.pi) * self.Vcirc(_r)

    def sigma(self, R):
        """
        Same approximate Zentner & Bullock fitting function as your NFW class.

        Important:
        This fit is technically calibrated for NFW. Since the profile is
        unchanged inside r200 and most relevant radii are usually inside r200,
        keeping this method is probably fine for your gravothermal initial
        tests.
        """
        r = np.sqrt(R ** 2.0)
        x = r / self.r_s
        return self.Vmax * 1.4393 * x ** 0.354 / (1.0 + 1.1756 * x ** 0.725)

    def sigma_accurate(self, _r):
        """
        Velocity dispersion [kpc/Gyr] from isotropic Jeans equation:

            sigma_r^2(r) = 1/rho(r) * integral_r^inf rho(s) G M(s)/s^2 ds

        This version works for the truncated profile.
        """

        G = cfg.const_G_starUnits
        r = np.asarray(_r, dtype=float)

        def sigma_scalar(rr):
            integrand = lambda s: self.rho(s) * G * self.Mass(s) / s ** 2
            I = quad(integrand, rr, np.inf, epsabs=0.0, epsrel=1e-6)[0]
            return math.sqrt(I / self.rho(rr))

        if np.isscalar(_r):
            return float(sigma_scalar(float(r)))

        return np.array([sigma_scalar(float(rr)) for rr in r])