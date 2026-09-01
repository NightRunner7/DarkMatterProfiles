import numpy as np
import math
from scipy.integrate import quad
from scipy.special import gammainc, gamma

import config as cfg
from NFWProfile import NFWProfile


class TruncatedNFWProfile(object):
    def __init__(self, _M_vir, _con, _r200=None, _r_s=None, _log_rho_s=None, _r_d=None, strict_consistency=False):
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

            self.r200 = (3.0 * self.M_vir_input / (4.0 * math.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
            self.r_vir = self.r200
            self.r_s = self.r200 / self.con
            self.rho_s = (self.M_vir_input / (4.0 * math.pi * self.r_s ** 3 * self.f(self.con)))

            self.initialization_mode = "standard"

        # ------------------------------------------------------------------
        # Mode 2: fully explicit benchmark parameters
        # ------------------------------------------------------------------
        elif (
            _r200 is not None
            and _r_s is not None
            and _log_rho_s is not None
        ):

            # Keep every supplied benchmark value exactly as given
            self.r200 = float(_r200)
            self.r_vir = self.r200
            self.r_s = float(_r_s)
            self.rho_s = 10.0 ** float(_log_rho_s)

            self.initialization_mode = "fully_explicit"

        else:
            raise ValueError(
                "Use either:\n"
                "  1. only _M_vir and _con for a self-consistent halo, or\n"
                "  2. _M_vir, _con, _r200, _r_s and _log_rho_s "
                "for a fully explicit benchmark halo."
            )

        # ------------------------------------------------------------------
        # Case 2: explicit initialization from r200 and log10(rho_s)
        # ------------------------------------------------------------------
        # elif _r200 is not None and _log_rho_s is not None:
        #
        #     self.r200 = _r200  # [kpc]
        #     self.r_vir = self.r200  # [kpc]
        #
        #     self.r_s = self.r200 / self.con
        #     self.rho_s = 10.0 ** _log_rho_s  # [M_sun / kpc^3]
        #
        #     self.initialization_mode = "explicit"

        # ------------------------------------------------------------------
        # Bad mixed input
        # ------------------------------------------------------------------
        # else:
        #     raise ValueError(
        #         "You must either provide both _r200 and _log_rho_s, "
        #         "or provide neither of them."
        #     )

        # ------------------------------------------------------------------
        # Diagnostics: mass and concentration implied by the actual profile
        # ------------------------------------------------------------------
        self.M200_inner = self.Mass_NFW_inner(self.r200)

        # Preserve the nominal mass supplied by the user
        self.M_vir = self.M_vir_input

        # Relative difference between nominal mass and mass implied by
        # rho_s, r_s and r200
        self.mass_mismatch = (self.M200_inner - self.M_vir_input) / self.M_vir_input

        # The actual dimensionless radius corresponding to r200
        self.x200 = self.r200 / self.r_s

        # Concentration implied by r200 and r_s
        self.con_implied = self.x200

        # Difference between supplied concentration and r200 / r_s
        self.con_mismatch = (self.con_implied - self.con) / self.con

        # ------------------------------------------------------------------
        # Local NFW density at r200
        # ------------------------------------------------------------------
        # Use x200 = r200 / r_s rather than the supplied concentration.
        # This remains correct even in fully explicit mode, where c and
        # r200 / r_s may differ because all tabulated values are enforced.
        self.rho_r200 = self.rho_s / (self.x200 * (1.0 + self.x200) ** 2)
        # self.rho_r200 = self.rho_s / (self.con * (1.0 + self.con) ** 2)

        # Compatibility alias
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
        self.eps_d = (self.r200 / self.r_d - (1.0 + 3.0 * self.con) / (1.0 + self.con))

        # ------------------------------------------------------------------
        # Characteristic NFW radius and velocity
        # ------------------------------------------------------------------
        self.rmax = self.r_s * 2.16258
        self.Vmax = self.Vcirc(self.rmax)

        # ------------------------------------------------------------------
        # Consistency diagnostics
        # ------------------------------------------------------------------
        mass_tolerance = 1e-2
        concentration_tolerance = 1e-2

        inconsistent = (
            abs(self.mass_mismatch) > mass_tolerance
            or abs(self.con_mismatch) > concentration_tolerance
        )

        if inconsistent:
            message = (
                "\nWarning: supplied halo parameters are not fully "
                "NFW-consistent.\n"
                f"Initialization mode    = {self.initialization_mode}\n"
                f"M200 input             = {self.M_vir_input:.8e} M_sun\n"
                f"M200 from profile      = {self.M200_inner:.8e} M_sun\n"
                f"mass mismatch          = {self.mass_mismatch:+.4e}\n"
                f"c input                = {self.con:.8f}\n"
                f"r200 / r_s             = {self.con_implied:.8f}\n"
                f"concentration mismatch = {self.con_mismatch:+.4e}\n"
                f"r200                    = {self.r200:.8f} kpc\n"
                f"r_s                     = {self.r_s:.8f} kpc\n"
                f"log10(rho_s)            = {math.log10(self.rho_s):.8f}"
            )

            if strict_consistency:
                raise ValueError(message)

            print(message)

    # ------------------------------------------------------------------
    # tau linear
    # ------------------------------------------------------------------
    def tau(self, beta, sigma_eff):
        r_s = self.r_s
        rho_s = self.rho_s
        sigma_m_SI = sigma_eff * 1e-4 * 1e3  # Convert from [cm^2/g] to [m^2/kg]
        sigma_m_SU = sigma_m_SI * cfg.kpc_SI ** -2 * cfg.M_solar_SI  # Convert to [kpc^2 * M_sun^-1]
        G_SU = cfg.const_G_starUnits
        time_c = (150 / beta) * (1 / (r_s * rho_s) * 1 / sigma_m_SU) * (4 * np.pi * G_SU * rho_s) ** (-1 / 2)
        # print("====================================================================================")
        # print("==================== TAU IN CLASS ==================================================")
        # print("con:", self.con)
        # print("sigma_eff:", sigma_eff)
        # print("beta:", beta)
        # print("r_s:", r_s)
        # print("log_rho_s:", np.log10(rho_s))
        # print("sigma_m_SU:", sigma_m_SU)
        # print("G_SU:", G_SU)
        # print("time_c:", time_c)
        return time_c

        # sigma_m_SI = sigma_eff * 1e-4 * 1e3  # Convert from [cm^2/g] to [m^2/kg]
        # sigma_m_SU = sigma_m_SI * cfg.kpc_SI ** -2 * cfg.M_solar_SI  # Convert to [kpc^2 * M_sun^-1]
        # G_SU = cfg.const_G_starUnits
        # nu_star = np.sqrt(G_SU * self.rho_s * self.r_s**2)
        # tau = (150 / beta) * (1 / sigma_m_SU) * (1 / self.rho_s) * (1 / (4 * np.pi * nu_star**2))**(1/2)  # [Gyr]
        # return tau


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