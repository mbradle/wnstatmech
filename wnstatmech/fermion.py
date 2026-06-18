"""This is the module that handles fermions."""

import math
import numpy as np
import gslconsts.consts as gc
import gslconsts.math as gm
import wnstatmech.base as wbst

D_GAMMA_ALPHA = 1.0e-3


class Fermion(wbst.Particle):
    """A class for fermions.

    Args:
        ``name`` (:obj:`str`): The name of the fermion.

        ``rest_mass_mev`` (:obj:`float`): The rest mass energy of the fermion (in MeV).

        ``multiplicity`` (:obj:`int`):  The multiplicity of the internal degrees of
        freedom of the fermion (typically 2 times the spin plus one).

        ``charge`` (:obj:`int`):  The charge of the fermion.

        ``integration_epsabs`` (:obj:`float`): Absolute integration tolerance.

        ``integration_epsrel`` (:obj:`float`): Relative integration tolerance.

    """

    def __init__(
        self,
        name,
        rest_mass_mev,
        multiplicity,
        charge,
        integration_epsabs=wbst.DEFAULT_INTEGRATION_EPSABS,
        integration_epsrel=wbst.DEFAULT_INTEGRATION_EPSREL,
    ):
        super().__init__(
            name,
            rest_mass_mev,
            multiplicity,
            charge,
            integration_epsabs=integration_epsabs,
            integration_epsrel=integration_epsrel,
        )

        self.update_function(
            "number density", self.default_number_density_function
        )
        self.update_function("pressure", None)
        self.update_function("energy density", None)
        self.update_function("internal energy density", None)
        self.update_function("entropy density", None)

        self.update_integrand(
            "number density", self.default_number_density_integrand
        )
        self.update_integrand("pressure", self.default_pressure_integrand)
        self.update_integrand(
            "energy density", self.default_energy_density_integrand
        )
        self.update_integrand(
            "entropy density", self.default_entropy_density_integrand
        )
        self.update_integrand(
            "internal energy density",
            self.default_internal_energy_density_integrand,
        )

    def default_number_density_function(self, temperature, alpha):
        """The default number density function to treat case of zero-rest-mass fermion.

        Args:
            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute
            the function.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the net number density
            (fermions minus anti-fermions) in cgs units for the given input
            for the case when the rest mass is zero.  When it is not zero,
            the routine returns None so that other routines will compute the
            number density by integration.

        """
        if self.get_rest_mass_cgs() == 0:
            return self._prefactor(temperature, power=3) * (
                gm.M_PI**3 * alpha / 3.0 + alpha**3 / 3.0
            )
        return None

    def default_number_density_integrand(self, x, temperature, alpha):
        """The default number density integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute
            the integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the net number density integrand
            (fermions minus anti-fermions) in cgs units for the given input.

        """

        if np.ndim(temperature) == 0 and np.ndim(alpha) == 0:

            def n_part_scalar(y):
                if y > 0:
                    return self._safe_exp(-y) / (1.0 + self._safe_exp(-y))
                return 1.0 / (1.0 + self._safe_exp(y))

            gamma = self.get_gamma(temperature)
            f = math.sqrt(x**2 + 2 * x * gamma) * (x + gamma)

            if abs(alpha + gamma) < 1.0e-3:
                f *= (
                    math.expm1(2.0 * (alpha + gamma))
                    * n_part_scalar(alpha - x)
                    * n_part_scalar(x + 2 * gamma + alpha)
                )
            else:
                f *= n_part_scalar(x - alpha) - n_part_scalar(
                    x + 2 * gamma + alpha
                )

            return f * self._prefactor(temperature, power=3)

        def n_part(y):
            return np.exp(-np.logaddexp(0.0, y))

        gamma = self.get_gamma(temperature)
        f = np.sqrt(x**2 + 2 * x * gamma) * (x + gamma)

        delta = alpha + gamma
        regular_term = n_part(x - alpha) - n_part(x + 2 * gamma + alpha)
        near = np.abs(delta) < 1.0e-3
        term = np.array(regular_term, copy=True)
        if np.any(near):
            term[near] = (
                np.expm1(2.0 * delta[near])
                * n_part(alpha[near] - x)
                * n_part(x + 2 * gamma[near] + alpha[near])
            )
        f *= term

        return f * self._prefactor(temperature, power=3)

    def default_pressure_integrand(self, x, temperature, alpha):
        """The default pressure integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the pressure integrand for the fermions
            (plus anti-fermions) in cgs units for the given input.

        """
        gamma = self.get_gamma(temperature)

        if np.ndim(temperature) == 0 and np.ndim(alpha) == 0:
            if alpha - x <= 0:
                part1 = math.log1p(self._safe_exp(alpha - x))
            else:
                part1 = alpha - x + math.log1p(self._safe_exp(x - alpha))
            if x + 2 * gamma + alpha <= 0:
                part2 = (
                    -x
                    - 2 * gamma
                    - alpha
                    + math.log1p(self._safe_exp(x + 2 * gamma + alpha))
                )
            else:
                part2 = math.log1p(self._safe_exp(-x - 2 * gamma - alpha))
            f = math.sqrt(x**2 + 2 * x * gamma) * (x + gamma) * (part1 + part2)
            return f * self._prefactor(temperature, power=4)

        part1 = np.logaddexp(0.0, alpha - x)
        part2 = np.logaddexp(0.0, -x - 2 * gamma - alpha)
        f = np.sqrt(x**2 + 2 * x * gamma) * (x + gamma) * (part1 + part2)
        return f * self._prefactor(temperature, power=4)

    def default_energy_density_integrand(self, x, temperature, alpha):
        """The default energy density integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the energy density integrand for the fermions
            (plus anti-fermions) in cgs units for the given input.

        """
        gamma = self.get_gamma(temperature)

        if np.ndim(temperature) == 0 and np.ndim(alpha) == 0:
            nd_plus = ((x + gamma) ** 2) * math.sqrt(x**2 + 2 * x * gamma)
            part1 = 1 / (self._safe_exp(x - alpha) + 1)
            part2 = 1 / (self._safe_exp(x + 2 * gamma + alpha) + 1)
            f = nd_plus * (part1 + part2)
            return f * self._prefactor(temperature, power=4)

        nd_plus = ((x + gamma) ** 2) * np.sqrt(x**2 + 2 * x * gamma)
        part1 = np.exp(-np.logaddexp(0.0, x - alpha))
        part2 = np.exp(-np.logaddexp(0.0, x + 2 * gamma + alpha))
        f = nd_plus * (part1 + part2)
        return f * self._prefactor(temperature, power=4)

    def default_entropy_density_integrand(self, x, temperature, alpha):
        """The default entropy density integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the entropy density integrand for the fermions
            (plus anti-fermions) in cgs units for the given input.

        """

        gamma = self.get_gamma(temperature)

        if np.ndim(temperature) == 0 and np.ndim(alpha) == 0:

            def s_part_scalar(y):
                if y >= 0:
                    return y * self._safe_exp(-y) / (
                        1.0 + self._safe_exp(-y)
                    ) + math.log1p(self._safe_exp(-y))
                return (
                    (y / (1.0 + self._safe_exp(y)))
                    - y
                    + math.log1p(self._safe_exp(y))
                )

            f = (
                math.sqrt(x**2 + 2 * x * gamma)
                * (x + gamma)
                * (
                    s_part_scalar(x - alpha)
                    + s_part_scalar(x + 2 * gamma + alpha)
                )
            )
            return (
                gc.GSL_CONST_CGSM_BOLTZMANN
                * self._prefactor(temperature, power=3)
                * f
            )

        def s_part(y):
            return np.logaddexp(0.0, -y) + y * np.exp(-np.logaddexp(0.0, y))

        f = (
            np.sqrt(x**2 + 2 * x * gamma)
            * (x + gamma)
            * (s_part(x - alpha) + s_part(x + 2 * gamma + alpha))
        )
        return (
            gc.GSL_CONST_CGSM_BOLTZMANN
            * self._prefactor(temperature, power=3)
            * f
        )

    def default_internal_energy_density_integrand(self, x, temperature, alpha):
        """The default internal energy density integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the internal energy density integrand for the fermions
            (plus anti-fermions) in cgs units for the given input.

        """
        gamma = self.get_gamma(temperature)

        if np.ndim(temperature) == 0 and np.ndim(alpha) == 0:
            nd_plus = (x * (x + gamma)) * math.sqrt(x**2 + 2 * x * gamma)
            part1 = 1 / (self._safe_exp(x - alpha) + 1)
            part2 = 1 / (self._safe_exp(x + 2 * gamma + alpha) + 1)
            f = nd_plus * (part1 + part2)
            return f * self._prefactor(temperature, power=4)

        nd_plus = (x * (x + gamma)) * np.sqrt(x**2 + 2 * x * gamma)
        part1 = np.exp(-np.logaddexp(0.0, x - alpha))
        part2 = np.exp(-np.logaddexp(0.0, x + 2 * gamma + alpha))
        f = nd_plus * (part1 + part2)
        return f * self._prefactor(temperature, power=4)

    def compute_chemical_potential(self, temperature, number_density):
        """Routine to compute the chemical potential (less the rest mass) divided by kT.

        Args:
            ``temperature`` (:obj:`float` or array-like): The temperature
            (in K) at which to compute the chemical potential.

            ``number_density`` (:obj:`float` or array-like):  The number
            density (in per cc) at which to compute the chemical potential.

        Returns:
            A :obj:`float` or :obj:`numpy.ndarray` giving the chemical
            potential (less the rest mass) divided by kT.

        """
        return self._compute_chemical_potential(
            self.functions["number density"],
            self.integrands["number density"],
            temperature,
            number_density,
        )

    def compute_quantity(self, quantity, temperature, alpha):
        """Routine to compute a thermodynamic quantity for the fermion.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity to compute.

            ``temperature`` (:obj:`float` or array-like): The temperature
            (in K) at which to compute the quantity.

            ``alpha`` (:obj:`float` or array-like):  The chemical potential
            (less the rest mass) divided by kT at which to compute the
            quantity.

        Returns:
            A :obj:`float` or :obj:`numpy.ndarray` giving the quantity in cgs
            units.  The routine will first try to compute the quantity with a
            function, if set.  If the function is not
            set for the quantity, or if the conditions in the function are not met by
            the input, the routine will integrate the associated integrand.

        """
        assert (
            quantity in self.integrands
        ), "Integrand not specified for quantity."
        return self._compute_quantity(
            self.functions[quantity],
            self.integrands[quantity],
            temperature,
            alpha,
        )

    def compute_temperature_derivative(
        self, quantity, temperature, number_density
    ):
        """Routine to compute the temperature derivative of a thermodynamic quantity
        for the fermion at fixed number density.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity to compute.

            ``temperature`` (:obj:`float` or array-like): The temperature
            (in K) at which to compute the derivative.

            ``number_density`` (:obj:`float` or array-like):  The fixed
            number density at which to compute the derivative.

        Returns:
            A :obj:`float` or :obj:`numpy.ndarray` giving the temperature
            derivative of the quantity in cgs units.

        """
        assert (
            quantity in self.integrands
        ), "Integrand not specified for quantity."
        return self._compute_temperature_derivative(
            (
                self.functions[quantity],
                self.integrands[quantity],
                self.functions["number density"],
                self.integrands["number density"],
            ),
            temperature,
            number_density,
        )


def create_electron(
    integration_epsabs=wbst.DEFAULT_INTEGRATION_EPSABS,
    integration_epsrel=wbst.DEFAULT_INTEGRATION_EPSREL,
):
    """Convenience routine for creating an electron.

    Returns:
        An electron as a :obj:`wnstatmech.fermion.Fermion` object.

    """
    electron_mass = (
        gc.GSL_CONST_CGSM_MASS_ELECTRON
        * (gc.GSL_CONST_CGSM_SPEED_OF_LIGHT**2)
        / (gc.GSL_CONST_NUM_MEGA * gc.GSL_CONST_CGSM_ELECTRON_VOLT)
    )
    return Fermion(
        "electron",
        electron_mass,
        2,
        -1,
        integration_epsabs=integration_epsabs,
        integration_epsrel=integration_epsrel,
    )
