"""This is the module that handles bosons."""

import math
import gslconsts.consts as gc
import wnstatmech.base as wbst


class Boson(wbst.Particle):
    """A class for boson.

    Args:
        ``name`` (:obj:`str`): The name of the boson.

        ``rest_mass_mev`` (:obj:`float`): The rest mass energy of the boson (in MeV).

        ``multiplicity`` (:obj:`int`):  The multiplicity of the internal degrees of
        freedom of the boson (typically 2 times the spin plus one).

        ``charge`` (:obj:`int`):  The charge of the boson.

    """

    def __init__(self, name, rest_mass_mev, multiplicity, charge):
        super().__init__(name, rest_mass_mev, multiplicity, charge)

        self.update_functions("number density", None)
        self.update_functions("pressure", None)
        self.update_functions("energy density", None)
        self.update_functions("internal energy density", None)
        self.update_functions("entropy density", None)

        self.update_integrands("number density", self.number_density_integrand)
        self.update_integrands("pressure", self.pressure_integrand)
        self.update_integrands("energy density", self.energy_density_integrand)
        self.update_integrands(
            "entropy density", self.entropy_density_integrand
        )
        self.update_integrands(
            "internal energy density", self.internal_energy_density_integrand
        )

    def number_density_integrand(self, x, temperature, alpha):
        """The default number density integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute
            the integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the number density integrand in cgs units for the given input.

        """

        gamma = self.get_gamma(temperature)
        denom = self._safe_expm1(x - alpha)
        if denom == 0:
            return 0.0
        f = math.sqrt(x**2 + 2 * x * gamma) * (x + gamma) / denom
        return f * self._prefactor(temperature, power=3)

    def pressure_integrand(self, x, temperature, alpha):
        """The default pressure integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the pressure integrand for the boson in cgs units
            for the given input.

        """

        gamma = self.get_gamma(temperature)
        try:
            f = (
                math.sqrt(x**2 + 2 * x * gamma)
                * (x + gamma)
                * math.log1p(-self._safe_exp(alpha - x))
            )
        except ValueError:
            f = 0.0
        return -f * self._prefactor(temperature, power=4)

    def energy_density_integrand(self, x, temperature, alpha):
        """The default energy density integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the energy density integrand for the boson in cgs units
            for the given input.
        """

        gamma = self.get_gamma(temperature)
        denom = self._safe_expm1(x - alpha)
        if denom == 0:
            return 0.0
        nd_plus = ((x + gamma) ** 2) * math.sqrt(x**2 + 2 * x * gamma)
        f = nd_plus / denom
        return f * self._prefactor(temperature, power=4)

    def entropy_density_integrand(self, x, temperature, alpha):
        """The default entropy density integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the entropy density integrand for the boson in cgs units
            for the given input.

        """
        gamma = self.get_gamma(temperature)
        f = (
            math.sqrt(x**2 + 2 * x * gamma)
            * (x + gamma)
            * (
                math.log1p(-self._safe_exp(alpha - x))
                + (alpha - x) / (self._safe_expm1(x - alpha))
            )
        )
        return (
            -gc.GSL_CONST_CGSM_BOLTZMANN
            * self._prefactor(temperature, power=3)
            * f
        )

    def internal_energy_density_integrand(self, x, temperature, alpha):
        """The default internal energy density integrand.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the internal energy density integrand for the boson
            in cgs units for the given input.

        """

        gamma = self.get_gamma(temperature)
        denom = self._safe_expm1(x - alpha)
        if denom == 0:
            return 0.0
        nd_plus = x * (x + gamma) * math.sqrt(x**2 + 2 * x * gamma)
        f = nd_plus / denom
        return f * self._prefactor(temperature, power=4)

    def compute_chemical_potential(self, temperature, number_density):
        """Routine to compute the chemical potential (less the rest mass) divided by kT.

        Args:
            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            chemical potential.

            ``number_density`` (:obj:`float`):  The number density (in per cc) at which to
            compute the chemical potential.

        Returns:
            A :obj:`float` giving the chemical potential (less the rest mass) divided
            by kT.

        """

        return self._compute_chemical_potential(
            self.functions["number density"],
            self.integrands["number density"],
            temperature,
            number_density,
        )

    def compute_quantity(self, quantity, temperature, alpha):
        """Routine to compute a thermodynamic quantity for the boson.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity to compute.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            quantity.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass)
            divided by kT at which to compute the quantity.

        Returns:
            A :obj:`float` giving the quantity in cgs units.

        """

        return self._compute_quantity(
            self.functions[quantity],
            self.integrands[quantity],
            temperature,
            alpha,
        )

    def compute_temperature_derivative(self, quantity, temperature, alpha):
        """Routine to compute the temperature derivative of a thermodynamic quantity
        for the boson.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity to compute.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            derivative.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass)
            divided by kT at which to compute the derivative.

        Returns:
            A :obj:`float` giving the temperature derivative of the quantity in cgs units.

        """

        return self._compute_temperature_derivative(
            self.functions[quantity],
            self.integrands[quantity],
            temperature,
            alpha,
        )


def create_photon():
    """Convenience routine for creating a photon.

    Returns:
        A photon as a :obj:`wnstatmech.boson.Boson` object.

    """
    return Boson("photon", 0, 2, 0)
