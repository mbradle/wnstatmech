"""This is the module that handles bosons."""

import math
import gslconsts.consts as gc
import wnstatmech.base as wbst


class Boson(wbst.Particle):
    def __init__(self, name, rest_mass_mev, multiplicity, charge):
        super().__init__(name, rest_mass_mev, multiplicity, charge)

        self.integrands = {
            "number density": self.number_density_integrand,
            "pressure": self.pressure_integrand,
            "energy density": self.energy_density_integrand,
            "entropy density": self.entropy_density_integrand,
            "internal energy density": self.internal_energy_density_integrand,
        }

    def number_density_integrand(self, x, temperature, alpha):
        gamma = self.get_gamma(temperature)
        denom = self._safe_expm1(x - alpha)
        if denom == 0:
            return 0.0
        f = math.sqrt(x**2 + 2 * x * gamma) * (x + gamma) / denom
        return f * self._prefactor(temperature, power=3)

    def pressure_integrand(self, x, temperature, alpha):
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
        gamma = self.get_gamma(temperature)
        denom = self._safe_expm1(x - alpha)
        if denom == 0:
            return 0.0
        nd_plus = ((x + gamma) ** 2) * math.sqrt(x**2 + 2 * x * gamma)
        f = nd_plus / denom
        return f * self._prefactor(temperature, power=4)

    def entropy_density_integrand(self, x, temperature, alpha):
        gamma = self.get_gamma(temperature)
        f = (
            math.sqrt(x**2 + 2 * x * gamma)
            * (x + gamma)
            * (
                math.log1p(-self._safe_exp(alpha - x))
                + (alpha - x) / (1.0 + self._safe_exp(x - alpha))
            )
        )
        return (
            -gc.GSL_CONST_CGSM_BOLTZMANN
            * self._prefactor(temperature, power=3)
            * f
        )

    def internal_energy_density_integrand(self, x, temperature, alpha):
        gamma = self.get_gamma(temperature)
        denom = self._safe_expm1(x - alpha)
        if denom == 0:
            return 0.0
        nd_plus = x * (x + gamma) * math.sqrt(x**2 + 2 * x * gamma)
        f = nd_plus / denom
        return f * self._prefactor(temperature, power=4)

    def compute_chemical_potential(self, temperature, number_density):
        return self._compute_chemical_potential(
            self.integrands["number density"],
            temperature,
            number_density,
        )

    def compute_quantity(self, quantity, temperature, alpha):
        return self._compute_quantity(
            self.integrands[quantity], temperature, alpha
        )

    def compute_temperature_derivative(self, quantity, temperature, alpha):
        return self._compute_temperature_derivative(
            self.integrands[quantity], temperature, alpha
        )

    def update_integrands(self, quantity, integrand_fn):
        """A method to update the integrands for the boson.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity.

            ``integrand_fn`` (:obj:`float`): The integrand corresponding to the \
            quanity.  The integrand function must take three arguments.  The first \
            is the scaled energy *x*, the second is *T*, the temperature  in Kelvin, \
            and the third is the *alpha*, the chemical potential (less the rest mass) \
            divided by kT.  Other data can be bound to the integrand function.
        """

        self.integrands[quantity] = integrand_fn


def create_photon():
    return Boson("photon", 0, 2, 0)
