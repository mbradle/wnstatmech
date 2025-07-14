"""This is the module that handles fermions."""

import math
import gslconsts.consts as gc
import wnstatmech.base as wbst

D_GAMMA_ALPHA = 1.0e-3


class Fermion(wbst.Particle):
    """A class for fermions.

    Args:
        ``name`` (:obj:`str`): The name of the fermion.

        ``rest_mass_mev`` (:obj:`float`): The rest mass energy of the fermion (in MeV).

        ``multiplicity`` (:obj:`int`):  The multiplicity of the internal degrees of
        freedom of the fermion (2 times the spin plus one).

        ``charge`` (:obj:`int`):  The charge of the fermion.

    """

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
        """A class for fermions.

        Args:
            ``x`` (:obj:`float`): The argument of the integrand.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
              integrand.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        """
        gamma = self.get_gamma(temperature)
        part1 = 1 / (self._safe_exp(x - alpha) + 1)
        part2 = 1 / (self._safe_exp(x + 2 * gamma + alpha) + 1)
        if abs(gamma + alpha) < D_GAMMA_ALPHA:
            f = (
                math.sqrt(x**2 + 2 * x * gamma)
                * (x + gamma)
                * self._safe_expm1(2 * (alpha + gamma))
                * part1
                * part2
            )
        else:
            f = math.sqrt(x**2 + 2 * x * gamma) * (x + gamma) * (part1 - part2)
        return f * self._prefactor(temperature, power=3)

    def pressure_integrand(self, x, temperature, alpha):
        gamma = self.get_gamma(temperature)
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

    def energy_density_integrand(self, x, temperature, alpha):
        gamma = self.get_gamma(temperature)
        nd_plus = math.sqrt(x**2 + 2 * x * gamma) * (x + gamma)
        part1 = 1 / (self._safe_exp(x - alpha) + 1)
        part2 = 1 / (self._safe_exp(x + 2 * gamma + alpha) + 1)
        f = nd_plus * (part1 + part2)
        return f * self._prefactor(temperature, power=4)

    def entropy_density_integrand(self, x, temperature, alpha):
        gamma = self.get_gamma(temperature)
        a = alpha - x
        b = -x - 2 * gamma - alpha
        if a > 0:
            part1 = a / (self._safe_exp(a) + 1) + math.log1p(
                self._safe_exp(-a)
            )
        else:
            part1 = a / (self._safe_exp(a) + 1) + math.log1p(self._safe_exp(a))
        if b > 0:
            part2 = b / (self._safe_exp(b) + 1) + math.log1p(
                self._safe_exp(-b)
            )
        else:
            part2 = b / (self._safe_exp(b) + 1) + math.log1p(self._safe_exp(b))
        f = math.sqrt(x**2 + 2 * x * gamma) * (x + gamma) * (part1 - part2)
        return f * self._prefactor(temperature, power=3)

    def internal_energy_density_integrand(self, x, temperature, alpha):
        return self.energy_density_integrand(
            x, temperature, alpha
        ) - self.get_rest_mass_cgs() * self.number_density_integrand(
            x, temperature, alpha
        )

    def compute_chemical_potential(self, temperature, number_density):
        return self._compute_chemical_potential(
            self.integrands["number density"], temperature, number_density
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
        """A method to update the integrands for the fermion.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity.

            ``integrand_fn`` (:obj:`float`): The integrand corresponding to the \
            quanity.  The integrand function must take three arguments.  The first \
            is the scaled energy *x*, the second is *T*, the temperature  in Kelvin, \
            and the third is the *alpha*, the chemical potential (less the rest mass) \
            divided by kT.  Other data can be bound to the integrand function.
        """

        self.integrands[quantity] = integrand_fn


def create_electron():
    electron_mass = (
        gc.GSL_CONST_CGSM_MASS_ELECTRON
        * (gc.GSL_CONST_CGSM_SPEED_OF_LIGHT**2)
        / (gc.GSL_CONST_NUM_MEGA * gc.GSL_CONST_CGSM_ELECTRON_VOLT)
    )
    return Fermion("electron", electron_mass, 2, -1)
