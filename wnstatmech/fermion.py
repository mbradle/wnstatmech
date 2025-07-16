"""This is the module that handles fermions."""

import math
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

    """

    def __init__(self, name, rest_mass_mev, multiplicity, charge):
        super().__init__(name, rest_mass_mev, multiplicity, charge)

        self.update_functions(
            "number density", self.default_number_density_function
        )
        self.update_functions("pressure", None)
        self.update_functions("energy density", None)
        self.update_functions("internal energy density", None)
        self.update_functions("entropy density", None)

        self.update_integrands(
            "number density", self.default_number_density_integrand
        )
        self.update_integrands("pressure", self.default_pressure_integrand)
        self.update_integrands(
            "energy density", self.default_energy_density_integrand
        )
        self.update_integrands(
            "entropy density", self.default_entropy_density_integrand
        )
        self.update_integrands(
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

        def n_part(y):
            if y > 0:
                return self._safe_exp(-y) / (1.0 + self._safe_exp(-y))
            return 1.0 / (1.0 + self._safe_exp(y))

        gamma = self.get_gamma(temperature)
        f = (
            math.sqrt(x**2 + 2 * x * gamma)
            * (x + gamma)
            * (n_part(x - alpha) - n_part(x + 2 * gamma + alpha))
        )
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
        nd_plus = ((x + gamma) ** 2) * math.sqrt(x**2 + 2 * x * gamma)
        part1 = 1 / (self._safe_exp(x - alpha) + 1)
        part2 = 1 / (self._safe_exp(x + 2 * gamma + alpha) + 1)
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

        def s_part(y):
            if y >= 0:
                return y * self._safe_exp(-y) / (
                    1.0 + self._safe_exp(-y)
                ) + math.log1p(self._safe_exp(-y))
            return (
                (y / (1.0 + self._safe_exp(y)))
                - y
                + math.log1p(self._safe_exp(y))
            )

        gamma = self.get_gamma(temperature)
        f = (
            math.sqrt(x**2 + 2 * x * gamma)
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
        nd_plus = (x * (x + gamma)) * math.sqrt(x**2 + 2 * x * gamma)
        part1 = 1 / (self._safe_exp(x - alpha) + 1)
        part2 = 1 / (self._safe_exp(x + 2 * gamma + alpha) + 1)
        f = nd_plus * (part1 + part2)
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
        """Routine to compute a thermodynamic quantity for the fermion.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity to compute.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            quantity.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass)
            divided by kT at which to compute the quantity.

        Returns:
            A :obj:`float` giving the quantity in cgs units.  The routine will first try
            to compute the quantity with a function, if set.  If the function is not
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

    def compute_temperature_derivative(self, quantity, temperature, alpha):
        """Routine to compute the temperature derivative of a thermodynamic quantity
        for the fermion.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity to compute.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            derivative.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass)
            divided by kT at which to compute the derivative.

        Returns:
            A :obj:`float` giving the temperature derivative of the quantity in cgs units.

        """
        assert (
            quantity in self.integrands
        ), "Integrand not specified for quantity."
        return self._compute_temperature_derivative(
            self.functions[quantity],
            self.integrands[quantity],
            temperature,
            alpha,
        )


def create_electron():
    """Convenience routine for creating an electron.

    Returns:
        An electron as a :obj:`wnstatmech.fermion.Fermion` object.

    """
    electron_mass = (
        gc.GSL_CONST_CGSM_MASS_ELECTRON
        * (gc.GSL_CONST_CGSM_SPEED_OF_LIGHT**2)
        / (gc.GSL_CONST_NUM_MEGA * gc.GSL_CONST_CGSM_ELECTRON_VOLT)
    )
    return Fermion("electron", electron_mass, 2, -1)
