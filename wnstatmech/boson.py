"""This is the module that handles bosons."""

import math
from scipy.special import zeta
import gslconsts.consts as gc
import gslconsts.math as gm
import wnstatmech.base as wbst

a_rad = (
    4
    * gc.GSL_CONST_CGSM_STEFAN_BOLTZMANN_CONSTANT
    / gc.GSL_CONST_CGSM_SPEED_OF_LIGHT
)

a_num = (
    2
    * zeta(3)
    * gc.GSL_CONST_CGSM_BOLTZMANN**3
    / (
        gm.M_PI**2
        * (
            gc.GSL_CONST_CGSM_PLANCKS_CONSTANT_HBAR
            * gc.GSL_CONST_CGSM_SPEED_OF_LIGHT
        )
        ** 3
    )
)


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

        self.update_functions(
            "number density", self.default_number_density_function
        )
        self.update_functions("pressure", self.default_pressure_function)
        self.update_functions(
            "energy density", self.default_energy_density_function
        )
        self.update_functions(
            "internal energy density",
            self.default_internal_energy_density_function,
        )
        self.update_functions(
            "entropy density", self.default_entropy_density_function
        )

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
        """The default number density function to treat case of zero-rest-mass boson.

        Args:
            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute
            the function.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the boson number density in cgs units for the given input
            for the case when the rest mass and chemical potential are zero.  When they are
            not zero, the routine returns None so that other routines will compute the
            function by integration.

        """
        if self.get_rest_mass_cgs() == 0 and alpha == 0:
            return a_num * temperature**3
        return None

    def default_number_density_integrand(self, x, temperature, alpha):
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

    def default_pressure_function(self, temperature, alpha):
        """The default pressure function to treat case of zero-rest-mass boson.

        Args:
            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute
            the function.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the boson pressure in cgs units for the given input
            for the case when the rest mass and chemical potential are zero.  When they are
            not zero, the routine returns None so that other routines will compute the
            function by integration.

        """
        if self.get_rest_mass_cgs() == 0 and alpha == 0:
            return a_rad * temperature**4 / 3.0
        return None

    def default_pressure_integrand(self, x, temperature, alpha):
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

    def default_energy_density_function(self, temperature, alpha):
        """The default energy density function to treat case of zero-rest-mass boson.

        Args:
            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute
            the function.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the boson energy density in cgs units for the given input
            for the case when the rest mass and chemical potential are zero.  When they are
            not zero, the routine returns None so that other routines will compute the
            function by integration.

        """
        if self.get_rest_mass_cgs() == 0 and alpha == 0:
            return a_rad * temperature**4
        return None

    def default_energy_density_integrand(self, x, temperature, alpha):
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

    def default_entropy_density_function(self, temperature, alpha):
        """The default entropy density function to treat case of zero-rest-mass boson.

        Args:
            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute
            the function.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the boson entropy density in cgs units for the given input
            for the case when the rest mass and chemical potential are zero.  When they are
            not zero, the routine returns None so that other routines will compute the
            function by integration.

        """
        if self.get_rest_mass_cgs() == 0 and alpha == 0:
            return 4.0 * a_rad * temperature**3 / 3.0
        return None

    def default_entropy_density_integrand(self, x, temperature, alpha):
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

    def default_internal_energy_density_function(self, temperature, alpha):
        """The default internal energy function to treat case of zero-rest-mass boson.

        Args:
            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute
            the function.

            ``alpha`` (:obj:`float`):  The chemical potential (less the rest mass) divided by kT.

        Returns:
            A :obj:`float` giving the boson internal energy density in cgs units for the given
            input for the case when the rest mass and chemical potential are zero.  When they are
            not zero, the routine returns None so that other routines will compute the
            function by integration.

        """
        if self.get_rest_mass_cgs() == 0 and alpha == 0:
            return a_rad * temperature**4
        return None

    def default_internal_energy_density_integrand(self, x, temperature, alpha):
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

    def compute_temperature_derivative(self, quantity, temperature, number_density):
        """Routine to compute the temperature derivative of a thermodynamic quantity
        for the boson.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity to compute.

            ``temperature`` (:obj:`float`): The temperature (in K) at which to compute the
            derivative.

            ``number_density`` (:obj:`float`):  The fixed number density at which to
            compute the derivative.

        Returns:
            A :obj:`float` giving the temperature derivative of the quantity in cgs units.

        """

        return self._compute_temperature_derivative(
            self.functions[quantity],
            self.integrands[quantity],
            self.functions["number density"],
            self.integrands["number density"],
            temperature,
            number_density,
        )


def create_photon():
    """Convenience routine for creating a photon.

    Returns:
        A photon as a :obj:`wnstatmech.boson.Boson` object.

    """
    return Boson("photon", 0, 2, 0)
