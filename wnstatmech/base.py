"""This is the base module for the package."""

import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.differentiate import derivative
import gslconsts as gc


def _bracket_root(f, x0, args=()):
    factor = 1.6
    max_iter = 1000
    x1 = x0
    x2 = x1 + 1
    f1 = f(x1, *args)
    f2 = f(x2, *args)
    for _ in range(max_iter):
        if f1 * f2 < 0:
            return (x1, x2)
        if abs(f1) < abs(f2):
            x1 += factor * (x1 - x2)
            f1 = f(x1, *args)
        else:
            x2 += factor * (x2 - x1)
            f2 = f(x2, *args)
    return None


class Particle:
    """A class for base particles.

    Args:
        ``name`` (:obj:`str`): The name of the particle.

        ``rest_mass_mev`` (:obj:`float`): The rest mass energy of the particle (in MeV).

        ``multiplicity`` (:obj:`int`):  The multiplicity of the internal degrees of
        freedom of the particle (typically 2 times the spin plus one).

        ``charge`` (:obj:`int`):  The charge of the particle.

    """

    def __init__(self, name, rest_mass_mev, multiplicity, charge):
        if rest_mass_mev < 0 or multiplicity <= 0:
            raise ValueError("Invalid rest mass or multiplicity.")
        self.name = name
        self.rest_mass = rest_mass_mev
        self.multiplicity = multiplicity
        self.charge = charge
        self.functions = {}
        self.integrands = {}

    def get_rest_mass_cgs(self):
        """A method to return the rest mass energy of the particle
        in cgs units.

        Returns:
            A (:obj:`float`) with the rest mass energy of the particle in cgs units.
        """
        return (
            self.rest_mass
            * gc.consts.GSL_CONST_CGSM_ELECTRON_VOLT
            * gc.consts.GSL_CONST_NUM_MEGA
        )

    def get_gamma(self, temperature):
        """A method to return the rest mass energy of the particle
        divided by kT.

        Args:
            ``temperature`` (:obj:`float`): The temperature in K at which to compute
            the quanitity.

        Returns:
            A (:obj:`float`) with the rest mass energy of the particle in cgs units.
        """
        return self.get_rest_mass_cgs() / (
            gc.consts.GSL_CONST_CGSM_BOLTZMANN * temperature
        )

    def get_properties(self):
        """A method to return the particle properties.

        Returns:
            A (:obj:`dict`) the particles basic properties.
        """
        return {
            "name": self.name,
            "rest mass": self.rest_mass,
            "multiplicity": self.multiplicity,
            "charge": self.charge,
        }

    def _prefactor(self, temperature, power):
        return (
            (gc.consts.GSL_CONST_CGSM_BOLTZMANN * temperature) ** power
            * self.multiplicity
            / (
                2
                * gc.math.M_PI**2
                * (
                    gc.consts.GSL_CONST_CGSM_PLANCKS_CONSTANT_HBAR
                    * gc.consts.GSL_CONST_CGSM_SPEED_OF_LIGHT
                )
                ** 3
            )
        )

    def _safe_exp(self, x):
        try:
            return math.exp(x)
        except OverflowError:
            return float("inf")

    def _safe_expm1(self, x):
        try:
            return math.expm1(x)
        except OverflowError:
            return float("inf")

    def _compute_chemical_potential(
        self, func, integrand_fn, temperature, number_density
    ):
        def root_fn(alpha):
            return (
                self._compute_quantity(func, integrand_fn, temperature, alpha)
                - number_density
            )

        lower, upper = _bracket_root(root_fn, -1)

        return brentq(root_fn, lower, upper)

    def _compute_degenerate_quantity(self, integrand_fn, temperature, alpha):
        result = 0

        if alpha <= 20:
            tmp, _ = quad(
                integrand_fn,
                0.0,
                np.inf,
                args=(temperature, alpha),
            )
            result += tmp
        else:
            t_lims = [
                (0, alpha - 20),
                (alpha - 20, alpha - 10),
                (alpha - 10, alpha),
                (alpha, alpha + 10),
                (alpha + 10, alpha + 20),
                (alpha + 20, np.inf),
            ]
            for tup in t_lims:
                res_tup = quad(
                    integrand_fn,
                    tup[0],
                    tup[1],
                    limit=1000,
                    full_output=True,
                    args=(temperature, alpha),
                )
                result += res_tup[0]

        return result

    def _compute_quantity(self, func, integrand_fn, temperature, alpha):
        if func:
            result = func(temperature, alpha)
            if result:
                return result

        if alpha <= 0:
            result, _ = quad(
                integrand_fn,
                0,
                np.inf,
                args=(temperature, alpha),
            )
        else:
            result = self._compute_degenerate_quantity(
                integrand_fn, temperature, alpha
            )

        return result

    def _compute_temperature_derivative(
        self,
        func,
        integrand_fn,
        num_func,
        num_integrand_fn,
        temperature,
        number_density,
    ):
        def deriv_func(temp):
            if temp.ndim == 0:
                alpha = self._compute_chemical_potential(
                    num_func, num_integrand_fn, temp, number_density
                )
                return self._compute_quantity(func, integrand_fn, temp, alpha)

            result = np.zeros((temp.shape[0], temp.shape[1]))
            for i in range(temp.shape[1]):
                alpha = self._compute_chemical_potential(
                    num_func, num_integrand_fn, temp[0, i], number_density
                )
                result[0, i] = self._compute_quantity(
                    func, integrand_fn, temp[0, i], alpha
                )
            return result

        return derivative(
            deriv_func,
            temperature,
            initial_step=1e-2 * temperature,
        ).df

    def update_functions(self, quantity, func):
        """A method to update the functions for the particle.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity.

            ``func``: A user-defined function to return the value of the quantity \
            for some or all sets of conditions.  The function must take two arguments. \
            The second is *T*, the temperature  in Kelvin, \
            and the second is the *alpha*, the chemical potential (less the rest mass) \
            divided by kT.  Other data can be bound to the integrand function.
            The function must return the value of the quantity for the input conditions, \
            or None, if the conditions are not appropriate.
        """

        self.functions[quantity] = func

    def update_integrands(self, quantity, integrand_fn):
        """A method to update the integrands for the particle.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity.

            ``integrand_fn`` (:obj:`float`): The integrand corresponding to the \
            quantity.  The integrand function must take three arguments.  The first \
            is the scaled energy *x*, the second is *T*, the temperature  in Kelvin, \
            and the third is the *alpha*, the chemical potential (less the rest mass) \
            divided by kT.  Other data can be bound to the integrand function.
        """

        self.integrands[quantity] = integrand_fn
