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


def safe_entropy_term_boson(a):
    try:
        exp_a = safe_exp(a)
        denom = exp_a - 1
        if abs(denom) < 1e-12:
            return 0.0
        return a / denom - math.log1p(-exp_a)
    except:
        return 0.0


class Particle:
    def __init__(self, name, rest_mass_mev, multiplicity, charge):
        if rest_mass_mev < 0 or multiplicity <= 0:
            raise ValueError("Invalid rest mass or multiplicity.")
        self.name = name
        self.rest_mass = rest_mass_mev
        self.multiplicity = multiplicity
        self.charge = charge

    def get_rest_mass_cgs(self):
        return (
            self.rest_mass
            * gc.consts.GSL_CONST_CGSM_ELECTRON_VOLT
            * gc.consts.GSL_CONST_NUM_MEGA
        )

    def get_gamma(self, temperature):
        return self.get_rest_mass_cgs() / (
            gc.consts.GSL_CONST_CGSM_BOLTZMANN * temperature
        )

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
        self, integrand_fn, temperature, number_density
    ):
        def root_fn(alpha):
            return (
                self._compute_quantity(integrand_fn, temperature, alpha)
                - number_density
            )

        lower, upper = _bracket_root(root_fn, -1)

        return brentq(root_fn, lower, upper)

    def _compute_quantity(self, integrand_fn, temperature, alpha):
        result, _ = quad(
            integrand_fn,
            0,
            np.inf,
            args=(temperature, alpha),
            epsabs=1e-8,
            epsrel=1e-8,
        )
        return result

    def _compute_temperature_derivative(
        self, integrand_fn, temperature, alpha
    ):
        def func(temp, integrand_fn, alpha):
            return self._compute_quantity(integrand_fun, temp, alpha)

        return derivative(
            func,
            temperature,
            dx=1e-3 * temperature,
            n=1,
            order=3,
            args=(integrand_fn, alpha),
        )
