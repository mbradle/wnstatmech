"""This is the base module for the package."""

import math
from collections import OrderedDict
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq
import scipy.optimize.elementwise as optel
import gslconsts as gc

DEFAULT_INTEGRATION_EPSABS = 0.0
DEFAULT_INTEGRATION_EPSREL = 1.0e-6
DEFAULT_TEMPERATURE_DERIVATIVE_REL_STEP = 1.0e-4
DEFAULT_ALPHA_DERIVATIVE_REL_STEP = 1.0e-4
DEFAULT_CHEMICAL_POTENTIAL_WARM_START_FACTOR = 10.0
DEFAULT_CACHE_SIZE = 1024
DEGENERATE_BATCH_SIZE = 32


def _is_nearby_positive(value, reference, factor):
    if value <= 0 or reference <= 0:
        return value == reference
    ratio = value / reference
    return 1.0 / factor <= ratio <= factor


def _to_scalar_or_array(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def _bracket_root_scalar(f, x0):
    factor = 1.6
    max_iter = 1000
    x1 = x0
    x2 = x1 + 1.0
    f1 = f(x1)
    if f1 == 0:
        return x1, x1
    f2 = f(x2)
    if f2 == 0:
        return x2, x2

    for _ in range(max_iter):
        if f1 * f2 < 0:
            return x1, x2
        if abs(f1) < abs(f2):
            x1 += factor * (x1 - x2)
            f1 = f(x1)
            if f1 == 0:
                return x1, x1
        else:
            x2 += factor * (x2 - x1)
            f2 = f(x2)
            if f2 == 0:
                return x2, x2

    raise RuntimeError("Unable to bracket chemical potential root.")


class _VectorIntegrand:
    def __init__(self, integrand_fn, temperatures, alphas):
        self.integrand_fn = integrand_fn
        self.temperatures = np.asarray(temperatures, dtype=float).ravel()
        self.alphas = np.asarray(alphas, dtype=float).ravel()

    def __call__(self, x):
        try:
            result = np.asarray(
                self.integrand_fn(x, self.temperatures, self.alphas),
                dtype=float,
            )
            if result.shape == self.temperatures.shape:
                return result
            if result.shape == ():
                return np.full(self.temperatures.shape, result.item())
        except (TypeError, ValueError):
            pass

        return np.array(
            [
                self.integrand_fn(x, temperature, alpha)
                for temperature, alpha in zip(self.temperatures, self.alphas)
            ]
        )


class Particle:
    """A class for base particles.

    Args:
        ``name`` (:obj:`str`): The name of the particle.

        ``rest_mass_mev`` (:obj:`float`): The rest mass energy of the particle (in MeV).

        ``multiplicity`` (:obj:`int`):  The multiplicity of the internal degrees of
        freedom of the particle (typically 2 times the spin plus one).

        ``charge`` (:obj:`int`):  The charge of the particle.

        ``integration_epsabs`` (:obj:`float`): Absolute integration tolerance.

        ``integration_epsrel`` (:obj:`float`): Relative integration tolerance.

        ``cache_size`` (:obj:`int`): Maximum number of exact scalar results
        retained for each cache.  Set to zero to disable exact-result caching.

    """

    def __init__(
        self,
        name,
        rest_mass_mev,
        multiplicity,
        charge,
        integration_epsabs=DEFAULT_INTEGRATION_EPSABS,
        integration_epsrel=DEFAULT_INTEGRATION_EPSREL,
        cache_size=DEFAULT_CACHE_SIZE,
    ):
        if rest_mass_mev < 0 or multiplicity <= 0:
            raise ValueError("Invalid rest mass or multiplicity.")
        if integration_epsabs < 0 or integration_epsrel <= 0:
            raise ValueError("Invalid integration tolerance.")
        if not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("Invalid cache size.")
        self.name = name
        self.rest_mass = rest_mass_mev
        self.multiplicity = multiplicity
        self.charge = charge
        self.integration_epsabs = integration_epsabs
        self.integration_epsrel = integration_epsrel
        self.cache_size = cache_size
        self.functions = {}
        self.integrands = {}
        self.chemical_potential_function = None
        self._chemical_potential_cache = OrderedDict()
        self._chemical_potential_seed = {}
        self._quantity_cache = OrderedDict()

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

    def _cache_get(self, cache, key):
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    def _cache_set(self, cache, key, value):
        if self.cache_size == 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.cache_size:
            cache.popitem(last=False)

    def clear_cache(self):
        """Clear exact-result caches and the chemical-potential warm start."""
        self._chemical_potential_cache.clear()
        self._chemical_potential_seed.clear()
        self._quantity_cache.clear()

    def _compute_chemical_potential(
        self, func, integrand_fn, temperature, number_density
    ):
        temperatures, number_densities = np.broadcast_arrays(
            np.asarray(temperature, dtype=float),
            np.asarray(number_density, dtype=float),
        )
        if temperatures.shape == ():
            direct_alpha = self._compute_direct_chemical_potential(
                temperatures.item(), number_densities.item()
            )
            if direct_alpha is not None:
                return direct_alpha
            return self._compute_chemical_potential_numerical(
                func,
                integrand_fn,
                temperatures.item(),
                number_densities.item(),
            )

        result = np.empty(temperatures.size)
        flat_temperatures = temperatures.ravel()
        flat_number_densities = number_densities.ravel()
        pending = np.ones(temperatures.size, dtype=bool)

        if self.chemical_potential_function is not None:
            for i, (temp, num_den) in enumerate(
                zip(flat_temperatures, flat_number_densities)
            ):
                direct_alpha = self._compute_direct_chemical_potential(
                    temp.item(), num_den.item()
                )
                if direct_alpha is not None:
                    result[i] = direct_alpha
                    pending[i] = False

        pending_indices = np.nonzero(pending)[0]
        if pending_indices.size:
            result[pending_indices] = np.asarray(
                self._compute_chemical_potential_numerical(
                    func,
                    integrand_fn,
                    flat_temperatures[pending_indices],
                    flat_number_densities[pending_indices],
                ),
                dtype=float,
            ).ravel()

        return _to_scalar_or_array(result.reshape(temperatures.shape))

    def _compute_direct_chemical_potential(self, temperature, number_density):
        if self.chemical_potential_function is None:
            return None
        return self.chemical_potential_function(temperature, number_density)

    def _compute_chemical_potential_numerical(
        self, func, integrand_fn, temperature, number_density
    ):
        if np.ndim(temperature) == 0 and np.ndim(number_density) == 0:
            seed_key = (id(func), id(integrand_fn))
            temperature = float(temperature)
            number_density = float(number_density)
            cache_key = (
                id(func),
                id(integrand_fn),
                temperature,
                number_density,
            )
            result = self._cache_get(self._chemical_potential_cache, cache_key)
            if result is not None:
                self._chemical_potential_seed[seed_key] = (
                    result,
                    temperature,
                    number_density,
                )
                return result

            def root_fn_scalar(alpha):
                return (
                    self._compute_quantity_scalar(
                        func, integrand_fn, temperature, alpha
                    )
                    - number_density
                )

            x0 = -1.0
            seed = self._chemical_potential_seed.get(seed_key)
            if seed is not None:
                seed_alpha, seed_temperature, seed_number_density = seed
                factor = DEFAULT_CHEMICAL_POTENTIAL_WARM_START_FACTOR
                if _is_nearby_positive(
                    temperature, seed_temperature, factor
                ) and _is_nearby_positive(
                    number_density, seed_number_density, factor
                ):
                    x0 = seed_alpha

            lower, upper = _bracket_root_scalar(root_fn_scalar, x0)
            if lower == upper:
                result = lower
            else:
                result = brentq(root_fn_scalar, lower, upper)
            self._cache_set(self._chemical_potential_cache, cache_key, result)
            self._chemical_potential_seed[seed_key] = (
                result,
                temperature,
                number_density,
            )
            return result

        def root_fn(alpha, temp, num_den):
            return (
                self._compute_quantity(func, integrand_fn, temp, alpha)
                - num_den
            )

        args = (temperature, number_density)
        bracket = optel.bracket_root(
            root_fn,
            -1.0,
            args=args,
            factor=1.6,
            maxiter=1000,
        )
        if not np.all(bracket.success):
            raise RuntimeError("Unable to bracket chemical potential root.")

        result = optel.find_root(root_fn, bracket.bracket, args=args)
        if not np.all(result.success):
            raise RuntimeError("Unable to compute chemical potential root.")

        return _to_scalar_or_array(result.x)

    def _integrate(self, integrand_fn, lower, upper, temperature, alpha):
        result, _ = quad(
            integrand_fn,
            lower,
            upper,
            args=(temperature, alpha),
            epsabs=self.integration_epsabs,
            epsrel=self.integration_epsrel,
            limit=1000,
        )
        return result

    def _integrate_batch(
        self, integrand_fn, lower, upper, temperatures, alphas
    ):
        vector_integrand = _VectorIntegrand(integrand_fn, temperatures, alphas)
        result, _ = quad_vec(
            vector_integrand,
            lower,
            upper,
            epsabs=self.integration_epsabs,
            epsrel=self.integration_epsrel,
        )
        result = np.asarray(result, dtype=float)
        if result.shape == ():
            result = np.full(vector_integrand.temperatures.shape, result)
        return result.reshape(vector_integrand.temperatures.shape)

    def _compute_degenerate_quantity_scalar(
        self, integrand_fn, temperature, alpha
    ):
        result = 0
        split_points = np.array(
            [alpha - 20, alpha - 10, alpha, alpha + 10, alpha + 20],
            dtype=float,
        )
        split_points = split_points[split_points > 0]
        limits = np.concatenate(([0.0], np.unique(split_points), [np.inf]))

        for lower, upper in zip(limits[:-1], limits[1:]):
            if lower == upper:
                continue
            result += self._integrate(
                integrand_fn,
                lower,
                upper,
                temperature,
                alpha,
            )

        return result

    def _compute_degenerate_quantity_batch(
        self, integrand_fn, result, indices, temperatures, alphas
    ):
        """Compute positive-alpha states in small, similarly valued batches."""
        sorted_indices = indices[np.argsort(alphas[indices])]

        for start in range(0, sorted_indices.size, DEGENERATE_BATCH_SIZE):
            batch_indices = sorted_indices[
                start : start + DEGENERATE_BATCH_SIZE
            ]
            batch_alphas = alphas[batch_indices]
            batch_result = np.zeros(batch_indices.size)
            points = np.unique(
                np.concatenate(
                    (
                        batch_alphas - 20,
                        batch_alphas - 10,
                        batch_alphas,
                        batch_alphas + 10,
                        batch_alphas + 20,
                    )
                )
            )
            points = points[points > 0]
            limits = np.concatenate(([0.0], points, [np.inf]))

            for lower, upper in zip(limits[:-1], limits[1:]):
                if lower == upper:
                    continue
                batch_result += self._integrate_batch(
                    integrand_fn,
                    lower,
                    upper,
                    temperatures[batch_indices],
                    batch_alphas,
                )
            result[batch_indices] = batch_result

    def _quantity_cache_key(self, integrand_fn, temperature, alpha):
        return (
            id(integrand_fn),
            float(temperature),
            float(alpha),
            self.integration_epsabs,
            self.integration_epsrel,
        )

    def _compute_quantity_scalar(self, func, integrand_fn, temperature, alpha):
        temperature = float(temperature)
        alpha = float(alpha)

        if func:
            result = func(temperature, alpha)
            if result is not None:
                return result

        cache_key = self._quantity_cache_key(integrand_fn, temperature, alpha)
        result = self._cache_get(self._quantity_cache, cache_key)
        if result is not None:
            return result

        if alpha <= 0:
            result = self._integrate(
                integrand_fn,
                0,
                np.inf,
                temperature,
                alpha,
            )
        else:
            result = self._compute_degenerate_quantity_scalar(
                integrand_fn, temperature, alpha
            )

        self._cache_set(self._quantity_cache, cache_key, result)
        return result

    def _compute_quantity(self, func, integrand_fn, temperature, alpha):
        temps, alphas = np.broadcast_arrays(
            np.asarray(temperature, dtype=float),
            np.asarray(alpha, dtype=float),
        )
        if temps.shape == ():
            return self._compute_quantity_scalar(
                func, integrand_fn, temps.item(), alphas.item()
            )

        result = np.empty(temps.size)
        flat_temps = temps.ravel()
        flat_alphas = alphas.ravel()
        pending = np.ones(temps.size, dtype=bool)

        if func:
            for i, (temp, alpha_i) in enumerate(zip(flat_temps, flat_alphas)):
                value = func(temp.item(), alpha_i.item())
                if value is not None:
                    result[i] = value
                    pending[i] = False

        pending_indices = np.nonzero(pending)[0]
        if pending_indices.size:
            regular = pending_indices[flat_alphas[pending_indices] <= 0]
            if regular.size:
                result[regular] = self._integrate_batch(
                    integrand_fn,
                    0.0,
                    np.inf,
                    flat_temps[regular],
                    flat_alphas[regular],
                )

            degenerate = pending_indices[flat_alphas[pending_indices] > 0]
            if degenerate.size:
                self._compute_degenerate_quantity_batch(
                    integrand_fn,
                    result,
                    degenerate,
                    flat_temps,
                    flat_alphas,
                )

        return result.reshape(temps.shape)

    def _partial_temperature_derivative(
        self, func, integrand_fn, temperature, alpha
    ):
        step = DEFAULT_TEMPERATURE_DERIVATIVE_REL_STEP * np.asarray(
            temperature, dtype=float
        )
        return (
            self._compute_quantity(
                func, integrand_fn, temperature + step, alpha
            )
            - self._compute_quantity(
                func, integrand_fn, temperature - step, alpha
            )
        ) / (2.0 * step)

    def _partial_alpha_derivative(
        self, func, integrand_fn, temperature, alpha
    ):
        alpha_array = np.asarray(alpha, dtype=float)
        step = DEFAULT_ALPHA_DERIVATIVE_REL_STEP * np.maximum(
            1.0, np.abs(alpha_array)
        )
        return (
            self._compute_quantity(
                func, integrand_fn, temperature, alpha_array + step
            )
            - self._compute_quantity(
                func, integrand_fn, temperature, alpha_array - step
            )
        ) / (2.0 * step)

    def _compute_temperature_derivative_finite_difference(
        self,
        func_int_tuple,
        temperature,
        number_density,
    ):
        def quantity_at(temp):
            alpha = self._compute_chemical_potential(
                func_int_tuple[2],
                func_int_tuple[3],
                temp,
                number_density,
            )
            return self._compute_quantity(
                func_int_tuple[0], func_int_tuple[1], temp, alpha
            )

        temp_array = np.asarray(temperature, dtype=float)
        step = DEFAULT_TEMPERATURE_DERIVATIVE_REL_STEP * temp_array
        result = (
            quantity_at(temp_array + step) - quantity_at(temp_array - step)
        ) / (2.0 * step)

        return _to_scalar_or_array(result)

    def _compute_temperature_derivative(
        self,
        func_int_tuple,
        temperature,
        number_density,
    ):
        temp_array, number_density_array = np.broadcast_arrays(
            np.asarray(temperature, dtype=float),
            np.asarray(number_density, dtype=float),
        )

        alpha = self._compute_chemical_potential(
            func_int_tuple[2],
            func_int_tuple[3],
            temp_array,
            number_density_array,
        )
        alpha_array = np.asarray(alpha, dtype=float)

        try:
            number_t = self._partial_temperature_derivative(
                func_int_tuple[2],
                func_int_tuple[3],
                temp_array,
                alpha_array,
            )
            number_alpha = self._partial_alpha_derivative(
                func_int_tuple[2],
                func_int_tuple[3],
                temp_array,
                alpha_array,
            )
            quantity_t = self._partial_temperature_derivative(
                func_int_tuple[0],
                func_int_tuple[1],
                temp_array,
                alpha_array,
            )
            quantity_alpha = self._partial_alpha_derivative(
                func_int_tuple[0],
                func_int_tuple[1],
                temp_array,
                alpha_array,
            )
            result = quantity_t - quantity_alpha * number_t / number_alpha
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            ZeroDivisionError,
        ):
            return self._compute_temperature_derivative_finite_difference(
                func_int_tuple, temperature, number_density
            )

        if not np.all(np.isfinite(result)):
            return self._compute_temperature_derivative_finite_difference(
                func_int_tuple, temperature, number_density
            )

        return _to_scalar_or_array(result)

    def update_function(self, quantity, func):
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
        self.clear_cache()

    def update_chemical_potential_function(self, func):
        """Set a direct chemical-potential function or ``None``.

        The function receives scalar ``(temperature, number_density)`` values
        in K and cm^-3 and returns alpha, the chemical potential less rest
        mass divided by kT.  Return ``None`` when the direct calculation does
        not apply; wnstatmech will then use its numerical root solve.  Array
        inputs are dispatched to the function one state at a time, allowing
        direct and numerical results in the same batch.
        """
        self.chemical_potential_function = func
        self.clear_cache()

    def update_integrand(self, quantity, integrand_fn):
        """A method to update an integrand for the particle.

        Args:
            ``quantity`` (:obj:`str`): The name of the quantity.

            ``integrand_fn`` (:obj:`float`): The integrand corresponding to the \
            quantity.  The integrand function must take three arguments.  The first \
            is the scaled energy *x*, the second is *T*, the temperature  in Kelvin, \
            and the third is the *alpha*, the chemical potential (less the rest mass) \
            divided by kT.  Other data can be bound to the integrand function.
        """

        self.integrands[quantity] = integrand_fn
        self.clear_cache()
