"""Performance benchmarks for representative wnstatmech workloads.

Run with ``pytest benchmarks --benchmark-only`` after installing the
``benchmark`` optional dependency.  Scalar benchmarks use fresh particles to
avoid measuring exact-value cache hits rather than numerical work.
"""

import numpy as np
import wnstatmech as ws

CLASSICAL_TEMPERATURE = 5.2512e5
CLASSICAL_ALPHA = -19.773
RELATIVISTIC_TEMPERATURE = 1.0e10
RELATIVISTIC_ALPHA = -1.0
DEGENERATE_TEMPERATURE = 1.0e8
DEGENERATE_ALPHA = 8.0


def _cold_quantity(quantity, temperature, alpha):
    electron = ws.fermion.create_electron()
    return electron.compute_quantity(quantity, temperature, alpha)


def _cold_chemical_potential(temperature, alpha):
    electron = ws.fermion.create_electron()
    number_density = electron.compute_quantity(
        "number density", temperature, alpha
    )
    return electron.compute_chemical_potential(temperature, number_density)


def _batch_states(size, alpha_start, alpha_stop):
    temperatures = np.geomspace(1.0e7, 1.0e10, size)
    alphas = np.linspace(alpha_start, alpha_stop, size)
    return temperatures, alphas


def test_scalar_classical_pressure(benchmark):
    benchmark(
        _cold_quantity,
        "pressure",
        CLASSICAL_TEMPERATURE,
        CLASSICAL_ALPHA,
    )


def test_scalar_relativistic_energy_density(benchmark):
    benchmark(
        _cold_quantity,
        "energy density",
        RELATIVISTIC_TEMPERATURE,
        RELATIVISTIC_ALPHA,
    )


def test_scalar_degenerate_internal_energy_density(benchmark):
    benchmark(
        _cold_quantity,
        "internal energy density",
        DEGENERATE_TEMPERATURE,
        DEGENERATE_ALPHA,
    )


def test_scalar_classical_chemical_potential(benchmark):
    benchmark(
        _cold_chemical_potential,
        CLASSICAL_TEMPERATURE,
        CLASSICAL_ALPHA,
    )


def test_batched_pressure_128_states(benchmark):
    electron = ws.fermion.create_electron()
    temperatures, alphas = _batch_states(128, -8.0, -1.0)
    benchmark(electron.compute_quantity, "pressure", temperatures, alphas)


def test_batched_energy_density_128_states(benchmark):
    electron = ws.fermion.create_electron()
    temperatures, alphas = _batch_states(128, -2.0, 4.0)
    benchmark(
        electron.compute_quantity,
        "energy density",
        temperatures,
        alphas,
    )


def test_batched_degenerate_energy_density_128_states(benchmark):
    electron = ws.fermion.create_electron()
    temperatures, alphas = _batch_states(128, 0.25, 16.25)
    benchmark(
        electron.compute_quantity,
        "energy density",
        temperatures,
        alphas,
    )


def test_batched_energy_density_derivative_16_states(benchmark):
    temperatures, alphas = _batch_states(16, -3.0, 3.0)

    def derivative():
        electron = ws.fermion.create_electron()
        number_densities = electron.compute_quantity(
            "number density", temperatures, alphas
        )
        return electron.compute_temperature_derivative(
            "energy density", temperatures, number_densities
        )

    benchmark(derivative)
