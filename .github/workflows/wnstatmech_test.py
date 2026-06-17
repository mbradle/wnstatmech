import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import requests, io
import numpy as np
import gslconsts.consts as gc
import gslconsts.math as gm
import wnstatmech as ws

a = (
    4
    * gc.GSL_CONST_CGSM_STEFAN_BOLTZMANN_CONSTANT
    / gc.GSL_CONST_CGSM_SPEED_OF_LIGHT
)


def test_alpha():
    electron = ws.fermion.Fermion("electron", 0.511, 2, -1)

    Ts = np.logspace(3, 12, 10)
    alphas = np.linspace(-10, 10, 21)

    for T in Ts:
        for alpha in alphas:
            n_fermion = electron.compute_quantity("number density", T, alpha)
            mu_fermion = electron.compute_chemical_potential(T, n_fermion)
            assert np.isclose(alpha, mu_fermion, atol=1.0e-8)


def test_electron_properties():
    electron = ws.fermion.create_electron()

    props = electron.get_properties()
    assert props["name"] == "electron"
    assert props["multiplicity"] == 2
    assert props["charge"] == -1


def test_photon_properties():
    photon = ws.boson.create_photon()

    props = photon.get_properties()
    assert props["name"] == "photon"
    assert props["rest mass"] == 0
    assert props["multiplicity"] == 2
    assert props["charge"] == 0


def test_electron_quantities():
    electron = ws.fermion.create_electron()

    Ts = np.logspace(0, 11, 12)
    alphas = np.linspace(-10, 10, 21)

    for T in Ts:
        for alpha in alphas:
            p_fermion = electron.compute_quantity("pressure", T, alpha)
            assert p_fermion > 0

            e_fermion = electron.compute_quantity("energy density", T, alpha)
            assert e_fermion > 0

            ei_fermion = electron.compute_quantity(
                "internal energy density", T, alpha
            )
            assert ei_fermion > 0

            s_fermion = electron.compute_quantity("entropy density", T, alpha)
            assert s_fermion > 0


def test_fermion_derivative():
    electron = ws.fermion.create_electron()

    Ts = np.logspace(7, 14, 4)
    n_dens = np.logspace(10, 32, 12)

    for T in Ts:
        for n_den in n_dens:
            dUdT = electron.compute_temperature_derivative(
                "energy density", T, n_den
            )
            TdSdT = T * electron.compute_temperature_derivative(
                "entropy density", T, n_den
            )
            assert np.isclose(dUdT, TdSdT, 1.0e-3)


def fermion_pressure_function(T, alpha, n):
    return n * gc.GSL_CONST_CGSM_BOLTZMANN * T


def test_fermion_function():
    neutron = ws.fermion.Fermion("neutron", 939.55, 2, 0)
    classical_neutron = ws.fermion.Fermion("classical neutron", 939.55, 2, 0)

    T = 1.0e7

    n = 1.0e20
    alpha = neutron.compute_chemical_potential(T, n)

    P1 = neutron.compute_quantity("pressure", T, alpha)

    my_func = lambda T, alpha: fermion_pressure_function(T, alpha, n)
    classical_neutron.update_function("pressure", my_func)
    P2 = classical_neutron.compute_quantity("pressure", T, alpha)

    assert np.isclose(P1, P2, rtol = 1.0e-5)


def fermion_pressure_integrand(x, T, alpha, self):
    kT = gc.GSL_CONST_CGSM_BOLTZMANN * T
    gamma = self.get_rest_mass_cgs() / kT
    part1 = self.get_properties()["multiplicity"] / (2 * gm.M_PI**2)
    part2 = (
        kT
        / (
            gc.GSL_CONST_CGSM_PLANCKS_CONSTANT_HBAR
            * gc.GSL_CONST_CGSM_SPEED_OF_LIGHT
        )
    ) ** 3

    return (
        part1
        * part2
        * kT
        * np.exp(alpha)
        * np.sqrt(2. * x) * np.exp(-x)
        * np.power(gamma, 3.0 / 2.0)
    )


def test_fermion_integrand():
    neutron = ws.fermion.Fermion("neutron", 939.55, 2, 0)
    classical_neutron = ws.fermion.Fermion("classical neutron", 939.55, 2, 0)

    T = 1.0e7

    alpha = -15
    P1 = neutron.compute_quantity("pressure", T, alpha)

    my_integrand = lambda x, T, alpha: fermion_pressure_integrand(
        x, T, alpha, classical_neutron
    )
    classical_neutron.update_integrand("pressure", my_integrand)
    P2 = classical_neutron.compute_quantity("pressure", T, alpha)

    assert np.isclose(P1, P2, rtol = 1.0e-5)


def _assert_vectorized_quantity_matches_scalar(
    particle, quantity, temperatures, alphas, rtol=1.0e-5
):
    vector_result = np.asarray(
        particle.compute_quantity(quantity, temperatures, alphas), dtype=float
    )
    scalar_result = np.asarray(
        [
            particle.compute_quantity(quantity, T, alpha)
            for T, alpha in zip(temperatures, alphas)
        ],
        dtype=float,
    )
    assert np.allclose(vector_result, scalar_result, rtol=rtol, atol=0.0)


def test_fermion_vectorized_quantities_match_scalar():
    electron = ws.fermion.create_electron()

    temperatures = np.array([1.0e7, 1.0e8, 1.0e9, 1.0e10])
    alphas = np.array([-5.0, -1.0, 1.0, 5.0])

    for quantity in (
        "number density",
        "pressure",
        "energy density",
        "internal energy density",
        "entropy density",
    ):
        _assert_vectorized_quantity_matches_scalar(
            electron, quantity, temperatures, alphas
        )


def test_boson_vectorized_quantities_match_scalar():
    photon = ws.boson.create_photon()

    temperatures = np.array([1.0e7, 1.0e8, 1.0e9])
    alphas = np.array([-5.0, -1.0, -0.1])

    for quantity in (
        "number density",
        "pressure",
        "energy density",
        "internal energy density",
        "entropy density",
    ):
        _assert_vectorized_quantity_matches_scalar(
            photon, quantity, temperatures, alphas, rtol=1.0e-4
        )


def test_vectorized_chemical_potential_round_trip():
    electron = ws.fermion.create_electron()

    temperatures = np.array([1.0e7, 1.0e8, 1.0e9, 1.0e10])
    alphas = np.array([-5.0, -1.0, 1.0, 5.0])
    number_densities = electron.compute_quantity(
        "number density", temperatures, alphas
    )

    roots = electron.compute_chemical_potential(
        temperatures, number_densities
    )

    assert np.allclose(roots, alphas, atol=1.0e-5)


def test_worker_integration_matches_scalar(tmp_path):
    script = tmp_path / "worker_check.py"
    script.write_text(
        textwrap.dedent(
            """
            import numpy as np
            import wnstatmech as ws

            if __name__ == "__main__":
                scalar = ws.fermion.create_electron().compute_quantity(
                    "pressure", 1.0e9, -2.0
                )
                parallel = ws.fermion.create_electron(
                    workers=2
                ).compute_quantity("pressure", 1.0e9, -2.0)
                print(np.isclose(scalar, parallel, rtol=1.0e-6, atol=0.0))
            """
        )
    )
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.strip() == "True"


def test_zero_custom_function_result_is_used():
    neutron = ws.fermion.Fermion("neutron", 939.55, 2, 0)
    neutron.update_function("pressure", lambda T, alpha: 0.0)

    assert neutron.compute_quantity("pressure", 1.0e7, -15.0) == 0.0


def test_invalid_integration_tolerances_raise():
    with pytest.raises(ValueError, match="Invalid integration tolerance"):
        ws.fermion.create_electron(integration_epsabs=-1.0)

    with pytest.raises(ValueError, match="Invalid integration tolerance"):
        ws.fermion.create_electron(integration_epsrel=0.0)



def test_photon_quantities():
    photon = ws.boson.create_photon()

    Ts = np.logspace(0, 12, 13)

    for T in Ts:
        n_photon = photon.compute_quantity("number density", T, 0)
        assert n_photon > 0
        p_photon = photon.compute_quantity("pressure", T, 0)
        assert p_photon > 0
        e_photon = photon.compute_quantity("energy density", T, 0)
        assert np.isclose(3 * p_photon, e_photon, 1.0e-8)
        assert np.isclose(e_photon, a * T**4, 1.0e-8)
        ei_photon = photon.compute_quantity("internal energy density", T, 0)
        assert ei_photon == e_photon
        s_photon = photon.compute_quantity("entropy density", T, 0)
        assert np.isclose(s_photon, 4.0 * a * T**3 / 3.0, 1.0e-8)
