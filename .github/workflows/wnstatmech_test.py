import requests, io
import numpy as np
import gslconsts.consts as gc
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

            #dUdT = electron.compute_temperature_derivative("internal energy density", T, alpha)
            #TdSdT = T * electron.compute_temperature_derivative("entropy density", T, alpha)
            #assert np.isclose(dUdT, TdSdT, 1.e-3)


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
        assert np.isclose(s_photon, 4.0 * a * T**3 / 3.0, 1.0e-3)
        dUdT = photon.compute_temperature_derivative("energy density", T, 0)
        TdSdT = T * photon.compute_temperature_derivative("entropy density", T, 0)
        assert np.isclose(dUdT, TdSdT, 1.e-3)
