import requests, io
import numpy as np
import wnstatmech as ws


def test_alpha():
    electron = ws.fermion.Fermion("electron", 0.511, 2, -1)

    Ts = np.logspace(3, 12, 10)
    alphas = np.linspace(-10, 10, 21) 

    for T in Ts:
        for alpha in alphas:
            n_fermion = electron.compute_quantity("number density", T, alpha)
            mu_fermion = electron.compute_chemical_potential(T, n_fermion)
            assert np.isclose(alpha, mu_fermion, atol=1.0e-8)

def test_fermion_quantities():
    electron = ws.fermion.create_electron()

    Ts = np.logspace(0, 12, 13)
    alphas = np.linspace(-10, 10, 21) 

    for T in Ts:
        for alpha in alphas:
            p_fermion = electron.compute_quantity("pressure", T, alpha)
            assert p_fermion > 0
            e_fermion = electron.compute_quantity("energy density", T, alpha)
            assert e_fermion > 0
            ei_fermion = electron.compute_quantity("internal energy density", T, alpha)
            assert ei_fermion
            s_fermion = electron.compute_quantity("entropy density", T, alpha)
            assert s_fermion > 0

def test_photon_quantities():
    photon = ws.boson.create_photon()

    Ts = np.logspace(0, 12, 13)

    for T in Ts:
        n_photon = photon.compute_quantity("number density", T, 0)
        assert n_photon > 0
        p_photon = photon.compute_quantity("pressure", T, 0)
        assert p_photon > 0
        e_photon = photon.compute_quantity("energy density", T, 0)
        assert e_photon > 0
        ei_photon = photon.compute_quantity("internal energy density", T, 0)
        assert ei_photon > 0
        s_photon = photon.compute_quantity("entropy density", T, 0)
        assert s_photon > 0
