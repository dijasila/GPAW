# General modules
import pytest

from itertools import product

import numpy as np

import matplotlib.pyplot as plt

# Script modules
from ase.units import Hartree

from gpaw import GPAW
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.jdos import JDOSCalculator
from gpaw.response.symmetry import KPointFinder


@pytest.mark.response
@pytest.mark.kspair
def test_iron_jdos(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    wd = FrequencyDescriptor.from_array_or_dict(np.linspace(-10.0, 10.0, 321))
    eta = 0.2

    spincomponent_s = ['00', '+-']
    bandsummation_b = ['double', 'pairwise']

    # ---------- Script ---------- #

    # Get the ground state calculator from the fixture
    calc = GPAW(gpw_files['fe_pw_wfs'], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']

    # Set up the JDOSCalculator
    gs = ResponseGroundStateAdapter(calc)
    jdos_calc = JDOSCalculator(gs)

    # Set up reference MyManualJDOS
    jdos_refcalc = MyManualJDOS(calc)

    for q_c, spincomponent in product(q_qc, spincomponent_s):
        jdosref_w = jdos_refcalc.calculate(q_c, wd.omega_w * Hartree,
                                           eta=eta,
                                           spincomponent=spincomponent,
                                           nbands=nbands)
        for bandsummation in bandsummation_b:
            jdos_w = jdos_calc.calculate(q_c, wd,
                                         eta=eta,
                                         spincomponent=spincomponent,
                                         nbands=nbands)
            assert jdos_w == pytest.approx(jdosref_w)

        # plt.subplot()
        # plt.plot(wd.omega_w * Hartree, jdos_w)
        # plt.plot(wd.omega_w * Hartree, jdosref_w)
        # plt.title(f'{q_c} {spincomponent}')
        # plt.show()


class MyManualJDOS:
    def __init__(self, calc):
        self.calc = calc

        kd = calc.wfs.kd
        gd = calc.wfs.gd
        self.kd = kd
        self.kptfinder = KPointFinder(kd.bzk_kc)
        self.kweight = 1 / (gd.volume * len(kd.bzk_kc))

    def calculate(self, q_c, omega_w, eta=0.05, spincomponent='00', nbands=None):
        r"""Calculate the joint density of states:
                       __  __
                    1  \   \
        g_j(q, ω) = ‾  /   /  (f_nks - f_mk+qs') δ(ω-[ε_mk+qs' - ε_nks])
                    V  ‾‾  ‾‾
                       k   n,m

        for a given spin component specifying the spin transitions s -> s'.
        """
        q_c = np.asarray(q_c)
        # Internal frequencies in Hartree
        omega_w = omega_w / Hartree
        eta = eta / Hartree
        # Allocate array
        jdos_w = np.zeros_like(omega_w)
        
        for K1, k1_c in enumerate(self.kd.bzk_kc):
            # de = e2 - e1, df = f2 - f1
            de_t, df_t = self.get_transitions(K1, k1_c, q_c, spincomponent, nbands)

            # Set up jdos
            delta_wt = self.delta(omega_w, eta, de_t)
            jdos_wt = - df_t[np.newaxis] * delta_wt

            # Sum over transitions
            jdos_w += self.kweight * np.sum(jdos_wt, axis=1)

        return jdos_w

    @staticmethod
    def delta(omega_w, eta, de_t):
        r"""Create lorentzian delta-functions

                ~ 1       η
        δ(ω-Δε) = ‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                  π (ω-Δε)^2 + η^2
        """
        x_wt = omega_w[:, np.newaxis] - de_t[np.newaxis]
        return eta / np.pi / (x_wt**2. + eta**2.)

    def get_transitions(self, K1, k1_c, q_c, spincomponent, nbands):
        assert isinstance(nbands, int)
        if spincomponent == '00':
            s1_s = [0, 1]
            s2_s = [0, 1]
        elif spincomponent == '+-':
            s1_s = [0]
            s2_s = [1]
        else:
            raise ValueError(spincomponent)

        # Find k_c + q_c
        K2 = self.kptfinder.find(k1_c + q_c)

        de_t = []
        df_t = []
        kd = self.kd
        calc = self.calc
        for s1, s2 in zip(s1_s, s2_s):
            # Get composite u=(s,k) indices and KPoint objects
            u1 = kd.bz2ibz_k[K1] * 2 + s1  # nspins = 2
            u2 = kd.bz2ibz_k[K2] * 2 + s2
            kpt1, kpt2 = calc.wfs.kpt_u[u1], calc.wfs.kpt_u[u2]

            # Extract eigenenergies and occupation numbers
            eps1_n, eps2_n = kpt1.eps_n[:nbands], kpt2.eps_n[:nbands]
            f1_n, f2_n = kpt1.f_n[:nbands] / kpt1.weight, kpt2.f_n[:nbands] / kpt2.weight

            # Append data
            de_nm = eps2_n[:, np.newaxis] - eps1_n[np.newaxis]
            df_nm = f2_n[:, np.newaxis] - f1_n[np.newaxis]
            de_t += list(de_nm.flatten())
            df_t += list(df_nm.flatten())
        de_t = np.array(de_t)
        df_t = np.array(df_t)

        return de_t, df_t
        
