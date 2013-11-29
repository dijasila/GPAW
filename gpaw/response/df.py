import os
import sys
import pickle

import numpy as np

import gpaw.mpi as mpi
from gpaw.response.chi0 import Chi0


class DielectricFunction:
    """This class defines dielectric function related physical quantities."""
    def __init__(self, calc=None, name=None, omega_w=None, ecut=50,
                 eta=0.2, ftol=1e-6,
                 world=mpi.world, txt=sys.stdout):
        if omega_w is None:
            omega_w = np.linspace(0, 20, 51, end=True)
            
        self.chi0 = Chi0(calc, omega_w, ecut=ecut, eta=eta, ftol=ftol,
                         world=world, txt=txt)
        
        self.name = name

    def calculate_chi0(self, q_c):
        if self.name:
            kd = self.chi0.calc.wfs.kd
            name = self.name + '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round())
            if os.path.isfile(name):
                pd, chi0_wGG, ch0_wxvG = pickle.load(open(name))
                return pd, chi0_wGG, ch0_wxvG
        pd, chi0_wGG, ch0_wxvG = self.chi0.calculate(q_c)
        if self.name:
            pickle.dump(open(name, 'wb'), (pd, chi0_wGG, ch0_wxvG),
                        pickle.HIGHEST_PROTOCOL)
        return pd, chi0_wGG, ch0_wxvG
        
    def calculate_dielectric_matrix(self, xc='RPA', q_c=[0, 0, 0],
                                    direction='x'):
        pd, chi0_wGG, chi0_wxvG = self.calculate_chi0(q_c)
        
        G_G = pd.G2_qG[0]**0.5  # |G+q|
        if G_G[0] == 0.0:
            G_G[0] = 1.0
            d_v = {'x': [1, 0, 0],
                   'y': [0, 1, 0],
                   'z': [0, 0, 1]}.get(direction, direction)
            d_v = np.asarray(d_v) / np.linaglg.norm(d_v)
            chi0_wGG[:, 0, 0] = np.dot(chi0_wxvG[:, 0, :, 0], d_v**2)
            chi0_wGG[:, 0] = np.dot(d_v, chi0_wxvG[:, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0_wxvG[:, 1])
            
        assert xc == 'RPA'

        nG = len(G_G)
        for chi0_GG in chi0_wGG:
            e_GG = np.eye(nG) - 4 * np.pi * chi0_GG / G_G / G_G[:, np.newaxis]
            chi0_GG[:] = e_GG

        return chi0_wGG

    def calculate_dielectric_function(self, xc='RPA', q_c=[0, 0, 0],
                                      direction='x'):
        """Calculate the dielectric function.

        Returns dielectric function without and with local field correction.
        """
        
        e_wGG = self.calculate_dielectric_matrix(xc, q_c, direction)
        df1_w = np.zeros(len(e_wGG), complex)
        df2_w = np.zeros(len(e_wGG), complex)
        for w, e_GG in enumerate(e_wGG):
            df1_w[w] = e_GG[0, 0]
            df2_w[w] = 1 / np.linalg.inv(e_GG)[0, 0]
        return df1_w, df2_w
