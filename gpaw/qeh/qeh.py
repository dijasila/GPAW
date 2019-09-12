from __future__ import print_function

from pathlib import Path
import pickle
import numpy as np
from math import pi
import ase.units
from ase.parallel import world
from ase.utils import devnull
from ase.utils.timing import timer, Timer
import scipy

import sys
import os

Hartree = ase.units.Hartree
Bohr = ase.units.Bohr

default_ehmasses = {'BPx': {'emass1': [0.17, 1.12], 'hmass1': [0.15, 6.35]},
                    'BPy': {'emass1': [0.17, 1.12], 'hmass1': [0.15, 6.35]},
                    'H-CrO2-NM': {'emass1': 0.875, 'hmass1': 1.442},
                    'H-CrS2-NM': {'emass1': 0.872, 'hmass1': 0.883},
                    'H-CrSe2-NM': {'emass1': 0.936, 'hmass1': 0.955},
                    'H-CrTe2-NM': {'emass1': 0.855, 'hmass1': 0.9},
                    'H-HfS2-NM': {'emass1': 1.255, 'hmass1': 2.653},
                    'H-HfSe2-NM': {'emass1': 1.351, 'hmass1': 3.108},
                    'H-HfTe2-NM': {'emass1': 1.722, 'hmass1': 0.612},
                    'H-MoO2-NM': {'emass1': 0.419, 'hmass1': 0.764},
                    'H-MoS2-NM': {'emass1': 0.427, 'hmass1': 0.53},
                    'MoSSe': {'emass1': 0.48, 'hmass1': 0.60},
                    'H-MoSe2-NM': {'emass1': 0.492, 'hmass1': 0.583},
                    'H-MoTe2-NM': {'emass1': 0.493, 'hmass1': 0.597},
                    'H-PbS2-NM': {'emass1': 0.386, 'hmass1': 0.618},
                    'H-PbSe2-NM': {'emass1': 0.281, 'hmass1': 0.418},
                    'H-PdSe2-NM': {'emass1': 1.241, 'hmass1': 0.333},
                    'H-ScO2-FM': {'emass1': 2.94, 'hmass1': 10.669},
                    'H-ScS2-FM': {'emass1': 4.018, 'hmass1': 5.235},
                    'H-ScSe2-FM': {'emass1': 0.0, 'hmass1': 4.261},
                    'H-SnO2-NM': {'emass1': 0.282, 'hmass1': 7.291},
                    'H-SnS2-NM': {'emass1': 0.656, 'hmass1': 0.482},
                    'H-TiS2-NM': {'emass1': 0.0, 'hmass1': 2.585},
                    'H-TiSe2-NM': {'emass1': 0.0, 'hmass1': 1.654},
                    'H-VSe2-FM': {'emass1': 2.157, 'hmass1': 0.801},
                    'H-VTe2-FM': {'emass1': 3.332, 'hmass1': 1.105},
                    'H-WO2-NM': {'emass1': 0.346, 'hmass1': 0.781},
                    'H-WS2-NM': {'emass1': 0.328, 'hmass1': 0.336},
                    'H-WSe2-NM': {'emass1': 0.389, 'hmass1': 0.355},
                    'H-WTe2-NM': {'emass1': 0.364, 'hmass1': 0.286},
                    'H-ZrS2-NM': {'emass1': 2.881, 'hmass1': 2.2},
                    'H-ZrSe2-NM': {'emass1': 0.0, 'hmass1': 1.973},
                    'H-ZrTe2-NM': {'emass1': 0.0, 'hmass1': 1.136},
                    'T-GeO2-NM': {'emass1': 0.344, 'hmass1': 3.79},
                    'T-GeS2-NM': {'emass1': 0.689, 'hmass1': 1.289},
                    'T-HfO2-NM': {'emass1': 3.18, 'hmass1': 2.767},
                    'T-HfS2-NM': {'emass1': 2.372, 'hmass1': 0.249},
                    'T-HfSe2-NM': {'emass1': 2.286, 'hmass1': 0.159},
                    'T-Mn2Cl4-FM': {'emass1': 1.746, 'hmass1': 6.806},
                    'T-MnO2-FM': {'emass1': 1.055, 'hmass1': 43.724},
                    'T-NiO2-NM': {'emass1': 2.007, 'hmass1': 0.0},
                    'T-NiS2-NM': {'emass1': 0.403, 'hmass1': 0.617},
                    'T-PbO2-NM': {'emass1': 0.45, 'hmass1': 29.088},
                    'T-PbS2-NM': {'emass1': 0.895, 'hmass1': 4.138},
                    'T-PbSe2-NM': {'emass1': 1.07, 'hmass1': 0.36},
                    'T-PdO2-NM': {'emass1': 3.069, 'hmass1': 0.0},
                    'T-PdS2-NM': {'emass1': 0.565, 'hmass1': 2.247},
                    'T-PdSe2-NM': {'emass1': 0.337, 'hmass1': 0.635},
                    'T-PtO2-NM': {'emass1': 3.288, 'hmass1': 28.946},
                    'T-PtS2-NM': {'emass1': 0.682, 'hmass1': 1.546},
                    'T-PtSe2-NM': {'emass1': 0.463, 'hmass1': 2.893},
                    'T-PtTe2-NM': {'emass1': 0.251, 'hmass1': 0.359},
                    'T-SnO2-NM': {'emass1': 0.355, 'hmass1': 4.491},
                    'T-SnS2-NM': {'emass1': 0.779, 'hmass1': 2.034},
                    'T-SnSe2-NM': {'emass1': 0.744, 'hmass1': 0.402},
                    'T-TiO2-NM': {'emass1': 1.214, 'hmass1': 3.834},
                    'T-ZrO2-NM': {'emass1': 1.384, 'hmass1': 3.017},
                    'T-ZrS2-NM': {'emass1': 2.04, 'hmass1': 0.26},
                    'T-ZrSe2-NM': {'emass1': 1.928, 'hmass1': 0.158}}

default_thicknesses = {'H-MoS2-icsd-644245': 6.1511,
                       'H-TaSe2-icsd-651948': 6.375,
                       #  'H-TaSe2-icsd-651950': 6.36,
                       'T-PdTe2-icsd-649016': 5.118,
                       'T-CrSe2-icsd-626718': 5.915,
                       'T-ZrTe2-icsd-653213': 6.66,
                       'T-VS2-icsd-651361': 5.755,
                       'T-TiSe2-icsd-173923': 6.01,
                       'H-NbSe2-icsd-645369': 6.27,
                       'T-CrTlTe2-icsd-152836': 7.9352,
                       'T-SnSe2-icsd-651910': 6.132,
                       'T-HfS2-icsd-638847': 5.837,
                       'T-VSe2-icsd-652158': 6.048,
                       'H-MoTe2-icsd-15431': 6.982,
                       'T-CoTe2-icsd-625401': 5.405,
                       'T-SnS2-icsd-650992': 5.9,
                       'T-TiTe2-icsd-653071': 6.48,
                       'T-ZrS2-icsd-651465': 5.813,
                       'T-TaS2-icsd-651089': 5.9,
                       'T-HfTe2-icsd-638959': 6.65,
                       'T-PtS2-icsd-649534': 5.0389,
                       'T-TiS2-icsd-651178': 5.7,
                       'T-NbTe2-icsd-645529': 6.61,
                       'T-HfSe2-icsd-638899': 6.159,
                       'T-IrTe2-icsd-33934': 5.404,
                       'T-RhTe2-icsd-650448': 5.442,
                       'H-MoSe2-icsd-644334': 6.45,
                       'T-SiTe2-icsd-652385': 6.71,
                       'H-WSe2-icsd-40752': 6.48,
                       'T-PtTe2-icsd-649747': 5.2209,
                       'T-VTe2-icsd-603582': 6.582,
                       'H-TaS2-icsd-651092': 6.05,
                       'H-WS2-icsd-202366': 6.1615,
                       'T-ZrSe2-icsd-652236': 6.128,
                       'T-PtSe2-icsd-649589': 5.0813,
                       'T-NiTe2-icsd-159382': 5.266,
                       'BPx': 5.26,
                       'BPy': 5.26,
                       'MoSSe': 6.32,
                       'MoSSePrime': 6.32,
                       'graphene': 3.35,  # Wiki
                       'BN': 3.33, # ioffe.ru/SVA/NSM/Semicond/BN/basic.html
                       'GaSe': 9.3} 


def load(fd):
    try:
        return pickle.load(fd, encoding='latin1')
    except TypeError:
        return pickle.load(fd)


class Heterostructure:
    """This class defines dielectric function of heterostructures
        and related physical quantities."""

    def __init__(self, structure, d, thicknesses=None,
                 include_dipole=True, d0=None,
                 wmax=10, qmax=None, timer=None, substrate=None):
        """Creates a Heterostructure object.

        structure: list of str
            Heterostructure set up. Each entry should consist of number of
            layers + chemical formula.
            For example: ['3H-MoS2', graphene', '10H-WS2'] gives 3 layers of
            H-MoS2, 1 layer of graphene and 10 layers of H-WS2.
        thicknesses: array of floats
            Layers thicknesses in Ang.
            Length of array = number of layers
        d: array of floats
            Interlayer distances for neighboring layers in Ang.
            Length of array = number of layers - 1
        d0: float
            The width of a single layer in Ang, only used for monolayer
            calculation. The layer separation in bulk is typically a good
            measure.
        include_dipole: Bool
            Includes dipole contribution if True
        wmax: float
            Cutoff for frequency grid (eV)
        qmax: float
            Cutoff for wave-vector grid (1/Ang)
        substrate: dictionary {'eps': eps_w,
                               'omega': omega_w,
                               'd': d,
                               'isotropic': Bool}
            eps: Dielectric function of the substrate (frequency dependent)
                (epsx and epsz in the anisotropic case)
            omega: Fequency grid of the eps (eV)
            d: Distance of the substrate to the middle of the
                first layer (Ang)
            isotropic: includes the out-of-plane dielectric function in
                the substrate if True
        """

        self.timer = timer or Timer()
        self.substrate = substrate
        chi_monopole = []
        drho_monopole = []
        if include_dipole:
            chi_dipole = []
            drho_dipole = []
        else:
            self.chi_dipole = None
            drho_dipole = None
        self.z = []
        layer_indices = []
        self.n_layers = 0
        namelist = []
        n_rep = 0

        for il, layer in enumerate(structure):
            append = '-chi.npz'
            if append not in layer:
                structure[il] = layer + append

        structure = expand_layers(structure)

        if not check_building_blocks(list(set(structure))):
            raise ValueError('Building Blocks not on the same grid')
        self.n_layers = len(structure)
        for n, name in enumerate(structure):
            if name not in namelist:
                namelist.append(name)
                data = np.load(name)
                q = data['q_abs']
                w = data['omega_w']
                zi = data['z']
                zi -= np.mean(zi)
                chim = data['chiM_qw']
                chid = data['chiD_qw']
                drhom = data['drhoM_qz']
                drhod = data['drhoD_qz']
                if qmax is not None:
                    qindex = np.argmin(abs(q - qmax * Bohr)) + 1
                else:
                    qindex = None
                if wmax is not None:
                    windex = np.argmin(abs(w - wmax / Hartree)) + 1
                else:
                    windex = None
                chi_monopole.append(np.array(chim[:qindex, :windex]))
                drho_monopole.append(np.array(drhom[:qindex]))
                if include_dipole:
                    chi_dipole.append(np.array(chid[:qindex, :windex]))
                    drho_dipole.append(np.array(drhod[:qindex]))
                self.z.append(np.array(zi))
                n -= n_rep
            else:
                n = namelist.index(name)
                n_rep += 1

            layer_indices = np.append(layer_indices, n)

        self.layer_indices = np.array(layer_indices, dtype=int)

        self.q_abs = q[:qindex]
        if self.q_abs[0] == 0:
            self.q_abs[0] += 1e-12

        # parallelize over q in case of multiple processors
        from ase.parallel import world
        self.world = world
        nq = len(self.q_abs)
        mynq = (nq + self.world.size - 1) // self.world.size
        self.q1 = min(mynq * self.world.rank, nq)
        self.q2 = min(self.q1 + mynq, nq)
        self.mynq = self.q2 - self.q1
        self.myq_abs = self.q_abs[self.q1: self.q2]

        self.frequencies = w[:windex]
        self.n_types = len(namelist)

        chi_monopole = np.array(chi_monopole)[:, self.q1: self.q2]
        chi_dipole = np.array(chi_dipole)[:, self.q1: self.q2]
        for i in range(self.n_types):
            drho_monopole[i] = np.array(drho_monopole[i])[self.q1: self.q2]
            drho_dipole[i] = np.array(drho_dipole[i])[self.q1: self.q2]

        # layers and distances
        self.d = np.array(d) / Bohr  # interlayer distances
        if self.n_layers > 1:
            # space around each layer
            if thicknesses is not None:
                self.s = thicknesses / Bohr
            else:
                self.s = (np.insert(self.d, 0, self.d[0]) +
                          np.append(self.d, self.d[-1])) / 2.

        else:  # Monolayer calculation
            self.s = [d0 / Bohr]  # Width of single layer
        self.d0 = d0 / Bohr

        self.dim = np.copy(self.n_layers)
        if include_dipole:
            self.dim *= 2

        # Grid stuff
        edgesize = 50
        system_size = np.sum(self.d) + edgesize
        self.poisson_lim = 1000  # above this limit use potential model
        assert system_size < self.poisson_lim

        self.z_lim = system_size
        self.dz = 0.05
        # master grid
        self.z_big = np.arange(0, self.z_lim, self.dz) - edgesize / 2.
        self.z0 = np.append(np.array([0]), np.cumsum(self.d))

        # arange potential and density
        self.chi_monopole = chi_monopole
        if include_dipole:
            self.chi_dipole = chi_dipole
        self.drho_monopole, self.drho_dipole, self.basis_array, \
            self.drho_array = self.arange_basis(drho_monopole, drho_dipole)

        self.dphi_array = self.get_induced_potentials()
        self.kernel_qij = None

        if self.substrate is not None:
            self.kernelsub_qwij = None

    @timer('Arange basis')
    def arange_basis(self, drhom, drhod=None):
        from scipy.interpolate import interp1d
        Nz = len(self.z_big)
        drho_array = np.zeros([self.dim, self.mynq,
                               Nz], dtype=complex)
        basis_array = np.zeros([self.dim, self.mynq,
                                Nz], dtype=complex)

        for i in range(self.n_types):
            z = self.z[i]
            drhom_i = drhom[i]
            fm = interp1d(z, np.real(drhom_i))
            fm2 = interp1d(z, np.imag(drhom_i))
            if drhod is not None:
                drhod_i = drhod[i]
                fd = interp1d(z, np.real(drhod_i))
                fd2 = interp1d(z, np.imag(drhod_i))
            for k in [k for k in range(self.n_layers)
                      if self.layer_indices[k] == i]:
                z_big = self.z_big - self.z0[k]
                i_1s = np.argmin(np.abs(-self.s[k] / 2. - z_big))
                i_2s = np.argmin(np.abs(self.s[k] / 2. - z_big))

                i_1 = np.argmin(np.abs(z[0] - z_big))
                if z_big[i_1] < z[0]:
                    i_1 += 1
                i_2 = np.argmin(np.abs(z[-1] - z_big))
                if z_big[i_2] > z[-1]:
                    i_2 -= 1
                if drhod is not None:
                    drho_array[2 * k, :, i_1: i_2 + 1] = \
                        fm(z_big[i_1: i_2 + 1]) + 1j * fm2(z_big[i_1: i_2 + 1])
                    basis_array[2 * k, :, i_1s: i_2s] = 1. / self.s[k]
                    drho_array[2 * k + 1, :, i_1: i_2 + 1] = \
                        fd(z_big[i_1: i_2 + 1]) + 1j * fd2(z_big[i_1: i_2 + 1])
                    basis_array[2 * k + 1, :, i_1: i_2] = \
                        fd(z_big[i_1: i_2]) + 1j * fd2(z_big[i_1: i_2])
                else:
                    drho_array[k, :, i_1: i_2 + 1] = \
                        fm(z_big[i_1: i_2 + 1]) + 1j * fm2(z_big[i_1: i_2 + 1])
                    basis_array[k, :, i_1s:i_2s] = 1. / self.s[k]

        return drhom, drhod, basis_array, drho_array

    @timer('Get induced potentials')
    def get_induced_potentials(self):
        from scipy.interpolate import interp1d
        Nz = len(self.z_big)
        dphi_array = np.zeros([self.dim, self.mynq, Nz], dtype=complex)

        for i in range(self.n_types):
            z = self.z[i]
            for iq in range(self.mynq):
                q = self.myq_abs[iq]
                drho_m = self.drho_monopole[i][iq].copy()
                poisson_m = self.solve_poisson_1D(drho_m, q, z)
                z_poisson = self.get_z_grid(z, z_lim=self.poisson_lim)
                assert len(z_poisson) == len(np.real(poisson_m))
                fm = interp1d(z_poisson, np.real(poisson_m))
                fm2 = interp1d(z_poisson, np.imag(poisson_m))
                if self.chi_dipole is not None:
                    drho_d = self.drho_dipole[i][iq].copy()
                    #  delta is the distance between dipole peaks / 2
                    delta = np.abs(z[np.argmax(drho_d)] -
                                   z[np.argmin(drho_d)]) / 2.
                    poisson_d = self.solve_poisson_1D(drho_d, q, z,
                                                      dipole=True,
                                                      delta=delta)
                    fd = interp1d(z_poisson, np.real(poisson_d))
                    fd2 = interp1d(z_poisson, np.imag(poisson_d))

                for k in [k for k in range(self.n_layers)
                          if self.layer_indices[k] == i]:
                    z_big = self.z_big - self.z0[k]
                    i_1 = np.argmin(np.abs(z_poisson[0] - z_big))
                    if z_big[i_1] < z_poisson[0]:
                        i_1 += 1
                    i_2 = np.argmin(np.abs(z_poisson[-1] - z_big))
                    if z_big[i_2] > z_poisson[-1]:
                        i_2 -= 1

                    dphi_array[self.dim // self.n_layers * k, iq] = (
                        self.potential_model(self.myq_abs[iq], self.z_big,
                                             self.z0[k]))
                    dphi_array[self.dim // self.n_layers * k,
                               iq, i_1: i_2 + 1] = (fm(z_big[i_1: i_2 + 1]) +
                                                    1j * fm2(z_big[i_1: i_2 + 1]))
                    if self.chi_dipole is not None:
                        dphi_array[2 * k + 1, iq] = \
                            self.potential_model(self.myq_abs[iq], self.z_big,
                                                 self.z0[k], dipole=True,
                                                 delta=delta)
                        dphi_array[2 * k + 1, iq, i_1: i_2 + 1] = \
                            fd(z_big[i_1: i_2 + 1]) + 1j * \
                            fd2(z_big[i_1: i_2 + 1])

        return dphi_array
    
    def get_z_grid(self, z, z_lim=None):
        dz = z[1] - z[0]
        if z_lim is None:
            z_lim = self.z_lim

        z_lim = int(z_lim / dz) * dz
        z_grid = np.insert(z, 0, np.flip(
            np.arange(z[0] - dz, -z_lim - dz, -dz)))
        z_grid = np.append(z_grid, np.arange(z[-1] + dz, z_lim + dz, dz))
        return z_grid

    @timer('Potential model')
    def potential_model(self, q, z, z0=0, dipole=False, delta=None):
        """
        2D Coulomb: 2 pi / q with exponential decay in z-direction
        """
        if dipole:  # Two planes separated by 2 * delta
            V = np.pi / (q * delta) * \
                (-np.exp(-q * np.abs(z - z0 + delta)) +
                 np.exp(-q * np.abs(z - z0 - delta)))
        else:  # Monopole potential from single plane
            V = 2 * np.pi / q * np.exp(-q * np.abs(z - z0))

        return V

    @timer('Solve Poisson equation')
    def solve_poisson_1D(self, drho, q, z,
                         dipole=False, delta=None):
        """
        Solves poissons equation in 1D using finite difference method.

        drho: induced potential basis function
        q: momentum transfer.
        """
        dz = z[1] - z[0]
        z_grid = self.get_z_grid(z, z_lim=self.poisson_lim)

        assert np.allclose(np.diff(z_grid), dz)
        Nz_loc = (len(z_grid) - len(z)) // 2

        drho = np.append(np.insert(drho, 0, np.zeros([Nz_loc])),
                         np.zeros([Nz_loc]))
        Nint = len(drho) - 1

        bc_v0 = self.potential_model(q, z_grid[0], dipole=dipole,
                                     delta=delta)
        bc_vN = self.potential_model(q, z_grid[-1], dipole=dipole,
                                     delta=delta)
        f_z = np.zeros(Nint + 1, dtype=complex)
        f_z[:] = - 4 * np.pi * drho[:]
        f_z[0] = bc_v0
        f_z[Nint] = bc_vN
                                                  
        ab = np.zeros((3, Nint + 1), complex)
        ab[0, 2:] = 1. / dz**2
        ab[1, 1:-1] = -2 / dz**2 - q**2
        ab[1, 0] = 1
        ab[1, -1] = 1
        ab[2, :-2] = 1. / dz**2
        dphi = scipy.linalg.solve_banded((1, 1), ab, f_z)
        return dphi

    @timer('Get coulomb kernel')
    def get_Coulomb_Kernel(self, step_potential=False):
        kernel_qij = np.zeros([self.mynq, self.dim, self.dim],
                              dtype=complex)

        for iq in range(self.mynq):
            if np.isclose(self.myq_abs[iq], 0):
                # Special treatment of q=0 limit
                kernel_qij[iq] = np.eye(self.dim)
            else:
                if step_potential:
                    # Use step-function average for monopole contribution
                    kernel_qij[iq] = np.dot(self.basis_array[:, iq],
                                            self.dphi_array[:, iq].T) * self.dz
                else:
                    kernel_qij[iq] = np.dot(self.drho_array[:, iq],
                                            self.dphi_array[:, iq].T) * self.dz
        return kernel_qij

    @timer('Get substrate coulomb kernel')
    def get_substrate_Coulomb_Kernel(self, step_potential=False):
        from scipy.interpolate import interp1d
        kernelsub_qij = np.zeros([self.mynq, self.dim, self.dim],
                                 dtype=complex)

        subw_w = self.substrate['omega']
        if self.substrate['isotropic']:
            subeps_w = self.substrate['eps']
            fsubreal_w = interp1d(subw_w, np.real(subeps_w),
                                  bounds_error=False, fill_value='extrapolate')
            fsubimag_w = interp1d(subw_w, np.imag(subeps_w),
                                  bounds_error=False, fill_value='extrapolate')
            newsubeps_w = fsubreal_w(self.frequencies * Hartree) + \
                1j * fsubimag_w(self.frequencies * Hartree)
        else:
            subepsx_w = self.substrate['epsx']
            subepsz_w = self.substrate['epsz']
            fsubxreal_w = interp1d(subw_w, np.real(subepsx_w),
                                   bounds_error=False,
                                   fill_value='extrapolate')
            fsubximag_w = interp1d(subw_w, np.imag(subepsx_w),
                                   bounds_error=False,
                                   fill_value='extrapolate')
            fsubzreal_w = interp1d(subw_w, np.real(subepsz_w),
                                   bounds_error=False,
                                   fill_value='extrapolate')
            fsubzimag_w = interp1d(subw_w, np.imag(subepsz_w),
                                   bounds_error=False,
                                   fill_value='extrapolate')
            newsubepsx_w = fsubxreal_w(self.frequencies * Hartree) + \
                1j * fsubximag_w(self.frequencies * Hartree)
            newsubepsz_w = fsubzreal_w(self.frequencies * Hartree) + \
                1j * fsubzimag_w(self.frequencies * Hartree)
            newsubeps_w = np.sqrt(newsubepsx_w * newsubepsz_w)

        fphi_real = interp1d(self.z_big, np.real(self.dphi_array),
                             bounds_error=False, fill_value=0, axis=-1)
        fphi_imag = interp1d(self.z_big, np.imag(self.dphi_array),
                             bounds_error=False, fill_value=0, axis=-1)

        if self.substrate['d']:
            subdist = self.substrate['d'][0] / Bohr
        else:
            subdist = self.d0 / 2

        dphi_sub = (fphi_real(self.z_big - 2 * subdist) +
                    1j * fphi_imag(self.z_big - 2 * subdist))[:, :, ::-1]

        for iq in range(self.mynq):
            if step_potential:
                kernelsub_qij[iq] = np.dot(self.basis_array[:, iq],
                                           dphi_sub[:, iq].T)
            else:
                kernelsub_qij[iq] = np.dot(self.drho_array[:, iq],
                                           dphi_sub[:, iq].T)

        kernelsub_qwij = (-((newsubeps_w - 1) / (newsubeps_w + 1))[None, :,
                                                                   None, None]
                          * self.dz) * kernelsub_qij[:, None]

        return kernelsub_qwij.real

    @timer('Solve chi dyson equation')
    def get_chi_matrix(self):
        """
        Dyson equation: chi_full = chi_intra + chi_intra V_inter chi_full
        """
        print('Calculating full chi')
        Nls = self.n_layers
        q_abs = self.myq_abs
        chi_m_iqw = self.chi_monopole
        chi_d_iqw = self.chi_dipole

        if self.kernel_qij is None:
            self.kernel_qij = self.get_Coulomb_Kernel()

        if self.substrate is not None:
            if self.kernelsub_qwij is None:
                self.kernelsub_qwij = self.get_substrate_Coulomb_Kernel()

        chi_qwij = np.zeros((self.mynq, len(self.frequencies),
                             self.dim, self.dim), dtype=complex)
        
        dot = np.dot
        inv = np.linalg.inv
        eye = np.eye(self.dim)
        nq = len(q_abs)
        for iq in range(nq):
            if (1 + iq) % (nq // 10) == 0:
                print('{}%'.format(np.round((1 + iq) / nq, 1) * 100))

            kernel_ij = self.kernel_qij[iq].copy()
            np.fill_diagonal(kernel_ij, 0)  # Diagonal is set to zero

            if self.chi_dipole is not None:
                for j in range(self.n_layers):
                    kernel_ij[2 * j, 2 * j + 1] = 0
                    kernel_ij[2 * j + 1, 2 * j] = 0

            chi_intra_wij = np.zeros((len(self.frequencies), self.dim,
                                      self.dim),
                                     dtype=complex)
            for iw in range(0, len(self.frequencies)):
                chi_intra_i = chi_m_iqw[self.layer_indices, iq, iw]
                if self.chi_dipole is not None:
                    chi_intra_i = np.insert(chi_intra_i, np.arange(Nls) + 1,
                                            chi_d_iqw[self.layer_indices,
                                                      iq, iw])
                chi_intra_wij[iw] = np.diag(chi_intra_i)

                if self.substrate is not None:
                    kernelsub_ij = self.kernelsub_qwij[iq, iw].copy()
                    newkernel_ij = kernel_ij + kernelsub_ij

                    chi_qwij[iq, iw] = inv(eye - dot(chi_intra_wij[iw],
                                                     newkernel_ij))

            if self.substrate is None:
                chi_qwij[iq] = inv(eye - dot(chi_intra_wij, kernel_ij))

            for chi_ij, chi_intra_ij in zip(chi_qwij[iq], chi_intra_wij):
                chi_ij[:, :] = dot(chi_ij, chi_intra_ij)

        return chi_qwij

    @timer('Get eps matrix')
    def get_eps_matrix(self, step_potential=False, iq_q=None, iw_w=None):
        """
        Get dielectric matrix as: eps^{-1} = 1 + V chi_full
        """

        if iq_q is None:
            iq_q = np.arange(self.mynq)

        if iw_w is None:
            iw_w = np.arange(len(self.frequencies))

        nq = len(iq_q)
        nw = len(iw_w)
        if self.kernel_qij is None:
            sp = step_potential
            self.kernel_qij = self.get_Coulomb_Kernel(step_potential=sp)

            if self.substrate is not None:
                if self.kernelsub_qwij is None:
                    self.kernelsub_qwij = \
                        self.get_substrate_Coulomb_Kernel(step_potential=sp)

            chi_qwij = self.get_chi_matrix()
            self.chi_qwij = chi_qwij
        else:
            chi_qwij = self.chi_qwij

        eps_qwij = np.zeros((nq, nw,
                             self.dim, self.dim), dtype=complex)

        for i, iq in enumerate(iq_q):
            kernel_ij = self.kernel_qij[iq]
            if self.substrate is not None:
                for j, iw in enumerate(iw_w):
                    newkernel_ij = kernel_ij + self.kernelsub_qwij[iq, iw]
                    eps_qwij[i, j, :, :] = np.linalg.inv(
                        np.eye(newkernel_ij.shape[0]) + np.dot(newkernel_ij,
                                                               chi_qwij[iq, iw,
                                                                        :, :]))

            else:
                for j, iw in enumerate(iw_w):
                    eps_qwij[i, j, :, :] = np.linalg.inv(
                        np.eye(kernel_ij.shape[0]) + np.dot(kernel_ij,
                                                            chi_qwij[iq, iw,
                                                                     :, :]))

        return eps_qwij

    @timer('Get screened potential')
    def get_screened_potential(self, layer=0, step_potential=False,
                               subtract_bare_coulomb=False):
        r"""
        get the screened interaction averaged over layer "k":
        W_{kk}(q, w) = \sum_{ij} V_{ki}(q) \chi_{ij}(q, w) V_{jk}(q)

        parameters:
        layer: int
            index of layer to calculate the screened interaction for.

        returns: W(q,w)
        """
        self.kernel_qij =\
            self.get_Coulomb_Kernel(step_potential=step_potential)

        if self.substrate is not None:
            sp = step_potential
            self.kernelsub_qwij = \
                self.get_substrate_Coulomb_Kernel(step_potential=sp)

        chi_qwij = self.get_chi_matrix()
        W_qw = np.zeros((self.mynq, len(self.frequencies)),
                        dtype=complex)

        k = layer
        if self.chi_dipole is not None:
            k *= 2
        for iq in range(self.mynq):
            kernel_ij = self.kernel_qij[iq].copy()

            # if np.isclose(self.myq_abs[iq], 0):
            #     kernel_ij = 2 * np.pi * np.ones([self.dim, self.dim])

            # if self.chi_dipole is not None:
            #     for j in range(self.n_layers):
            #         kernel_ij[2 * j, 2 * j + 1] = 0
            #         kernel_ij[2 * j + 1, 2 * j] = 0

            for iw in range(0, len(self.frequencies)):
                if self.substrate is not None:
                    newkernel_ij = kernel_ij + \
                        self.kernelsub_qwij[iq, iw].copy()
                    W_qw[iq, iw] = np.dot(np.dot(newkernel_ij[k],
                                                 chi_qwij[iq, iw]),
                                          newkernel_ij[:, k])
                    V0 = newkernel_ij[k, k]
                else:
                    W_qw[iq, iw] = np.dot(np.dot(kernel_ij[k],
                                                 chi_qwij[iq, iw]),
                                          kernel_ij[:, k])
                    V0 = kernel_ij[k, k]

                if subtract_bare_coulomb is False:
                    W_qw[iq, iw] += V0

        W_qw = self.collect_qw(W_qw)

        return W_qw

    @timer('Get excition screened potential')
    def get_exciton_screened_potential(self, e_distr, h_distr):
        v_screened_q = np.zeros(self.mynq)
        eps_qwij = self.get_eps_matrix()
        h_distr = h_distr.transpose()
        kernel_qij = self.get_Coulomb_Kernel()

        if self.substrate is not None:
            if self.kernelsub_qwij is None:
                self.kernelsub_qwij = self.get_substrate_Coulomb_Kernel()
            kernel_qij += self.kernelsub_qwij[:, 0]

        for iq in range(0, self.mynq):
            ext_pot = np.dot(kernel_qij[iq], h_distr)
            v_screened_q[iq] =\
                np.dot(e_distr,
                       np.dot(np.linalg.inv(eps_qwij[iq, 0, :, :]),
                              ext_pot)).real

        if self.substrate is not None:
            kernel_qij -= self.kernelsub_qwij[:, 0]

        v_screened_q = self.collect_q(v_screened_q[:])

        return self.q_abs, -v_screened_q, kernel_qij

    def get_exciton_screened_potential_r(self, r_array, e_distr=None,
                                         h_distr=None, Wq_name=None,
                                         tweak=None):
        if Wq_name is not None:
            q_abs, W_q = load(open(Wq_name, 'rb'))
        else:
            q_temp, W_q, xxx = self.get_exciton_screened_potential(e_distr,
                                                                   h_distr)
 
        from scipy.special import jn

        inter = False
        if np.where(e_distr == 1)[0][0] != np.where(h_distr == 1)[0][0]:
            inter = True

        layer_dist = 0.
        if self.n_layers == 1:
            layer_thickness = self.s[0]
        else:
            if len(e_distr) == self.n_layers:
                div = 1
            else:
                div = 2

            if not inter:
                ilayer = np.where(e_distr == 1)[0][0] // div
                if ilayer == len(self.d):
                    ilayer -= 1
                layer_thickness = self.d[ilayer]
            else:
                ilayer1 = np.min([np.where(e_distr == 1)[0][0],
                                  np.where(h_distr == 1)[0][0]]) // div
                ilayer2 = np.max([np.where(e_distr == 1)[0][0],
                                  np.where(h_distr == 1)[0][0]]) // div
                layer_thickness = self.d[ilayer1] / 2.
                layer_dist = np.sum(self.d[ilayer1:ilayer2]) / 1.8
        if tweak is not None:
            layer_thickness = tweak

        W_q *= q_temp
        q = np.linspace(q_temp[0], q_temp[-1], 10000)
        Wt_q = np.interp(q, q_temp, W_q)
        Dq_Q2D = q[1] - q[0]

        if not inter:
            Coulombt_q = (-4. * np.pi / q *
                          (1. - np.exp(-q * layer_thickness / 2.)) /
                          layer_thickness)
        else:
            Coulombt_q = (-2 * np.pi / q *
                          (np.exp(-q * (layer_dist - layer_thickness / 2.)) -
                           np.exp(-q * (layer_dist + layer_thickness / 2.))) /
                          layer_thickness)

        W_r = np.zeros(len(r_array))
        for ir in range(len(r_array)):
            J_q = jn(0, q * r_array[ir])
            if r_array[ir] > np.exp(-13):
                Int_temp = (
                    -1. / layer_thickness *
                    np.log((layer_thickness / 2. - layer_dist +
                            np.sqrt(r_array[ir]**2 +
                                    (layer_thickness / 2. - layer_dist)**2)) /
                           (-layer_thickness / 2. - layer_dist +
                            np.sqrt(r_array[ir]**2 +
                                    (layer_thickness / 2. + layer_dist)**2))))
            else:
                if not inter:
                    Int_temp = (-1. / layer_thickness *
                                np.log(layer_thickness**2 / r_array[ir]**2))
                else:
                    Int_temp = (-1. / layer_thickness *
                                np.log((layer_thickness + 2 * layer_dist) /
                                       (2 * layer_dist - layer_thickness)))

            W_r[ir] = (Dq_Q2D / 2. / np.pi *
                       np.sum(J_q * (Wt_q - Coulombt_q)) +
                       Int_temp)

        return r_array, W_r

    def get_exciton_binding_energies(self, eff_mass, L_min=-50, L_max=10,
                                     Delta=0.1, e_distr=None, h_distr=None,
                                     Wq_name=None, tweak=None):
        from scipy.linalg import eig
        r_space = np.arange(L_min, L_max, Delta)
        Nint = len(r_space)

        r, W_r = self.get_exciton_screened_potential_r(r_array=np.exp(r_space),
                                                       e_distr=e_distr,
                                                       h_distr=h_distr,
                                                       Wq_name=None,
                                                       tweak=tweak)

        H = np.zeros((Nint, Nint), dtype=complex)
        for i in range(0, Nint):
            r_abs = np.exp(r_space[i])
            H[i, i] = - 1. / r_abs**2 / 2. / eff_mass \
                * (-2. / Delta**2 + 1. / 4.) + W_r[i]
            if i + 1 < Nint:
                H[i, i + 1] = -1. / r_abs**2 / 2. / eff_mass \
                    * (1. / Delta**2 - 1. / 2. / Delta)
            if i - 1 >= 0:
                H[i, i - 1] = -1. / r_abs**2 / 2. / eff_mass \
                    * (1. / Delta**2 + 1. / 2. / Delta)

        ee, ev = eig(H)
        index_sort = np.argsort(ee.real)
        ee = ee[index_sort]
        ev = ev[:, index_sort]
        return ee * Hartree, ev

    def get_macroscopic_dielectric_function(self, static=True, direction='x'):
        """
        Calculates the averaged dielectric function over the structure.

        Parameters:

        static: bool
            If True only include w=0

        direction: str 'x' or 'z'
            'x' for in plane dielectric function
            'z' for out of plane dielectric function

        Returns list of q-points, frequencies, dielectric function(q, w).
        """
        assert self.substrate is None, \
            print('get_macroscopic_dielectric_function does not '
                  'apply for structures with substrate')
        if direction == 'x':
            const_per = np.ones([self.n_layers])
            if self.chi_dipole is not None:
                const_per = np.insert(const_per, np.arange(self.n_layers) + 1,
                                      np.zeros([self.n_layers]))

        elif direction == 'z':
            const_per = np.zeros([self.n_layers])
            assert self.chi_dipole is not None
            const_per = np.insert(const_per, np.arange(self.n_layers) + 1,
                                  np.ones([self.n_layers]))

        if static:
            Nw = 1
        else:
            Nw = len(self.frequencies)

        eps_qwij = self.get_eps_matrix(step_potential=True)[:, :Nw]
        
        N = self.n_layers
        Nq = self.mynq
        epsM_qw = np.zeros([Nq, Nw], dtype=complex)
        
        for iw in range(Nw):
            for iq in range(Nq):
                eps_ij = eps_qwij[iq, iw]
                epsinv_ij = np.linalg.inv(eps_ij)
                epsinvM = 1 / N * np.dot(np.array(const_per),
                                         np.dot(epsinv_ij,
                                                np.array(const_per)))

                epsM_qw[iq, iw] = 1. / epsinvM

        epsM_qw = self.collect_qw(epsM_qw)

        return self.q_abs / Bohr, self.frequencies[:Nw] * Hartree, epsM_qw

    def get_dielectric_function(self, layer=0):
        """
        Calculates the dielectric function of the chosen layer

        Parameters:
        layer: int
            index of layer to calculate the dielectric function for.

        Returns list of q-points, frequencies, dielectric function(q, w).
        """

        W_qw = self.get_screened_potential(layer=layer)
        kernel_qij = self.get_Coulomb_Kernel()

        k = layer
        if self.chi_dipole is not None:
            k *= 2

        kernel_qw = kernel_qij[:, None, k, k]
        eps_qw = kernel_qw / W_qw
            
        return self.q_abs / Bohr, self.frequencies * Hartree, eps_qw


    def get_eels(self, dipole_contribution=False):
        r"""
        Calculates Electron energy loss spectrum, defined as:

        EELS(q, w) = - Im 4 \pi / q**2 \chiM(q, w)

        Returns list of q-points, Frequencies and the loss function
        """
        const_per = np.ones([self.n_layers])
        layer_weight = self.s / np.sum(self.s) * self.n_layers

        if self.chi_dipole is not None:
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.zeros([self.n_layers]))
            layer_weight = np.insert(layer_weight,
                                     np.arange(self.n_layers) + 1,
                                     layer_weight)

        if dipole_contribution:
            const_per = np.zeros([self.n_layers])
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.ones([self.n_layers]))

        N = self.n_layers
        eels_qw = np.zeros([self.mynq, len(self.frequencies)],
                           dtype=complex)

        chi_qwij = self.get_chi_matrix()

        for iq in range(self.mynq):
            for iw in range(len(self.frequencies)):
                eels_qw[iq, iw] = np.dot(np.array(const_per) * layer_weight,
                                         np.dot(chi_qwij[iq, iw],
                                                np.array(const_per)))

            eels_qw[iq, :] *= 1. / N * 4 * np.pi / self.q_abs[iq]**2

        eels_qw = self.collect_qw(eels_qw)

        return self.q_abs / Bohr, self.frequencies * Hartree, \
            - (Bohr * eels_qw).imag

    def get_absorption_spectrum(self, dipole_contribution=False):
        r"""
        Calculates absorption spectrum, defined as:

        ABS(q, w) = - Im 2 / q \epsM(q, w)

        Returns list of q-points, Frequencies and the loss function
        """
        const_per = np.ones([self.n_layers])
        layer_weight = self.s / np.sum(self.s) * self.n_layers

        if self.chi_dipole is not None:
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.zeros([self.n_layers]))
            layer_weight = np.insert(layer_weight,
                                     np.arange(self.n_layers) + 1,
                                     layer_weight)

        if dipole_contribution:
            const_per = np.zeros([self.n_layers])
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.ones([self.n_layers]))

        N = self.n_layers
        abs_qw = np.zeros([self.mynq, len(self.frequencies)],
                          dtype=complex)

        eps_qwij = self.get_eps_matrix()

        for iq in range(self.mynq):
            for iw in range(len(self.frequencies)):
                abs_qw[iq, iw] = np.dot(np.array(const_per) * layer_weight,
                                        np.dot(eps_qwij[iq, iw],
                                               np.array(const_per)))

            abs_qw[iq, :] *= 1. / N * 2. / self.q_abs[iq]

        abs_qw = self.collect_qw(abs_qw)
        return self.q_abs / Bohr, self.frequencies * Hartree, \
            (Bohr * abs_qw).imag

    def get_sum_eels(self, V_beam=100, include_z=False):
        r"""
        Calculates the q- averaged Electron energy loss spectrum usually
        obtained in scanning transmission electron microscopy (TEM).

        EELS(w) = - Im [sum_{q}^{q_max}  V(q) \chi(w, q) V(q)]
                    \delta(w - q \dot v_e)

        The calculation assumes a beam in the z-direction perpendicular to the
        layers, and that the response in isotropic within the plane.

        Input parameters:
        V_beam: float
            Acceleration voltage of electron beam in kV.
            Is used to calculate v_e that goes into \delta(w - q \dot v_e)

        Returns list of Frequencies and the loss function
        """
        const_per = np.ones([self.n_layers])
        layer_weight = self.s / np.sum(self.s) * self.n_layers

        if self.chi_dipole is not None:
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.zeros([self.n_layers]))
            layer_weight = np.insert(layer_weight,
                                     np.arange(self.n_layers) + 1,
                                     layer_weight)

        eels_w = np.zeros([len(self.frequencies)], dtype=complex)
        chi_qwij = self.get_chi_matrix()
        vol = np.pi * (self.q_abs[-1] + self.q_abs[1] / 2.)**2
        weight0 = np.pi * (self.q_abs[1] / 2.)**2 / vol
        c = (1 - weight0) / np.sum(self.q_abs)
        weights = c * self.q_abs
        weights[0] = weight0
        # Beam speed from relativistic eq
        me = ase.units._me
        c = ase.units._c
        E_0 = me * c**2  # Rest energy
        E = E_0 + V_beam * 1e3 / ase.units.J   # Relativistic energy
        v_e = c * (E**2 - E_0**2)**0.5 / E  # beam velocity in SI
        # Lower cutoff q_z = w / v_e
        w_wSI = self.frequencies * Hartree \
            / ase.units.J / ase.units._hbar  # w in SI units
        q_z = w_wSI / v_e / ase.units.m * Bohr  # in Bohr
        q_z[0] = q_z[1]
        print('Using a beam acceleration voltage of V = %3.1f kV' % (V_beam))
        print('Beam speed = %1.2f / c' % (v_e / c))
        # Upper cutoff q_c = q[1] / 2.
        q_c = self.q_abs[1] / 2.
        # Integral for q=0: \int_0^q_c \frac{q^3}{(q^2 + q_z^2)^2} dq
        I = 2 * np.pi / vol * \
            (q_z**2 / 2. / (q_c**2 + q_z**2) - 0.5 +
             0.5 * np.log((q_c / q_z)**2 + 1))
        I2 = 2 * np.pi / vol / 2. * (1. / q_z**2 - 1. / (q_z**2 + q_c**2))

        for iq in range(self.mynq):
            eels_temp = np.zeros([len(self.frequencies)], dtype=complex)
            for iw in range(len(self.frequencies)):
                # Longitudinal in-plane
                temp = np.dot(np.array(const_per) * layer_weight,
                              np.dot(chi_qwij[iq, iw], np.array(const_per)))
                eels_temp[iw] += temp

            if np.isclose(self.q_abs[iq], 0):
                eels_temp *= (4 * np.pi)**2 * I

            else:
                eels_temp *= 1. / (self.q_abs[iq]**2 + q_z**2)**2
                eels_temp *= (4 * np.pi)**2 * weights[iq]
            eels_w += eels_temp

            if include_z:
                eels_temp = np.zeros([len(self.frequencies)], dtype=complex)
                for iw in range(len(self.frequencies)):
                    # longitudinal out of plane
                    temp = np.dot(np.array(const_per[::-1]) * layer_weight,
                                  np.dot(chi_qwij[iq, iw],
                                         np.array(const_per[::-1])))
                    eels_temp[iw] += temp

                    # longitudinal cross terms
                    temp = 1J * np.dot(np.array(const_per) * layer_weight,
                                       np.dot(chi_qwij[iq, iw],
                                              np.array(const_per[::-1])))
                    eels_temp[iw] += temp / q_z[iw]

                    temp = -1J * np.dot(np.array(const_per[::-1]) *
                                        layer_weight,
                                        np.dot(chi_qwij[iq, iw],
                                               np.array(const_per)))
                    eels_temp[iw] += temp / q_z[iw]

                    # Transversal
                    temp = np.dot(np.array(const_per[::-1]) * layer_weight,
                                  np.dot(chi_qwij[iq, iw],
                                         np.array(const_per[::-1])))
                    temp *= (v_e / c)**4
                    eels_temp[iw] += temp

                if np.isclose(self.q_abs[iq], 0):
                    eels_temp *= (4 * np.pi)**2 * I2 * q_z**2
                else:
                    eels_temp *= 1. / (self.q_abs[iq]**2 + q_z**2)**2 * q_z**2
                    eels_temp *= (4 * np.pi)**2 * weights[iq]

                eels_w += eels_temp

        self.world.sum(eels_w)

        return self.frequencies * Hartree, - (Bohr**5 * eels_w * vol).imag

    def get_response(self, iw=0, dipole=False):
        r"""
        Get the induced density and potential due to constant perturbation
        obtained as: rho_ind(r) = int chi(r,r') dr'
        """
        const_per = np.ones([self.n_layers])
        if self.chi_dipole is not None:
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.zeros([self.n_layers]))

        if dipole:
            const_per = self.z0 - self.z0[-1] / 2.
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.ones([self.n_layers]))

        chi_qij = self.get_chi_matrix()[:, iw]
        Vind_qz = np.zeros((self.mynq, len(self.z_big)))
        rhoind_qz = np.zeros((self.mynq, len(self.z_big)))

        drho_array = self.drho_array.copy()
        dphi_array = self.dphi_array.copy()
        # Expand on potential and density basis function
        # to get spatial detendence
        for iq in range(self.mynq):
            chi_ij = chi_qij[iq]
            Vind_qi = np.dot(chi_ij, np.array(const_per))
            rhoind_qz[iq] = np.dot(drho_array[:, iq].T, Vind_qi)
            Vind_qz[iq] = np.dot(dphi_array[:, iq].T, Vind_qi)

        rhoind_qz = self.collect_qw(rhoind_qz)
        return self.z_big * Bohr, rhoind_qz, Vind_qz, self.z0 * Bohr

    @timer('Calculate plasmon eigenmodes')
    def get_plasmon_eigenmodes(self, filename=None):
        """
        Diagonalize the dieletric matrix to get the plasmon eigenresonances
        of the system.

        Returns:
            Eigenvalue array (shape Nq x nw x dim), z-grid, induced densities,
            induced potentials, energies at zero crossings.
        """

        assert self.world.size == 1
        # eps_qwij = self.get_eps_matrix()

        Nw = len(self.frequencies)
        Nq = self.mynq
        w_w = self.frequencies
        eig = np.zeros([Nq, Nw, self.dim], dtype=complex)
        abseps = np.zeros([Nq, Nw], dtype=complex)
        # vec = np.zeros([Nq, Nw, self.dim, self.dim],
        #                dtype=complex)

        # import scipy as sp
        omega0 = [[] for i in range(Nq)]

        rho_z = [np.zeros([0, len(self.z_big)]) for i in range(Nq)]
        phi_z = [np.zeros([0, len(self.z_big)]) for i in range(Nq)]
        for iq in range(Nq):
            if (1 + iq) % (Nq // 10) == 0:
                print('{}%'.format(np.round((1 + iq) / Nq, 1) * 100))

            eps_qwij = self.get_eps_matrix(iq_q=[iq])
            if iq == 0:
                print('Calculating plasmon modes')

            eig[iq], vec_wij = np.linalg.eig(eps_qwij[0])
            abseps[iq, :] = np.linalg.det(eps_qwij[0])

            vec_dual_wij = np.linalg.inv(vec_wij)
            iwref = 0
            for iw in range(1, Nw):
                vec = vec_wij[iw]
                vec_dual = vec_dual_wij[iwref]
                overlap = np.dot(vec_dual, vec)
                index = list(np.argsort(np.abs(overlap))[:, -1])
                iwref = iw
                if len(np.unique(index)) < self.dim:  # add missing indices
                    addlist = []
                    removelist = []
                    for j in range(self.dim):
                        if index.count(j) < 1:
                            addlist.append(j)
                        if index.count(j) > 1:
                            for l in range(1, index.count(j)):
                                removelist.append(
                                    np.argwhere(np.array(index) == j)[l])
                    for j in range(len(addlist)):
                        index[removelist[j][0]] = addlist[j]

                # Sort vectors
                vec_wij[iw, :] = np.copy(vec_wij[iw][:, index])
                vec_dual_wij[iw, :] = np.copy(vec_dual_wij[iw][index, :])

                eig[iq, iw, :] = np.copy(eig[iq, iw][index])
                klist = [k for k in range(self.dim)
                         if (eig[iq, iw - 1, k] < 0 and eig[iq, iw, k] > 0)]
                for k in klist:  # Eigenvalue crossing
                    a = np.real((eig[iq, iw, k] - eig[iq, iw - 1, k]) /
                                (w_w[iw] - w_w[iw - 1]))
                    # linear interp for crossing point
                    w0 = np.real(-eig[iq, iw - 1, k]) / a + w_w[iw - 1]
                    rho = np.dot(self.drho_array[:, iq, :].T,
                                 vec_dual_wij[iw, k, :])
                    phi = np.dot(self.dphi_array[:, iq, :].T,
                                 vec_wij[iw, :, k])
                    rho_z[iq] = np.append(rho_z[iq], rho[np.newaxis, :],
                                          axis=0)
                    phi_z[iq] = np.append(phi_z[iq], phi[np.newaxis, :],
                                          axis=0)
                    omega0[iq].append(w0 * Hartree)

        # Make eigenfrequencies more easy to work with
        nmodes = 0
        for freqs in omega0:
            nmodes = np.max([len(freqs), nmodes])

        freqs_qm = np.zeros((Nq, nmodes), float) + np.nan

        for freqs_m, omega0_m in zip(freqs_qm, omega0):
            freqs_m[:len(omega0_m)] = np.sort(omega0_m)

        z = self.z_big * Bohr
        if filename is not None:
            q_q = self.q_abs / Bohr
            omega_w = self.frequencies * Hartree
            data = {'eig': eig, 'z': z,
                    'rho_z': rho_z, 'freqs_qm': freqs_qm,
                    'q_q': q_q, 'omega_w': omega_w,
                    'abseps': abseps}
            np.savez_compressed(filename, **data)

        return eig, z, rho_z, phi_z, freqs_qm, abseps

    def collect_q(self, a_q):
        """ Collect arrays of dim (q)"""
        world = self.world
        nq = len(self.q_abs)
        mynq = (nq + self.world.size - 1) // self.world.size
        b_q = np.zeros(mynq, a_q.dtype)
        b_q[:self.q2 - self.q1] = a_q
        A_q = np.empty(mynq * world.size, a_q.dtype)
        if world.size == 1:
            A_q[:] = b_q
        else:
            world.all_gather(b_q, A_q)
        return A_q[:nq]

    def collect_qw(self, a_qw):
        """ Collect arrays of dim (q X w)"""
        nw = a_qw.shape[1]
        nq = len(self.q_abs)
        A_qw = np.zeros((nq, nw),
                        a_qw.dtype)
        for iw in range(nw):
            A_qw[:, iw] = self.collect_q(a_qw[:, iw])
        nq = len(self.q_abs)
        return A_qw[:nq]


class BuildingBlock():

    """ Module for using Linear response to calculate dielectric
    building block of 2D material with GPAW"""

    def __init__(self, filename, df, isotropic_q=True, nq_inf=10,
                 direction='x', qmax=None, txt=sys.stdout):
        """Creates a BuildingBlock object.

        filename: str
            used to save data file: filename-chi.npz
        df: DielectricFunction object
            Determines how linear response calculation is performed
        isotropic_q: bool
            If True, only q-points along one direction (1 0 0) in the
            2D BZ is included, thus asuming an isotropic material
        direction: 'x' or 'y'
            Direction used for isotropic q sampling.
        qmax: float
            Cutoff for q-grid. To be used if one wishes to sample outside the
            irreducible BZ. Only works for isotropic q-sampling.
        nq_inf: int
            number of extra q points in the limit q->0 along each direction,
            extrapolated from q=0, assumung that the head of chi0_wGG goes
            as q^2 and the wings as q.
            Note that this does not hold for (semi)metals!

        """
        if qmax is not None:
            assert isotropic_q
        self.filename = filename
        self.isotropic_q = isotropic_q
        self.nq_inf = nq_inf
        self.nq_inftot = nq_inf
        if not isotropic_q:
            self.nq_inftot *= 2

        if direction == 'x':
            qdir = 0
        elif direction == 'y':
            qdir = 1
        self.direction = direction

        self.df = df  # dielectric function object
        self.df.truncation = '2D'  # in case you forgot!
        self.omega_w = self.df.chi0.omega_w
        self.world = self.df.chi0.world

        if self.world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w', 1)
        self.fd = txt

        calc = self.df.chi0.calc
        kd = calc.wfs.kd
        self.kd = kd
        r = calc.wfs.gd.get_grid_point_coordinates()
        self.z = r[2, 0, 0, :]

        nw = self.omega_w.shape[0]
        self.chiM_qw = np.zeros([0, nw])
        self.chiD_qw = np.zeros([0, nw])
        self.drhoM_qz = np.zeros([0, self.z.shape[0]])
        self.drhoD_qz = np.zeros([0, self.z.shape[0]])

        # First: choose all ibzq in 2D BZ
        from ase.dft.kpoints import monkhorst_pack
        from gpaw.kpt_descriptor import KPointDescriptor
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        qd = KPointDescriptor(bzq_qc)
        qd.set_symmetry(calc.atoms, kd.symmetry)
        q_cs = qd.ibzk_kc
        rcell_cv = 2 * pi * np.linalg.inv(calc.wfs.gd.cell_cv).T
        if isotropic_q:  # only use q along [1 0 0] or [0 1 0] direction.
            Nk = kd.N_c[qdir]
            qx = np.array(range(1, Nk // 2)) / float(Nk)
            q_cs = np.zeros([Nk // 2 - 1, 3])
            q_cs[:, qdir] = qx
            q = 0
            if qmax is not None:
                qmax *= Bohr
                qmax_v = np.zeros([3])
                qmax_v[qdir] = qmax
                q_c = q_cs[-1]
                q_v = np.dot(q_c, rcell_cv)
                q = (q_v**2).sum()**0.5
                assert Nk % 2 == 0
                i = Nk / 2.0
                while q < qmax:
                    if i == Nk:  # omit BZ edge
                        i += 1
                        continue
                    q_c = np.zeros([3])
                    q_c[qdir] = i / Nk
                    q_cs = np.append(q_cs, q_c[np.newaxis, :], axis=0)
                    q_v = np.dot(q_c, rcell_cv)
                    q = (q_v**2).sum()**0.5
                    i += 1
        q_vs = np.dot(q_cs, rcell_cv)
        q_abs = (q_vs**2).sum(axis=1)**0.5
        sort = np.argsort(q_abs)
        q_abs = q_abs[sort]
        q_cs = q_cs[sort]
        if isotropic_q:
            q_cut = q_abs[0] / 2  # extrapolate to half of smallest finite q
        else:
            q_cut = q_abs[1]  # smallest finite q
        self.nq_cut = self.nq_inftot + 1

        q_infs = np.zeros([q_cs.shape[0] + self.nq_inftot, 3])
        # x-direction:
        q_infs[: self.nq_inftot, qdir] = \
            np.linspace(1e-05, q_cut, self.nq_inftot + 1)[:-1]
        if not isotropic_q:  # y-direction
            q_infs[self.nq_inf:self.nq_inftot + 1, 1] = \
                np.linspace(0, q_cut, self.nq_inf + 1)[1:]

        # add q_inf to list
        self.q_cs = np.insert(q_cs, 0, np.zeros([self.nq_inftot, 3]), axis=0)
        self.q_vs = np.dot(self.q_cs, rcell_cv)
        self.q_vs += q_infs
        self.q_abs = (self.q_vs**2).sum(axis=1)**0.5
        self.q_infs = q_infs
        self.complete = False
        self.nq = 0
        if self.load_chi_file():
            if self.complete:
                print('Building block loaded from file', file=self.fd)
        world.barrier()

    def calculate_building_block(self, add_intraband=False):
        if self.complete:
            return
        Nq = self.q_cs.shape[0]
        for nq in range(self.nq, Nq):
            self.nq = nq
            self.save_chi_file()
            q_c = self.q_cs[nq]
            q_inf = self.q_infs[nq]
            if np.allclose(q_inf, 0):
                q_inf = None

            qcstr = '(' + ', '.join(['%.3f' % x for x in q_c]) + ')'
            print('Calculating contribution from q-point #%d/%d, q_c=%s'
                  % (nq + 1, Nq, qcstr), file=self.fd)
            if q_inf is not None:
                qstr = '(' + ', '.join(['%.3f' % x for x in q_inf]) + ')'
                print('    and q_inf=%s' % qstr, file=self.fd)
            pd, chi0_wGG, \
                chi_wGG = self.df.get_dielectric_matrix(
                    symmetric=False,
                    calculate_chi=True,
                    q_c=q_c,
                    q_v=q_inf,
                    direction=self.direction,
                    add_intraband=add_intraband)
            print('calculated chi!', file=self.fd)

            nw = len(self.omega_w)
            world = self.df.chi0.world
            w1 = min(self.df.mynw * world.rank, nw)

            q, omega_w, chiM_qw, chiD_qw, z, drhoM_qz, drhoD_qz = \
                get_chi_2D(self.omega_w, pd, chi_wGG)

            chiM_w = chiM_qw[0]
            chiD_w = chiD_qw[0]
            chiM_w = self.collect(chiM_w)
            chiD_w = self.collect(chiD_w)

            if self.world.rank == 0:
                assert w1 == 0  # drhoM and drhoD in static limit
                self.update_building_block(chiM_w[np.newaxis, :],
                                           chiD_w[np.newaxis, :],
                                           drhoM_qz, drhoD_qz)

        # Induced densities are not probably described in q-> 0 limit-
        # replace with finite q result:
        if self.world.rank == 0:
            for n in range(Nq):
                if np.allclose(self.q_cs[n], 0):
                    self.drhoM_qz[n] = self.drhoM_qz[self.nq_cut]
                    self.drhoD_qz[n] = self.drhoD_qz[self.nq_cut]

        self.complete = True
        self.save_chi_file()

        return

    def update_building_block(self, chiM_qw, chiD_qw, drhoM_qz,
                              drhoD_qz):

        self.chiM_qw = np.append(self.chiM_qw, chiM_qw, axis=0)
        self.chiD_qw = np.append(self.chiD_qw, chiD_qw, axis=0)
        self.drhoM_qz = np.append(self.drhoM_qz, drhoM_qz, axis=0)
        self.drhoD_qz = np.append(self.drhoD_qz, drhoD_qz, axis=0)

    def save_chi_file(self, filename=None):
        if filename is None:
            filename = self.filename
        data = {'last_q': self.nq,
                'complete': self.complete,
                'isotropic_q': self.isotropic_q,
                'q_cs': self.q_cs,
                'q_vs': self.q_vs,
                'q_abs': self.q_abs,
                'omega_w': self.omega_w,
                'chiM_qw': self.chiM_qw,
                'chiD_qw': self.chiD_qw,
                'z': self.z,
                'drhoM_qz': self.drhoM_qz,
                'drhoD_qz': self.drhoD_qz}

        if self.world.rank == 0:
            np.savez_compressed(filename + '-chi.npz',
                                **data)
        world.barrier()

    def load_chi_file(self):
        try:
            data = np.load(self.filename + '-chi.npz')
        except IOError:
            return False
        if (np.all(data['omega_w'] == self.omega_w) and
            np.all(data['q_cs'] == self.q_cs) and
                np.all(data['z'] == self.z)):
            self.nq = data['last_q']
            self.complete = data['complete']
            self.chiM_qw = data['chiM_qw']
            self.chiD_qw = data['chiD_qw']
            self.drhoM_qz = data['drhoM_qz']
            self.drhoD_qz = data['drhoD_qz']
            return True
        else:
            return False

    def interpolate_to_grid(self, q_grid, w_grid):
        """
        Parameters
        q_grid: in Ang. should start at q=0
        w_grid: in eV
        """

        from scipy.interpolate import RectBivariateSpline
        from scipy.interpolate import interp1d
        if not self.complete:
            self.calculate_building_block()
        q_grid *= Bohr
        w_grid /= Hartree

        assert np.max(q_grid) <= np.max(self.q_abs), \
            'q can not be larger that %1.2f Ang' % np.max(self.q_abs / Bohr)
        assert np.max(w_grid) <= np.max(self.omega_w), \
            'w can not be larger that %1.2f eV' % \
            np.max(self.omega_w * Hartree)

        sort = np.argsort(self.q_abs)
        q_abs = self.q_abs[sort]

        # chi monopole
        self.chiM_qw = self.chiM_qw[sort]

        omit_q0 = False
        if np.isclose(q_abs[0], 0) and not np.isclose(self.chiM_qw[0, 0], 0):
            omit_q0 = True  # omit q=0 from interpolation
            q0_abs = q_abs[0].copy()
            q_abs[0] = 0.
            chi0_w = self.chiM_qw[0].copy()
            self.chiM_qw[0] = np.zeros_like(chi0_w)

        yr = RectBivariateSpline(q_abs, self.omega_w,
                                 self.chiM_qw.real,
                                 s=0)

        yi = RectBivariateSpline(q_abs, self.omega_w,
                                 self.chiM_qw.imag, s=0)

        self.chiM_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)
        if omit_q0:
            yr = interp1d(self.omega_w, chi0_w.real)
            yi = interp1d(self.omega_w, chi0_w.imag)
            chi0_w = yr(w_grid) + 1j * yi(w_grid)
            q_abs[0] = q0_abs
            if np.isclose(q_grid[0], 0):
                self.chiM_qw[0] = chi0_w

        # chi dipole
        yr = RectBivariateSpline(q_abs, self.omega_w,
                                 self.chiD_qw[sort].real,
                                 s=0)
        yi = RectBivariateSpline(q_abs, self.omega_w,
                                 self.chiD_qw[sort].imag,
                                 s=0)

        self.chiD_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)

        # drho monopole

        yr = RectBivariateSpline(q_abs, self.z,
                                 self.drhoM_qz[sort].real, s=0)
        yi = RectBivariateSpline(q_abs, self.z,
                                 self.drhoM_qz[sort].imag, s=0)

        self.drhoM_qz = yr(q_grid, self.z) + 1j * yi(q_grid, self.z)

        # drho dipole
        yr = RectBivariateSpline(q_abs, self.z,
                                 self.drhoD_qz[sort].real, s=0)
        yi = RectBivariateSpline(q_abs, self.z,
                                 self.drhoD_qz[sort].imag, s=0)

        self.drhoD_qz = yr(q_grid, self.z) + 1j * yi(q_grid, self.z)

        self.q_abs = q_grid
        self.omega_w = w_grid

        self.save_chi_file(filename=self.filename + '_int')

    def collect(self, a_w):
        world = self.df.chi0.world
        b_w = np.zeros(self.df.mynw, a_w.dtype)
        b_w[:self.df.w2 - self.df.w1] = a_w
        nw = len(self.omega_w)
        A_w = np.empty(world.size * self.df.mynw, a_w.dtype)
        world.all_gather(b_w, A_w)
        return A_w[:nw]

    def clear_temp_files(self):
        if not self.savechi0:
            world = self.df.chi0.world
            if world.rank == 0:
                while len(self.temp_files) > 0:
                    filename = self.temp_files.pop()
                    os.remove(filename)


"""TOOLS"""


def check_building_blocks(BBfiles=None):
    """ Check that building blocks are on same frequency-
    and q- grid.

    BBfiles: list of str
        list of names of BB files
    """
    name = BBfiles[0]
    data = np.load(name)
    try:
        q = data['q_abs'].copy()
        w = data['omega_w'].copy()
    except TypeError:
        # Skip test for old format:
        return True
    for name in BBfiles[1:]:
        data = np.load(name)
        if not ((data['q_abs'] == q).all and
                (data['omega_w'] == w).all):
            return False
    return True


def interpolate_building_blocks(BBfiles=None, BBmotherfile=None,
                                q_grid=None, w_grid=None, pad=True):
    """ Interpolate building blocks to same frequency-
    and q- grid

    BBfiles: list of str
        list of names of BB files to be interpolated
    BBmother: str
        name of BB file to match the grids to. Will
        also be interpolated to common grid.
    q_grid: float
        q-grid in Ang. Should start at q=0
    w_grid: float
        in eV
    """

    from scipy.interpolate import RectBivariateSpline, interp1d

    if BBmotherfile is not None:
        BBfiles.append(BBmotherfile)

    if BBmotherfile is not None and '-chi.npz' not in BBmotherfile:
        BBmotherfile = BBmotherfile + '-chi.npz'

    for il, filename in enumerate(BBfiles):
        if '-chi.npz' not in filename:
            BBfiles[il] = filename + '-chi.npz'

    q_max = 1000
    w_max = 1000
    for name in BBfiles:
        data = np.load(open(name, 'rb'))
        q_abs = data['q_abs']
        q_max = np.min([q_abs[-1], q_max])
        ow = data['omega_w']
        w_max = np.min([ow[-1], w_max])

    if BBmotherfile is not None:
        data = np.load(BBmotherfile)
        q_grid = data['q_abs']
        w_grid = data['omega_w']
    else:
        q_grid = q_grid * Bohr
        w_grid = w_grid / Hartree

    q_grid = [q for q in q_grid if q < q_max]

    if pad:
        q_grid.append(q_max)
    w_grid = [w for w in w_grid if w < w_max]
    if pad:
        w_grid.append(w_max)
    q_grid = np.array(q_grid)
    w_grid = np.array(w_grid)
    for name in BBfiles:
        # assert data['isotropic_q']
        data = np.load(name)
        q_abs = data['q_abs']
        w = data['omega_w']
        z = data['z']
        chiM_qw = data['chiM_qw']
        chiD_qw = data['chiD_qw']
        drhoM_qz = data['drhoM_qz']
        drhoD_qz = data['drhoD_qz']

        # chi monopole
        omit_q0 = False
        if np.isclose(q_abs[0], 0) and not np.isclose(chiM_qw[0, 0], 0):
            omit_q0 = True  # omit q=0 from interpolation
            q0_abs = q_abs[0].copy()
            q_abs[0] = 0.
            chi0_w = chiM_qw[0].copy()
            chiM_qw[0] = np.zeros_like(chi0_w)

        yr = RectBivariateSpline(q_abs, w,
                                 chiM_qw.real,
                                 s=0)

        yi = RectBivariateSpline(q_abs, w,
                                 chiM_qw.imag, s=0)

        chiM_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)

        if omit_q0:
            yr = interp1d(w, chi0_w.real)
            yi = interp1d(w, chi0_w.imag)
            chi0_w = yr(w_grid) + 1j * yi(w_grid)
            q_abs[0] = q0_abs
            if np.isclose(q_grid[0], 0):
                chiM_qw[0] = chi0_w

        # chi dipole
        yr = RectBivariateSpline(q_abs, w,
                                 chiD_qw.real,
                                 s=0)
        yi = RectBivariateSpline(q_abs, w,
                                 chiD_qw.imag,
                                 s=0)

        chiD_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)

        # drho monopole

        yr = RectBivariateSpline(q_abs, z,
                                 drhoM_qz.real, s=0)
        yi = RectBivariateSpline(q_abs, z,
                                 drhoM_qz.imag, s=0)

        drhoM_qz = yr(q_grid, z) + 1j * yi(q_grid, z)

        # drho dipole
        yr = RectBivariateSpline(q_abs, z,
                                 drhoD_qz.real, s=0)
        yi = RectBivariateSpline(q_abs, z,
                                 drhoD_qz.imag, s=0)

        drhoD_qz = yr(q_grid, z) + 1j * yi(q_grid, z)

        q_abs = q_grid
        omega_w = w_grid

        data = {'q_abs': q_abs,
                'omega_w': omega_w,
                'chiM_qw': chiM_qw,
                'chiD_qw': chiD_qw,
                'z': z,
                'drhoM_qz': drhoM_qz,
                'drhoD_qz': drhoD_qz,
                'isotropic_q': True}

        np.savez_compressed(name[:-8] + "_int-chi.npz",
                            **data)


def get_chi_2D(omega_w=None, pd=None, chi_wGG=None, q0=None,
               filenames=None, name=None):
    r"""Calculate the monopole and dipole contribution to the
    2D susceptibillity chi_2D, defined as

    ::

      \chi^M_2D(q, \omega) = \int\int dr dr' \chi(q, \omega, r,r') \\
                          = L \chi_{G=G'=0}(q, \omega)
      \chi^D_2D(q, \omega) = \int\int dr dr' z \chi(q, \omega, r,r') z'
                           = 1/L sum_{G_z,G_z'} z_factor(G_z)
                           chi_{G_z,G_z'} z_factor(G_z'),
      Where z_factor(G_z) =  +/- i e^{+/- i*G_z*z0}
      (L G_z cos(G_z L/2)-2 sin(G_z L/2))/G_z^2

    input parameters:

    filenames: list of str
        list of chi_wGG.pckl files for different q calculated with
        the DielectricFunction module in GPAW
    name: str
        name writing output files
    """

    q_list_abs = []
    if chi_wGG is None and filenames is not None:
        omega_w, pd, chi_wGG, q0 = read_chi_wGG(filenames[0])
        nq = len(filenames)
    elif chi_wGG is not None:
        nq = 1
    nw = chi_wGG.shape[0]
    r = pd.gd.get_grid_point_coordinates()
    z = r[2, 0, 0, :]
    L = pd.gd.cell_cv[2, 2]  # Length of cell in Bohr
    z0 = L / 2.  # position of layer
    chiM_qw = np.zeros([nq, nw], dtype=complex)
    chiD_qw = np.zeros([nq, nw], dtype=complex)
    drhoM_qz = np.zeros([nq, len(z)], dtype=complex)  # induced density
    drhoD_qz = np.zeros([nq, len(z)], dtype=complex)  # induced dipole density
    for iq in range(nq):
        if iq != 0:
            omega_w, pd, chi_wGG, q0 = read_chi_wGG(filenames[iq])
        if q0 is not None:
            q = q0
        else:
            q = pd.K_qv
        npw = chi_wGG.shape[1]
        Gvec = pd.get_reciprocal_vectors(add_q=False)

        Glist = []
        for iG in range(npw):  # List of G with Gx,Gy = 0
            if Gvec[iG, 0] == 0 and Gvec[iG, 1] == 0:
                Glist.append(iG)

        chiM_qw[iq] = L * chi_wGG[:, 0, 0]
        drhoM_qz[iq] += chi_wGG[0, 0, 0]
        q_abs = np.linalg.norm(q)
        q_list_abs.append(q_abs)
        for iG in Glist[1:]:
            G_z = Gvec[iG, 2]
            qGr_R = np.inner(G_z, z.T).T
            # Fourier transform to get induced density at \omega=0
            drhoM_qz[iq] += np.exp(1j * qGr_R) * chi_wGG[0, iG, 0]
            for iG1 in Glist[1:]:
                G_z1 = Gvec[iG1, 2]
                # integrate with z along both coordinates
                factor = z_factor(z0, L, G_z)
                factor1 = z_factor(z0, L, G_z1, sign=-1)
                chiD_qw[iq, :] += 1. / L * factor * chi_wGG[:, iG, iG1] * \
                    factor1
                # induced dipole density due to V_ext = z
                drhoD_qz[iq, :] += 1. / L * np.exp(1j * qGr_R) * \
                    chi_wGG[0, iG, iG1] * factor1
    # Normalize induced densities with chi
    drhoM_qz /= np.repeat(chiM_qw[:, 0, np.newaxis], drhoM_qz.shape[1],
                          axis=1)
    drhoD_qz /= np.repeat(chiD_qw[:, 0, np.newaxis], drhoM_qz.shape[1],
                          axis=1)

    """ Returns q array, frequency array, chi2D monopole and dipole, induced
    densities and z array (all in Bohr)
    """
    if name is not None:
        arrays = [np.array(q_list_abs), omega_w, chiM_qw, chiD_qw,
                  z, drhoM_qz, drhoD_qz]
        names = ['q_abs', 'omega_w', 'z',
                 'chiM_qw', 'chiD_qw',
                 'drhoM_qz', 'drhoD_qz']
        data = {}
        for array, name in zip(arrays, names):
            data[name] = array
        np.save(name + '-chi.npz', **data)
    return np.array(q_list_abs) / Bohr, omega_w * Hartree, chiM_qw, \
        chiD_qw, z, drhoM_qz, drhoD_qz


def z_factor(z0, d, G, sign=1):
    factor = -1j * sign * np.exp(1j * sign * G * z0) * \
        (d * G * np.cos(G * d / 2.) - 2. * np.sin(G * d / 2.)) / G**2
    return factor


def z_factor2(z0, d, G, sign=1):
    factor = sign * np.exp(1j * sign * G * z0) * np.sin(G * d / 2.)
    return factor


def expand_layers(structure):
    newlist = []
    for name in structure:
        num = ''
        while name[0].isdigit():
            num += name[0]
            name = name[1:]
        try:
            num = int(num)
        except ValueError:
            num = 1
        for n in range(num):
            newlist.append(name)
    return newlist


def read_chi_wGG(name):
    """
    Read density response matrix calculated with the DielectricFunction
    module in GPAW.
    Returns frequency grid, gpaw.wavefunctions object, chi_wGG
    """
    fd = open(name, 'rb')
    omega_w, pd, chi_wGG, q0, chi0_wvv = load(fd)
    nw = len(omega_w)
    nG = pd.ngmax
    chi_wGG = np.empty((nw, nG, nG), complex)
    for chi_GG in chi_wGG:
        chi_GG[:] = load(fd)
    return omega_w, pd, chi_wGG, q0


def plot_plasmons(hs, output,
                  plot_eigenvalues=False,
                  plot_density=False,
                  plot_potential=False,
                  save_plots=False,
                  show=True):
    eig, z, rho_z, phi_z, omega0, abseps = output

    import matplotlib.pyplot as plt
    q_q = hs.q_abs / Bohr
    nq = len(q_q)
    omega_w = hs.frequencies * Hartree

    plt.figure()
    plt.title('Plasmon modes')
    for iq in range(nq):
        freqs = np.array(omega0[iq])
        plt.plot([q_q[iq], ] * len(freqs), freqs, 'k.')
        plt.ylabel(r'$\hbar\omega$ (eV)')
        plt.xlabel(r'q (Å$^{-1}$)')
    if save_plots is not None:
        plt.savefig('Plasmon_Modes_' + str(save_plots))

    plt.figure()
    plt.title('Loss Function')
    loss_qw = np.sum(-np.imag(eig**(-1)), axis=-1)
    plt.pcolor(q_q, omega_w, loss_qw.T, vmax=10)
    plt.ylabel(r'$\hbar\omega$ (eV)')
    plt.xlabel(r'q (Å$^{-1}$)')
    plt.colorbar()
    if save_plots is not None:
        plt.savefig('Loss_' + str(save_plots))

    if plot_eigenvalues:
        plt.figure()
        for iq in range(0, nq, nq // 10):
            plt.plot(omega_w, eig[iq].real)
            plt.plot(omega_w, eig[iq].imag, '--')
        if save_plots is not None:
            plt.savefig('Eigenvalues_' + str(save_plots))

    if plot_potential:
        plt.figure()
        plt.title('Induced potential')
        q = nq // 5
        pots = np.array(phi_z[q]).real
        plt.plot(z, pots.T)
        plt.xlabel(r'z $(\AA)$')
        if save_plots is not None:
            plt.savefig('Potential_' + str(save_plots))

    if plot_density:
        plt.figure()
        plt.title('Induced density')
        q = nq // 5
        dens = np.array(rho_z[q]).real
        plt.plot(z, dens.T)
        plt.xlabel(r'z $(\AA)$')
        if save_plots is not None:
            plt.savefig('Density_' + str(save_plots))


def make_heterostructure(layers,
                         frequencies=[0.001, 5.0, 5000],
                         momenta=[0.0001, 5.0, 2000],
                         thicknesses=None,
                         substrate=None):
    """Easy function for making a heterostructure based on some layers"""

    # Copy for internal handling
    layers = layers.copy()
    layers = expand_layers(layers)

    # Remove those annoying '-chi.npz'
    for il, layer in enumerate(layers):
        if '-chi.npz' in layer:
            layers[il] = layer[:-8]

    # Parse input for layer
    originallayers = []  # Unmodified layer identifiers
    layerargs = []  # Modifiers for layers like +phonons and +doping

    for layer in layers:
        tmp = layer.split('+')
        name, layerargs = tmp[0], tmp[1:]
        originallayers.append(name)
        layerargs.append(layerargs)

    if thicknesses is None:
        thicknesses = []
        for layer in layers:
            for key in default_thicknesses:
                if '-icsd-' in key:
                    key2 = key.split('-icsd-')[0]
                else:
                    key2 = key

                if key2 in layer:
                    thicknesses.append(default_thicknesses[key])
                    break
            else:
                raise NotImplementedError
                print('define the thickness of all layers')

    q_q = np.linspace(*momenta)
    omega_w = np.linspace(*frequencies)
    # Interpolate the building blocks such that they are
    # represented on the same q and omega grid
    print('Interpolating building blocks to same grid')
    interpolate_building_blocks(BBfiles=originallayers, q_grid=q_q,
                                w_grid=omega_w,
                                pad=False)

    # The layers now have appended an '_int'
    for il, layer in enumerate(layers):
        ind = layer.find('+')
        if ind < 0:
            layers[il] = layer + '_int'
            continue

        layers[il] = layer[:ind] + '_int' + layer[ind:]

    # Parse args and modify building blocks accordingly
    from gpaw.qeh.buildingblocks import (GrapheneBB,
                                         dopedsemiconductor,
                                         phonon_polarizability)
    for layer in set(layers):
        # Everything that comes after a '+' is a modifier
        tmp = layer.split('+')
        modifiers = tmp[1:]

        if not len(modifiers):
            continue

        origin = tmp[0]
        originpath = origin + '-chi.npz'

        bb = np.load(originpath)

        for im, modifier in enumerate(modifiers):
            if 'doping' in modifier:
                subargs = modifier.split(',')
                kwargs = {'doping': 0,
                          'temperature': 0,
                          'eta': 1e-4,
                          'direction': 'x'}

                # Try to find emass in default values
                for key, val in default_ehmasses.items():
                    if origin[:-4] in key:
                        kwargs['effectivemass'] = val['emass1']

                # Overwrite
                for subarg in subargs:
                    key, value = subarg.split('=')
                    if key == 'T':
                        key = 'temperature'
                    elif key == 'em':
                        key = 'effectivemass'
                    elif key == 'direction':
                        try:
                            kwargs[key] = float(value)
                        except ValueError:
                            kwargs[key] = value
                        continue
                    kwargs[key] = float(value)

                mod = ['{}={}'.format(str(key), str(val))
                       for key, val in kwargs.items()]
                modifiers[im] = ','.join(mod)

                if 'graphene' in layer:
                    # Treat graphene specially, since in this case we are using
                    # an analytical approximation of the building block
                    assert np.allclose(kwargs['temperature'], 0.0), \
                        print('Graphene currently cannot be at a finite temp.')
                    bb = GrapheneBB(bb, doping=kwargs['doping'],
                                    eta=kwargs['eta'])
                else:
                    bb = dopedsemiconductor(bb, **kwargs)


            if 'phonons' in modifier:
                phonons = Path(origin[:-4] + '-phonons.npz')

                if not phonons.exists():
                    continue
                dct = np.load(str(phonons))

                Z_avv, freqs, modes, masses, cell = (dct['Z_avv'],
                                                     dct['freqs'],
                                                     dct['modes'],
                                                     dct['masses'],
                                                     dct['cell'])
                bb = phonon_polarizability(bb, Z_avv, freqs, modes,
                                           masses, cell)
                

        # Save modified building block
        newlayer = '{}+{}'.format(origin, '+'.join(modifiers))
        newlayerpath = newlayer + '-chi.npz'
        np.savez_compressed(newlayerpath, **bb)

        for il, layer2 in enumerate(layers):
            if layer2 == layer:
                layers[il] = newlayer

    thicknesses = np.array(thicknesses)

    # Calculate distance between layers
    d = (thicknesses[1:] + thicknesses[:-1]) / 2
    d0 = thicknesses[0]

    # Print summary of structure
    print('Structure:')
    if substrate is not None:
        print('    Substrate d = {} Å'.format(substrate['d'][0]))
    for thickness, layer in zip(thicknesses, layers):
        print('    {} d = {} Å'.format(layer, thickness))
    het = Heterostructure(structure=layers, d=d,
                          thicknesses=thicknesses, d0=d0,
                          substrate=substrate)
    return het


def main(args=None):
    import argparse
    from pathlib import Path
    from os.path import expanduser
    import shutil

    description = 'QEH Model command line interface'

    example_text = """examples:
    (In the following "qeh = python -m gpaw.qeh". It can be nice
    to set this as an alias in your bashrc.)

    Calculate graphene plasmons with doping and plot them:
        qeh graphene+doping=0.5 --plasmons --plot

    Same but 5 layers of graphene and save to numpy npz file:
        qeh 5graphene+doping=0.5 --plasmons --plasmonfile plasmons.npz

    Graphene-Boron Nitride-Graphene heterostructure with doping and phonons:
        qeh graphene+doping=0.5 3BN+phonons graphene+doping=0.5

    Set up doped MoS2 at finite temperatures (T is in eV) with a
    custom relaxation rate:
        qeh H-MoS2+doping=0.1,T=25e-3,eta=1e-3,em=0.43

    Set custom omega and q grid:
       qeh 2graphene+doping=0.5 --q 1e-3 0.1 100 --omega 1e-3 2 1000

    """
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=description,
                                     epilog=example_text,
                                     formatter_class=formatter)

    help = """
    '3H-MoS2 graphene 10H-WS2' gives 3 layers of H-MoS2,
    1 layer of graphene and 10 layers of H-WS2. Each layer can be further
    modified (e.g. adding doping and phonon contribution) by appending
    +doping=0.1 or +phonons.

    For example 3 layers of 0.1 eV doped MoS2 (with effective mass m*=0.43)
    can be modelled by '3H-MoS2+doping=0.1,em=0.43'. Phonon contributions
    are added as '3H-MoS2+phonons'. Additional arguments such as the
    temperature, effective masses, relaxation rate can also be added.
    Please see the provided examples in the bottom.
    """

    parser.add_argument('layers', nargs='+', help=help, type=str)
    help = ("For above example: '6.2 3.2 6.2' gives thicknesses of "
            "6.2 3.2 and 6.2 AA to MoS2, graphene and WS2 "
            "respectively. If not set, the QEH module will use a "
            "default set of thicknesses")
    parser.add_argument('--thicknesses', nargs='*', help=help,
                        default=None, type=float)

    help = ("Path to folder containing dielectric building blocks. "
            "Defaults to current folder, ./chi-data and ~/.chi-data "
            "in that order")
    parser.add_argument('--buildingblockpath', nargs='*', help=help,
                        default=['.', './chi-data', '~/chi-data'], type=str)

    help = ("Calculate plasmon spectrum")
    parser.add_argument('--plasmons', action='store_true', help=help)

    help = ("Calculate eigenvalues of dielectric matrix")
    parser.add_argument('--eigenvalues', action='store_true', help=help)

    help = ("Calculate induced potential for finite q")
    parser.add_argument('--potential', action='store_true', help=help)

    help = ("Calculate induced density for finite q")
    parser.add_argument('--density', action='store_true', help=help)

    help = ("Save plasmon modes to file")
    parser.add_argument('--plasmonfile', type=str, default=None, help=help)

    help = ("Plot calculated quantities")
    parser.add_argument('--plot', action='store_true', help=help)

    help = ("Save plots to file")
    parser.add_argument('--saveplots', type=str, default=None, help=help)

    help = ("Add a substrate to the structure."
            "A file '-sub.npz' that contains the dielectric function of the "
            "substrate, (in x and z direction in the anisotropic case)"
            "the omega grid where the epsilon is defined, the distance"
            "of the substrate to the bottom most layer must to be given"
            "and a True or False 'isotropic' argument."
            "E. g.: --substrate SiO2")
    parser.add_argument('--substrate', type=str, default=None, help=help)

    help = ("Custom frequencies to represent quantities on (in eV). "
            "The format is: min. frequency, max. frequency, "
            "number of frequencies. E. g.: 0.001 1.0 500")
    parser.add_argument('--omega', default=[0.001, 1.0, 500],
                        nargs=3,
                        help=help, type=float)

    help = ("Custom momentas to respresent quantities on (in AA^-1). "
            "The format is: min. q, max. q, number of q's. "
            "E. g.: 0.0001 0.15 100")
    parser.add_argument('--q', default=[0.0001, 0.15, 200],
                        nargs=3,
                        help=help, type=float)

    args = parser.parse_args(args)
    layers = args.layers
    paths = args.buildingblockpath

    layers = expand_layers(layers)
    # for il, layer in enumerate(layers):
    #     layers[il] = layer + '-chi.npz'

    # Make sure that the building block files can be found
    link = "https://cmr.fysik.dtu.dk/_downloads/chi-data.tar.gz"
    msg = """

    Building block file ({bb}) cannot be found!
    Please download and unpack the dielectric building blocks.
    This can be done with:
    wget {link}
    tar -zxf chi-data.tar.gz
    mv chi-data ~
    which will put all building blocks in a folder in
    your home directory
    """

    # Locate files for layers
    print('Looking for building blocks')
    layer_paths = []
    for il, layer in enumerate(layers):
        # Parse layer and its arguments
        tmp = layer.split('+')
        layer = tmp[0]

        layerpath = layer + '-chi.npz'
        p = Path(layerpath)
        for path in paths:
            bb = Path(expanduser(path)) / p
            if bb.is_file():
                break
        else:
            raise FileNotFoundError(msg.format(bb=layerpath,
                                               link=link))

        layer_paths.append(str(bb))

    # Copy files to current directory
    for layerpath in set(layer_paths):
        p = Path(layerpath)
        layer = str(p.name).split('-chi')[0]

        src = str(p)
        dest = str(p.name)

        if src != dest:
            print(('Copying building block '
                   'to current folder from {}').format(str(p)))
            shutil.copy(src, dest)

        # Also copy phonons if present
        phonons = (Path(layerpath).parent /
                   Path('{}-phonons.npz'.format(layer)))
        if not phonons.exists():
            continue
        src = str(phonons)
        dest = str(phonons.name)
        if src != dest:
            shutil.copy(src, dest)
            
    if args.substrate:
        substrate = np.load('{}-sub.npz'.format(str(args.substrate)))
    else:
        substrate = None

    # Make QEH calculation
    print('Initializing heterostructure')
    hs = make_heterostructure(layers,
                              thicknesses=args.thicknesses,
                              momenta=args.q,
                              frequencies=args.omega,
                              substrate=substrate)

    if args.plot:
        import matplotlib.pyplot as plt

    if args.plasmons:
        print('Calculate plasmon spectrum')
        tmp = hs.get_plasmon_eigenmodes(filename=args.plasmonfile)
        if args.plot:
            plot_plasmons(hs, tmp, plot_eigenvalues=args.eigenvalues,
                          plot_potential=args.potential,
                          plot_density=args.density,
                          save_plots=args.saveplots,
                          show=False)
    if args.eels:
        q_abs, frequencies, eels_qw = hs.get_eels(dipole_contribution=True)

    hs.timer.write()

    if args.plot:
        plt.show()


if __name__ == '__main__':
    main()
