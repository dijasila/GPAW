import numpy as np
from scipy.interpolate import interp1d
from ase.units import Bohr, Hartree as Ha
from gpaw import GPAW, PW
from ase.parallel import parprint


class ElectrostaticCorrections:
    def __init__(self, pristine, defect, q, FWHM, model='gaussian'):
        if isinstance(pristine, str):
            pristine = GPAW(pristine, txt=None, parallel={'domain': 1})
        if isinstance(defect, str):
            defect = GPAW(defect, txt=None)
        calc = GPAW(mode=PW(500, force_complex_dtype=True),
                    kpts={'size': (1, 1, 1),
                          'gamma': True},
                    parallel={'domain': 1},
                    symmetry='off')
        atoms = pristine.atoms.copy()
        calc.initialize(atoms)
        self.pristine = pristine
        self.defect = defect
        self.calc = calc
        self.q = q
        self.FWHM = FWHM
        self.sigma = self.FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        self.model = model
        self.pd = self.calc.wfs.pd
        self.G_Gv = self.pd.get_reciprocal_vectors(q=0,
                                                   add_q=False)  # G in Bohr^-1
        self.G2_G = self.pd.G2_qG[0]  # |\vec{G}|^2 in Bohr^-2
        if model == 'gaussian':
            self.rho_G = self.calculate_gaussian_density()
        self.Omega = atoms.get_volume() / (Bohr ** 3.0)  # Cubic Bohr
        self.data = None
        self.El = None

    def calculate_potentials(self, epsilon):
        if self.data is not None and self.data['epsilon'] == epsilon:
            return self.data
        r_3xyz = self.pristine.density.gd.refine().get_grid_point_coordinates()
        nrz = np.shape(r_3xyz)[3]
        z = (1.0 * r_3xyz[2].flatten())[:nrz]
        V_0 = - self.pristine.get_electrostatic_potential().mean(0).mean(0)
        V_X = - self.defect.get_electrostatic_potential().mean(0).mean(0)
        z_model, V_model = self.calculate_z_avg_model_potential(epsilon)
        V_model = interp1d(z_model[:-1], V_model[:-1], kind='cubic')(z)
        D_V_mean = self.average(V_model - V_X + V_0, z_model)
        data = {'epsilon': epsilon,
                'z': z,
                'V_0': V_0,
                'V_X': V_X,
                'V_model': V_model,
                'D_V': V_model - V_X + V_0,
                'D_V_mean': D_V_mean}
        self.data = data
        return data

    def average(self, V, z):
        N = len(V)
        middle = N // 2
        points = range(middle - N // 8, middle + N // 8 + 1)
        restricted = V[points]
        V_mean = np.mean(restricted)
        return V_mean
    
    def calculate_formation_energies(self, epsilon):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.defect.get_potential_energy()
        El = self.calculate_lattice_electrostatics(epsilon)
        self.El = El
        data = self.calculate_potentials(epsilon)
        Delta_V = self.average(data['D_V'], data['z'])
        return (E_X - E_0, E_X - E_0 - El + Delta_V * self.q)
    
    def calculate_gaussian_density(self):
        # Fourier transformed gaussian:
        rho_G = self.q * np.exp(-0.5 * self.G2_G * self.sigma * self.sigma)
        return rho_G

    def calculate_lattice_electrostatics(self, epsilon):
        if self.El is not None and self.El['epsilon'] == epsilon:
            return self.El['El']
        Elp = 0.0
        for rho, G2 in zip(self.rho_G, self.G2_G):
            if np.isclose(G2, 0.0):
                parprint('Skipping G^2=0 contribution to Elp')
                continue
            Elp += 2.0 * np.pi / (epsilon * self.Omega) * rho * rho / G2
        El = (Elp - self.q * self.q
              / (2 * epsilon * self.sigma * np.sqrt(np.pi))) * Ha
        self.El = {'El': El, 'epsilon': epsilon}
        return El

    def calculate_z_avg_model_potential(self, epsilon):
        r_3xyz = self.calc.density.gd.refine().get_grid_point_coordinates()

        nrz = np.shape(r_3xyz)[3]

        cell = self.calc.atoms.get_cell()
        vox3 = cell[2, :] / Bohr / nrz

        # The grid is arranged with z increasing fastest, then y
        # then x (like a cube file)
        z_g = 1.0 * r_3xyz[2].flatten()

        selectedG = []
        for iG, G in enumerate(self.G_Gv):
            if np.isclose(G[0], 0.0) and np.isclose(G[1], 0.0):
                selectedG.append(iG)

        assert(np.isclose(self.G2_G[selectedG[0]], 0.0))
        selectedG.pop(0)
        zs = []
        Vs = []
        for idx in np.arange(nrz):
            phase_G = np.exp(1j * (self.G_Gv[selectedG, 2] * z_g[idx]))
            V = (np.sum(phase_G * self.rho_G[selectedG]
                        / (self.G2_G[selectedG]))
                 * Ha * 4.0 * np.pi / (epsilon * self.Omega))
            zs.append(z_g[idx])
            Vs.append(V)

        V = (np.sum(self.rho_G[selectedG] / (self.G2_G[selectedG]))
             * Ha * 4.0 * np.pi / (epsilon * self.Omega))
        zs.append(vox3[2])
        Vs.append(V)
        return np.array(zs), np.array(Vs)
