import numpy as np
import numbers
from scipy.optimize import minimize
from scipy.integrate import simps
from gpaw.defects import ElectrostaticCorrections
from ase.parallel import parprint
import scipy.integrate as integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from gpaw.wavefunctions.pw import PWDescriptor


class ElectrostaticCorrections2D(ElectrostaticCorrections):
    def __init__(self, pristine, neutral, charged, q, FWHM, z0,
                 model='gaussian'):
        ElectrostaticCorrections.__init__(self, pristine, neutral,
                                          charged, q, FWHM)
        self.z0 = z0
        self.L = self.pd.gd.cell_cv[2, 2]

        # We assume that the dielectric profile epsilon(z) follows the
        # electronic density of the pristine system n(z).
        density = self.pristine.get_pseudo_density(gridrefinement=2)
        density_1d = density.mean(axis=(0, 1))
        coarse_z = find_z(self.pristine.density.gd.refine())
        fine_z = find_z(self.pd.gd.refine())
        transformer = InterpolatedUnivariateSpline(coarse_z, density_1d)
        self.density_1d = np.array([transformer(x) for x in fine_z])
        self.z_g = fine_z

        self.G_z = find_G_z(self.pd)
        self.G_parallel = np.unique(self.G_Gv[:, :2], axis=0)
        self.GG = np.outer(self.G_z, self.G_z)  # G * Gprime

        # We need G-vectors up to twice the normal length in order to evaulate
        # epsilon(G - G'). We could manipulate the existing ones, or we could
        # just be lazy and make a new plane wave descriptor with four times the
        # cutoff:
        pd2 = PWDescriptor(self.pd.ecut * 4, self.pd.gd.refine(),
                           dtype=complex)
        self.G_z_fine = find_G_z(pd2)
        self.index_array = self.get_index_array()

        self.epsilons_z = None
        self.Eli = None
        self.Elp = None
        self.epsilon_GG = None
        
    def set_epsilons(self, epsilons):
        """Set the bulk dielectric constant of the system corresponding to the atomic
        configuration of the pristine system. This is used to calculate the
        screened coulomb potential of the gaussian model distribution that we
        use. In 2D, the dielectric constant is poorly defined. However, when we
        do DFT calculations with periodic systems, we are always working with
        pseudo bulk-like structures, and it is possible to calculate some
        dielectric constant. The screening of the system must come from where
        there is an electronic density, and we thus set

        epsilon(z) - 1 \propto n(z),

        under the constraint that the total screening int dz epsilon(z) gives
        the correct value.

        Parameters:
          epsilons: A float, or a list of two floats. If a single
            float is given, the screening in-plane and out-of-plane is assumed
            to be the same. If two floats are given, the first represents the
            in-plane response, and the second represents the out-of-plane
            response.

        Returns:
            epsilons_z: dictionary containnig the normalized dielectric
              profiles in the in-plane and out-of-plane directions along the z
              axis.
        """
        # Reset the state of the object; the corrections need to be
        # recalculated
        self.Elp = None
        self.Eli = None
        self.epsilon_GG = None

        if isinstance(epsilons, numbers.Number):
            epsilons = [epsilons, epsilons]
        elif len(epsilons) == 1:
            epsilons = [epsilons[0]] * 2
        self.epsilons = np.array(epsilons)

        z = self.z_g
        density_1d = self.density_1d
        L = z[-1] - z[0]
        N = simps(density_1d, z)
        epsilons_z = np.zeros((2,) + np.shape(density_1d))
        epsilons_z += density_1d

        # In-plane
        epsilons_z[0] = epsilons_z[0] * (epsilons[0] - 1) / N * L + 1

        # Out-of-plane
        def objective_function(k):
            k = k[0]
            integral = simps(1 / (k * density_1d + 1), z) / L
            return np.abs(integral - 1 / epsilons[1])

        test = minimize(objective_function, [1], method='Nelder-Mead')
        assert test.success, "Unable to normalize dielectric profile"
        k = test.x[0]
        epsilons_z[1] = epsilons_z[1] * k + 1
        self.epsilons_z = {'in-plane': epsilons_z[0],
                           'out-of-plane': epsilons_z[1]}

    def get_index_array(self):
        """
        Calculate the indices that map between the G vectors on the fine grid,
        and the matrix defined by M(G, Gprime) = G - Gprime.

        We use this to generate the matrix epsilon_GG = epsilon(G - Gprime)
        based on knowledge of epsilon(G) on the fine grid.
        """
        G, Gprime = np.meshgrid(self.G_z, self.G_z)
        difference_GG = G - Gprime
        index_array = np.zeros(np.shape(difference_GG))
        index_array[:] = np.nan
        for idx, val in enumerate(self.G_z_fine):
            mask = np.isclose(difference_GG, val)
            index_array[mask] = idx
        if np.isnan(index_array).any():
            print("Missing indices found in mapping between plane wave "
                  "sets. Something is wrong with your G-vectors!")
            print(np.isnan(index_array).all())
            print(np.where(np.isnan(index_array)))
            print(self.G_z_fine)
            print(difference_GG)
            assert False
        return index_array.astype(int)
        
    def calculate_epsilon_GG(self):
        assert self.epsilons_z is not None, ("You provide the dielectric "
                                             "constant first!")
        N = len(self.density_1d)
        epsilon_par_G = np.fft.fft(self.epsilons_z['in-plane']) / N
        epsilon_perp_G = np.fft.fft(self.epsilons_z['out-of-plane']) / N
        epsilon_par_GG = epsilon_par_G[self.index_array]
        epsilon_perp_GG = epsilon_perp_G[self.index_array]

        self.epsilon_GG = {'in-plane': epsilon_par_GG,
                           'out-of-plane': epsilon_perp_GG}

    def calculate_periodic_correction(self):
        G_z = self.G_z
        if self.Elp is not None:
            return self.Elp
        if self.epsilon_GG is None:
            self.calculate_epsilon_GG()
        Elp = 0.0

        for vector in self.G_parallel:
            norm_G = (np.dot(vector, vector) + G_z * G_z)
            rho_G = self.q * np.exp(- norm_G * self.sigma ** 2 / 2)
            A_GG = (self.GG * self.epsilon_GG['out-of-plane']
                    + np.dot(vector, vector) * self.epsilon_GG['in-plane'])
            if np.allclose(vector * vector, 0):
                A_GG[0, 0] = 1  # The d.c. potential is poorly defined
            V_G = np.linalg.solve(A_GG, rho_G)
            if np.allclose(vector * vector, 0):
                parprint('Skipping G^2=0 contribution to Elp')
                V_G[0] = 0  # So we skip it!
            Elp += (rho_G * V_G).sum()

        Elp *= 2 * np.pi / self.Omega
        self.Elp = Elp
        return Elp

    def calculate_isolated_correction(self):
        if self.Eli is not None:
            return self.Eli
        
        L = self.L
        G_z = self.G_z

        G, Gprime = np.meshgrid(G_z, G_z)
        gaussian = (G ** 2 + Gprime ** 2) * self.sigma ** 2 / 2

        prefactor = np.exp(- gaussian)

        if self.epsilon_GG is None:
            self.calculate_epsilon_GG()

        dE_GG_par = (self.epsilon_GG['in-plane'] - 1)
        dE_GG_perp = (self.epsilon_GG['out-of-plane'] - 1)

        def integrand(k):
            K_G = (L * (k ** 2 + G_z ** 2) /
                   (1 - np.exp(-k * L / 2) * np.cos(L * G_z / 2)))
            K_GG = np.diag(K_G)
            D_GG = K_GG + L * 0 * (self.GG * dE_GG_perp
                                   + k ** 2 * dE_GG_par)
            return (k * np.exp(-k ** 2 * self.sigma ** 2) *
                    (prefactor * 1 / (D_GG)).sum())

        Eli = self.q * self.q * integrate.quad(integrand, 1e-10, np.inf,
                                               limit=500)
        self.debug_k = np.linspace(0, 10, 1000)
        self.debug_U = np.array([integrand(x) for x in self.debug_k])
        self.Eli = Eli
        return Eli
    
    def calculate_potential_alignment(self):
        V_neutral = np.mean(self.neutral.get_electrostatic_potential(), (0, 1))
        V_charged = np.mean(self.charged.get_electrostatic_potential(), (0, 1))

        return ((V_charged[0] + V_charged[-1]) / 2 -
                (V_neutral[0] + V_neutral[-1]) / 2)


def find_G_z(pd):
    G_Gv = pd.get_reciprocal_vectors(q=0)
    mask = (G_Gv[:, 0] == 0) & (G_Gv[:, 1] == 0)
    G_z = G_Gv[mask][:, 2]  # G_z vectors in Bohr^{-1}
    return G_z


def find_z(gd):
    r3_xyz = gd.get_grid_point_coordinates()
    nrz = r3_xyz.shape[2]
    return r3_xyz[2].flatten()[:nrz]


def gaussian(z, sigma=1, mu=0):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * (z - mu)**2
                                                     / sigma ** 2)
