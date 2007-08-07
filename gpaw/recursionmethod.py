import pickle

import Numeric as num
from ASE.Units import units, Convert

from gpaw.utilities.cg import CG
import gpaw.mpi as mpi


class RecursionMethod:
    """This class implements the Haydock recursion method. """

    def __init__(self, paw=None, filename=None,
                 tol=1e-10, maxiter=100):

        if paw is not None:
            assert not paw.spinpol # restricted - for now

            self.paw = paw
            self.tmp1_cG = paw.gd.zeros(3)
            self.tmp2_cG = paw.gd.zeros(3)
            self.z_cG = paw.gd.zeros(3)
            self.nkpts = self.paw.nmyu
            self.swaps = {}  # Python 2.4: use a set
            if paw.symmetry is not None:
                for swap, mirror in paw.symmetry.symmetries:
                    self.swaps[swap] = None

        self.tol = tol
        self.maxiter = maxiter
        
        if filename is not None:
            self.read(filename)
        else:
            self.initialize_start_vector()

    def read(self, filename):
        data = pickle.load(open(filename))
        self.a_kci, self.b_kci = data['ab']
        self.nkpts = data['nkpts']
        self.swaps = data['swaps']
        if 'arrays' in data:
            (self.w_kcG,
             self.wold_kcG,
             self.y_kcG) = data['arrays']
         
    def write(self, filename, mode=''):
        data = {'ab': (self.a_kci, self.b_kci),
                'nkpts': self.nkpts,
                'swaps': self.swaps}
        if mode == 'all':
            data['arrays'] = (self.w_kcG,
                              self.wold_kcG,
                              self.y_kcG)
        pickle.dump(data, open(filename, 'w'))
        
    def initialize_start_vector(self):
        # Create initial wave function:
        nkpts = self.nkpts
        self.w_kcG = self.paw.gd.zeros((nkpts, 3))
        for nucleus in self.paw.nuclei:
            if nucleus.setup.fcorehole != 0.0:
                break
        A_ci = nucleus.setup.A_ci
        if nucleus.pt_i is not None: # not all CPU's will have a contribution
            for k in range(nkpts):
                nucleus.pt_i.add(self.w_kcG[k], A_ci, k)

        self.wold_kcG = self.paw.gd.zeros((nkpts, 3))
        self.y_kcG = self.paw.gd.zeros((nkpts, 3))
            
        self.a_kci = num.zeros((nkpts, 3, 0), num.Float)
        self.b_kci = num.zeros((nkpts, 3, 0), num.Float)
        
    def run(self, nsteps):
        ni = self.a_kci.shape[2]
        a_kci = num.empty((self.nkpts, 3, ni + nsteps), num.Float)
        b_kci = num.empty((self.nkpts, 3, ni + nsteps), num.Float)
        a_kci[:, :, :ni]  = self.a_kci
        b_kci[:, :, :ni]  = self.b_kci
        self.a_kci = a_kci
        self.b_kci = b_kci

        for k in range(self.paw.nmyu):
            for i in range(nsteps):
                self.step(k, ni + i)
            
    def step(self, k, i):
        integrate = self.paw.gd.integrate
        w_cG = self.w_kcG[k]
        y_cG = self.y_kcG[k]
        wold_cG = self.wold_kcG[k]
        z_cG = self.z_cG
        
        self.solve(w_cG, self.z_cG, k)
        I_c = num.reshape(integrate(z_cG * w_cG)**-0.5, (3, 1, 1, 1))
        z_cG *= I_c
        w_cG *= I_c
        b_c = num.reshape(integrate(z_cG * y_cG), (3, 1, 1, 1))
        self.paw.kpt_u[k].apply_hamiltonian(self.paw.hamiltonian, 
                                            z_cG, y_cG)
        a_c = num.reshape(integrate(z_cG * y_cG), (3, 1, 1, 1))
        wnew_cG = (y_cG - a_c * w_cG - b_c * wold_cG)
        wold_cG[:] = w_cG
        w_cG[:] = wnew_cG
        self.a_kci[k, :, i] = a_c[:, 0, 0, 0]
        self.b_kci[k, :, i] = b_c[:, 0, 0, 0]

    def continued_fraction(self, e, k, c, i, imax):
        a_i = self.a_kci[k, c]
        b_i = self.b_kci[k, c]
        if i == imax - 2:
            return self.terminator(a_i[i], b_i[i], e)
        return 1.0 / (a_i[i] - e -
                      b_i[i + 1]**2 *
                      self.continued_fraction(e, k, c, i + 1, imax))

    def get_spectra(self, eps_n, delta=0.1, imax=None):
        assert not mpi.parallel
        n = len(eps_n)
        sigma_cn = num.zeros((3, n), num.Float)
        if imax is None:
            imax = self.a_kci.shape[2]
        energyunit = units.GetEnergyUnit()
        Ha = Convert(1, 'Hartree', energyunit)
        eps_n = (eps_n + delta * 1.0j) / Ha
        for k in range(self.nkpts):
            for c in range(3):
                sigma_cn[c] += self.continued_fraction(eps_n, k, c,
                                                       0, imax).imag

        if len(self.swaps) > 0:
            sigma0_cn = sigma_cn
            sigma_cn = num.zeros((3, n), num.Float)
            for swap in self.swaps:
                sigma_cn += num.take(sigma0_cn, swap)
            sigma_cn /= len(self.swaps)

        return sigma_cn
    
    def solve(self, w_cG, z_cG, k):
        self.paw.kpt_u[k].apply_inverse_overlap(self.paw.pt_nuclei,
                                                w_cG, self.tmp1_cG)
        self.k = k
        CG(self.A, z_cG, self.tmp1_cG,
           tolerance=self.tol, maxiter=self.maxiter)
        
    def A(self, in_cG, out_cG):
        """Function that is called by CG. It returns S~-1Sx_in in x_out
        """

        kpt = self.paw.kpt_u[self.k]
        kpt.apply_overlap(self.paw.pt_nuclei, in_cG, self.tmp2_cG)
        kpt.apply_inverse_overlap(self.paw.pt_nuclei, self.tmp2_cG, out_cG)

    def terminator(self, a, b, e):
        """ Analytic formula to terminate the continued fraction from
        [R Haydock, V Heine, and M J Kelly, J Phys. C: Solid State Physics, Vol 8, (1975), 2591-2605]
        """
        return 0.5 * (e - a - num.sqrt((e - a)**2 - 4 * b**2) / b**2)
