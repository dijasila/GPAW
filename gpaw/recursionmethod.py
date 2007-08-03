
import Numeric as num
from Numeric import NewAxis
import gpaw.mpi as mpi
import gpaw.tddft.BiCGStab as BiCGStab
from gpaw.utilities.cg import CG
import math

class RecursionMethod:
    """This class implements the Haydock recursion method. """

    def __init__(self, paw=None, filename=None,
                 tol=1e-10, max_iter=100):

        self.paw = paw
        
        if filename is not None:
            self.read(filename)
        else:
            self.initialize_start_vector()

    def initialize_start_vector(self):
        # Create initial wave function:
        self.w_cG = self.paw.gd.zeros(3)
        for nucleus in self.calc.nuclei:
            if nucleus.setup.fcorehole != 0.0:
                break
        A_ci = nucleus.setup.A_ci
        if nucleus.pt_i is not None: # not all CPU's will have a contribution
            nucleus.pt_i.add(self.w_cG, A_ci)

        self.wold_cG = self.paw.gd.zeros(3)
        self.y_cG = self.paw.gd.zeros(3)
        self.z_cG = self.paw.gd.empty(3)
            
        self.a_ic = []
        self.b_ic = []
        
    def run(self, nsteps):
        for i in range(nsteps):
            self.step()
            
    def step(self):
        self.solve(self.w_cG, self.z_cG)
        I_c = num.reshape(self.gd.integrate(self.z_cG * self.w_cG)**-0.5,
                          (3, 1, 1, 1))
        self.z_cG *= I_c
        self.w_cG *= I_c
        b_c = num.reshape(self.gd.integrate(self.z_cG * self.y_cG),
                          (3, 1, 1, 1))
        self.paw.kpt_u[0].apply_hamiltonian(self.paw.hamiltonian, 
                                            self.z_cG, self.y_cG)
        a_c = num.reshape(self.gd.integrate(self.z_cG * self.y_cG)
                          (3, 1, 1, 1))
        wnew_cG = (self.y_cG - a_c * self.w_cG - b_c * self.wold_cG)
        self.wold_cG = self.w_cG
        self.w_cG = wnew_cG
        self.a_ic.append(a_c[:, 0, 0, 0])
        self.b_ic.append(b_c[:, 0, 0, 0])

    def continued_fraction(self, e, c, i=0, imax=None):
        if imax is None:
            imax = len(self.a_ic)
        if i == imax - 2:
            return self.terminator(self.a_ic[i][c], self.b_ic[i][c], e)
        return 1.0 / (self.a_ic[i][c] - e -
                      self.b_ic[i + 1][c]**2 *
                      self.continued_fraction(e, c, i + 1, imax))

    def solve(self, in_cG, out_cG):
        self.paw.kpt_u[0].apply_inverse_overlap(self.paw.pt_nuclei,
                                                in_cG, self.tmp_cG)
        CG(self.A, out_cG, tmp1_cG, tolerance=self.tol)
        
    def A(self, in_cG, out_cG):
        """Function that is called by CG. It returns S~-1Sx_in in x_out
        """

        kpt = self.paw.kpt_u[0]
        kpt.apply_overlap(self.paw.pt_nuclei, in_cG, self.tmp2_cG)
        kpt.apply_inverse_overlap(self.paw.pt_nuclei, self.tmp2_cG, out_cG)

    def terminator(self, a, b, e):
        """ Analytic formula to terminate the continued fraction from
        [R Haydock, V Heine, and M J Kelly, J Phys. C: Solid State Physics, Vol 8, (1975), 2591-2605]
        """
        return 0.5 * (e - a - num.sqrt((e - a)**2 - 4 * b**2) / b**2)
