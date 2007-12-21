import numpy as npy
from gpaw.gllb import construct_density1D, find_reference_level1D, SMALL_NUMBER
from gpaw.gllb.responsefunctional import ResponseFunctional
from gpaw.gllb.nonlocalfunctional import NonLocalFunctionalDesc

from gpaw.mpi import world

SLATER_FUNCTIONAL = "X_B88-None"
K_G = 0.382106112167171

def gllb_weight(epsilon, reference_level):
    """
    Calculates the weight for GLLB functional.
    The parameter K_G is adjusted such that the correct result is obtained for
    exchange energy of non-interacting electron gas.

    All orbitals closer than 1e-5 Ha to fermi level are consider the
    give zero response. This is to improve convergence of systems with
    degenerate orbitals.

    =============== ==========================================================
    Parameters:
    =============== ==========================================================
    epsilon         The eigenvalue of current orbital
    reference_level The fermi-level of the system
    =============== ==========================================================
    """

    if (epsilon +1e-5 > reference_level):
        return 0

    return K_G * npy.sqrt(reference_level-epsilon)


class GLLB1DFunctional:
    """GLLB1DFunctional.

    For simplicity, the 1D-GLLB and 1D-KLI have moved to separate
    classes.  1D-codes are so simple, that there is not much trouble
    creating own class for each of them. Besides, it is a good idea to
    start implementing functionals from 1D-generator.  """

    def pass_stuff1D(self, ae):
        pass

    def get_slater1D(self, gd, n_g, u_j, f_j, l_j, vrho_xc, vbar=False):
        """Return approximate exchange energy.

        Used by get_non_local_energy_and_potential1D to calculate an
        approximation to 1D-Slater potential. Returns the exchange
        energy. 

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        gd          Radial grid descriptor
        n_g         The density
        u_j         The 1D-wavefunctions
        f_j         Occupation numbers
        l_j         The angular momentum numbers
        vrho_xc     The slater part multiplied by density is added to this
                    array.
        v_bar       hmmm
        =========== ==========================================================
        """
        # Create temporary arrays only once
        if self.v_g1D == None:
            self.v_g = n_g.copy()

        if self.e_g1D == None:
            self.e_g = n_g.copy()

        # Do we have already XCRadialGrid object, if not, create one
        if self.slater_part1D == None:
            from gpaw.xc_functional import XCFunctional, XCRadialGrid
            self.slater_part1D = XCRadialGrid(XCFunctional(SLATER_FUNCTIONAL, 1), gd)

        self.v_g[:] = 0.0
        self.e_g[:] = 0.0
        
        # Calculate B88-energy density
        self.slater_part1D.get_energy_and_potential_spinpaired(n_g, self.v_g, e_g=self.e_g)

        # Calculate the exchange energy
        Exc = npy.dot(self.e_g, gd.dv_g)

        # The Slater potential is approximated by 2*epsilon / rho
        vrho_xc[:] += 2 * self.e_g

        return Exc

    def get_response_weights1D(self,  u_j, f_j, e_j):
        """
          Calculates the weights for response part of GLLB functional.

          =========== ==========================================================
          Parameters:
          =========== ==========================================================
          u_j         The 1D-wave functions
          f_j         The occupation numbers
          e_j         Eigenvalues
          =========== ==========================================================
        """
        reference_level = find_reference_level1D(f_j, e_j)
        w_j = [ gllb_weight(e, reference_level) for e in e_j ]
        return w_j

    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j,
                                             v_xc, njcore=None, density=None,
                                             vbar=False):
        """Used by setup generator to calculate the one dimensional potential

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        gd          Radial grid descriptor
        u_j         The wave functions
        f_j         The occupation numbers
        e_j         The eigenvalues
        l_j         The angular momentum quantum numbers
        v_xc        V_{xc} is added to this array.
        nj_core     If njcore is set, only response part will be returned for
                    wave functions u_j[:nj_core]
        density     If density is supplied, it overrides the density
                    calculated from orbitals.
                    This is used is setup-generation.
        vbar        hmmm
        =========== ==========================================================
        """

        # Construct the density if required
        if density == None:
            n_g = construct_density1D(gd, u_j, f_j)
        else:
            n_g = density

        # Construct the slater potential if required
        if njcore == None:
            # Get the slater potential multiplied by density
            Exc = self.get_slater1D(gd, n_g, u_j, f_j, l_j, v_xc, vbar=vbar)
            # Add response from all the orbitals
            imax = len(f_j)
        else:
            # Only response part of core orbitals is desired
            v_xc[:] = 0.0
            # Add the potential only from core orbitals
            imax = njcore
            Exc = 0

        # Get the response weigths
        w_j = self.get_response_weights1D(u_j, f_j, e_j)

        # Add the response multiplied with density to potential
        v_xc[:] += construct_density1D(gd, u_j[:imax], [f*w for f,w in zip(f_j[:imax] , w_j[:imax])])

        # Divide with the density, beware division by zero
        v_xc[1:] /= n_g[1:] + SMALL_NUMBER

        # Fix the r=0 value
        v_xc[0] = v_xc[1]

        return Exc

    # input:  ae : AllElectron object.
    # output: extra_xc_data : dictionary. A Dictionary with pair ('name', radial grid)
    def calculate_extra_setup_data(self, extra_xc_data, ae):
        """
        For GLLB-functional one needs the response part of core orbitals to be stored in setup file,
        which is calculated in this section.

        ============= ==========================================================
        Parameters:
        ============= ==========================================================
        extra_xc_data Input: an empty dictionary
                      Output: dictionary with core_response-keyword containing data
        ae            All electron object containing all important data for calculating the core response.
        ============= ==========================================================

        """

        # Allocate new array for core_response
        N = len(ae.rgd.r_g)
        v_xc = npy.zeros(N)

        # Calculate the response part using wavefunctions, eigenvalues etc. from AllElectron calculator
        self.get_non_local_energy_and_potential1D(ae.rgd, ae.u_j, ae.f_j, ae.e_j, ae.l_j, v_xc,
                                                  njcore = ae.njcore)

        extra_xc_data['core_response'] = v_xc.copy()

        if self.relaxed_core_response:

            for nc in range(0, ae.njcore):
                # Add the response multiplied with density to potential
                orbital_density = construct_density1D(ae.rgd, ae.u_j[nc], ae.f_j[nc])
                extra_xc_data['core_orbital_density_'+str(nc)] = orbital_density
                extra_xc_data['core_eigenvalue_'+str(nc)] = [ ae.e_j[nc] ]

            extra_xc_data['njcore'] = [ ae.njcore ]
        



#################################################################################
#                                                                               #
# Implementation of GLLB begins                                                 #
#                                                                               #
#################################################################################

class GLLBFunctional(ResponseFunctional, GLLB1DFunctional):
    """GLLB Functional.
    
    Calculates the energy and potential determined by GLLB-Functional
    [1]_. This functional:
    
    1) approximates the numerator part of Slater-potential from
       2*GGA-energy density. This implementation follows the orginal
       authors and uses the Becke88-functional.

    2) approximates the response part coefficients from eigenvalues,
       given correct result for non-interacting electron gas.

    .. [1] Gritsenko, Leeuwen, Lenthe, Baerends: Self-consistent
       approximation to the Kohn-Shan exchange potential Physical
       Review A, vol. 51, p. 1944, March 1995.

    GLLB-Functional is of the same form than KLI-Functional, but it ...

    """

    def __init__(self, relaxed_core_response=False, lumo=False):
        """Initialize GLLB Functional class.

        About relax_core_resonse flag:
       
        Normally, core response is calculated using reference-level of
        setup-generator.  If relax_core_response is true, the
        GLLB-coefficients for core response are recalculated using
        current reference-level. That is::

          v^{resp,core} = sum_i^{core} K_G sqrt{epsilon_i - epsilon_f} |psi_i|^2 / rho.
        
        As usually in frozen core approximation, the core orbital
        psi_i, and core energy epsilon_i are kept fixed.

        About lumo flag:

        Normally, the reference energy (epsilon_f in the article [1])
        is set to HOMO of the system.  However, if lumo==True, the
        reference energy is set to LUMO of the system.
        """

        ResponseFunctional.__init__(self, relaxed_core_response)

        self.v_g1D = None
        self.e_g1D = None
        self.slater_part1D = None
        self.slater_part = None
        self.xcname = 'GLLB'
        self.lumo = lumo

    def pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation, kpt_comm):
        ResponseFunctional.pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation, kpt_comm)

        # Temporary arrays needed for evaluating the Slater part of GLLB
        self.tempvxc_g = finegd.zeros()
        self.tempe_g = finegd.zeros()

    def get_functional_desc(self):
        """
        Retruns info for GLLBFunctional. The GLLB-functional needs density, gradient, wavefunctions
        and eigenvalues.

        """
        return NonLocalFunctionalDesc(True, True, True, True)

    def find_reference_level(self, info_s):
        if self.lumo:
            c = -1
        else:
            c = 1

        # In previous version there was a maximum taken over spin.
        # This was wrong, since exchange interaction does not affect different spins.
        # Now the problem is fixed.

        # Find the reference level for each spin
        # First over own eigenvalues of this processor, then over all k-points.
        return [ c*self.kpt_comm.max(c*find_reference_level1D(info['f_n'], info['eps_n'], lumo=self.lumo)) for info in info_s ]

    def ensure_B88(self):
        # Create the B88 functional for Slater part (only once per calculation)
        if self.slater_part == None:
            from gpaw.xc_functional import XCFunctional
            self.slater_part = XCFunctional(SLATER_FUNCTIONAL, 1)
            self.initialization_ready = True

    def get_slater_part_paw_correction(self, rgd, n_g, a2_g, v_g, pseudo = True, ndenom_g=None):

        if ndenom_g == None:
            ndenom_g = n_g

        # TODO: This method needs more arguments to support arbitary slater part
        self.ensure_B88()
        N = len(n_g)
        # TODO: Allocate these only once
        vtemp_g = npy.zeros(N)
        etemp_g = npy.zeros(N)
        deda2temp_g = npy.zeros(N)

        self.slater_part.calculate_spinpaired(etemp_g, n_g, vtemp_g, a2_g, deda2temp_g)

        # Grr... When n_g = 0, B88 returns -0.03 for e_g!!!!!!!!!!!!!!!!!
        etemp_g[:] = npy.where(abs(n_g) < SMALL_NUMBER, 0, etemp_g)

        v_g[:] = 2 * etemp_g / (ndenom_g + SMALL_NUMBER)

        return npy.sum(etemp_g * rgd.dv_g)

    def get_slater_part(self, info_s, v_sg, e_g):

        self.ensure_B88()

        # Calculate the Slater-part

        deg = self.nspins

        # Go through all spin densities
        for s, (v_g, info) in enumerate(zip(v_sg, info_s)):
            # Calculate the slater potential. self.tempvxc_g and self.vt_g are used just for dummy
            # arrays and they are not used after calculation. Fix?
            self.slater_part.calculate_spinpaired(self.tempe_g, deg*info['n_g'], self.tempvxc_g,
                                                  a2_g = deg*deg*info['a2_g'], deda2_g = self.vt_g)

            # Add it to the total potential
            v_g[:] += 2*self.tempe_g / (deg * (info['n_g']) + SMALL_NUMBER)

            # Add the contribution of this spin to energy density
            e_g[:] += self.tempe_g.ravel() / deg

    def get_slater_part_and_weights(self, info_s, v_sg, e_g):

        self.get_slater_part(info_s, v_sg, e_g)

        # Find out the coefficients

        # First, locate the reference-levels (of each spins)
        self.reference_levels = self.find_reference_level(info_s) 
        w_sn =  [ [ gllb_weight(e, reference_level) for e in info['eps_n'] ] for info, reference_level in zip(info_s, self.reference_levels) ]

        return w_sn

