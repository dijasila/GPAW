# Imports
import numpy as num
from gpaw.xc_functional import XC3DGrid, XCFunctional
SMALL_NUMBER = 1e-10
SMALL_NUMBER_ATOM = 1e-10

# Some useful constants
EXCHANGE_FUNCTIONAL = "X_B88"
CORRELATION_FUNCTIONAL = "C_PW91"
K_G = 0.382106112167171

# Few useful functions
def find_nucleus(nuclei, a):
    nucleus = None
    for nuc in nuclei:
        if a == nuc.a:
            nucleus = nuc
    assert(nucleus is not None)
    return nucleus

def safe_sqr(u_j):
    return u_j**2
    #OBSOLETE!
    #Thanks to numpy, no more safe_sqr
    #return num.where(abs(u_j) < 1e-160, 0, u_j)**2

def construct_density1D(gd, u_j, f_j):
    """
    Creates one dimensional density from specified wave functions and occupations.

    =========== ==========================================================
    Parameters:
    =========== ==========================================================
    gd          Radial grid descriptor
    u_j         The wave functions
    f_j         The occupation numbers
    =========== ==========================================================
    """


    n_g = num.dot(f_j, safe_sqr(u_j))
    n_g[1:] /=  4 * num.pi * gd.r_g[1:]**2
    n_g[0] = n_g[1]
    return n_g

def find_reference_level1D(f_j, e_j, lumo=False):
    """Finds the reference level from occupations and eigenvalue energies.
    
    Uses tolerance 1e-3 for occupied orbital.

    =========== ==========================================================
    Parameters:
    =========== ==========================================================
    f_j         The occupations list
    e_j         The eigenvalues list
    lumo        If lumo==True, find LUMO energy instead of HOMO energy.
    =========== ==========================================================
    """
    
    if lumo:
        lumo_level = 1000
        for f,e in zip(f_j, e_j):
            if f < 1e-3:
                if lumo_level > e:
                    lumo_level = e
        return lumo_level

    homo_level = -1000
    for f,e in zip(f_j, e_j):
        if f > 1e-3:
            if homo_level < e:
                homo_level = e
    return homo_level

class GLLBFunctional:

    def __init__(self, relaxed_core_response = False, lumo_reference = False, correlation = False, mixing = 1.0, slater_xc_name = None):
        """Initialize GLLB Functional class.

        About relax_core_resonse flag:

        Normally, core response is calculated using reference-level of
        setup-generator.  If relax_core_response is true, the
        GLLB-coefficients for core response are recalculated using
        current reference-level. That is::

          v^{resp,core} = sum_i^{core} K_G sqrt{epsilon_f - epsilon_i} |psi_i|^2 / rho.

        As usually in frozen core approximation, the core orbital
        psi_i, and core energy epsilon_i are kept fixed.

        About lumo_reference flag:

        Normally, the reference energy (epsilon_f in the article [1])
        is set to HOMO of the system.  However, if lumo==True, the
        reference energy is set to LUMO of the system.

        About correlation flag:

        Correlation is PW91.
        """

        self.relaxed_core_response = relaxed_core_response
        self.lumo_reference = lumo_reference
        self.correlation = correlation
        self.mixing = mixing

        self.slater_xc_name = slater_xc_name

        self.old_v_sg = []
        self.gga_xc = None

        self.v_g1D = None
        self.e_g1D = None
        self.gga_xc1D = None

        self.initialized = False
        self.reference_level_s = [ -1000 ]

    def gllb_weight(self, epsilon, reference_level):
        """
        Calculates the weight for GLLB functional.
        The parameter K_G is adjusted such that the correct result is obtained for
        exchange energy of non-interacting electron gas.
        
        =============== ==========================================================
        Parameters:
        =============== ==========================================================
        epsilon         The eigenvalue of current orbital
        reference_level The fermi-level of the system
        =============== ==========================================================
        """
        # 0.05 eV means degenerate
        if (epsilon + 0.05 / 27.21> reference_level):
            return 0.0

        diff = reference_level-epsilon
        return K_G * num.sqrt(diff) 

    def calculate_spinpaired(self, e_g, n_g, v_g):
        """Calculates the KS-exchange potential for spin paired calculation
           and adds it to v_g. Supplies also the energy density.

           This method is called from xc_functional.py.

           =========== ==========================================================
           Parameters:
           =========== ==========================================================
           e_g         The energy density
           n_g         The electron density
           v_g         The Kohn-Sham potential.
           =========== ==========================================================

        """

        # Calculate the exchange potential
        self.calculate_gllb([n_g], [v_g], e_g)

        # Mix the potential
        # Whatever is already in v_g gets mixed too. This is ok for now, but needs to be checked later
        self.potential_mixing([v_g])

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        """Calculates the KS-exchange potential for spin polarized calculation
        and adds it to v_g. Supplies also the energy density.

        This method is called from xc_functional.py.

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        e_g          The energy density
        na_g         The electron density for spin alpha
        va_g         The Kohn-Sham potential for spin alpha
        na_g         The electron density for spin beta
        va_g         The Kohn-Sham potential for spin beta
        =========== ==========================================================

        """
        self.calculate_gllb([na_g, nb_g], [va_g, vb_g], e_g)

        # Mix the potentials
        # Whatever is already in va_g or vb_g gets mixed too. This is ok for now, but needs to be checked later
        # how this affects convergence.
        self.potential_mixing([v_ag, vb_g])


    def potential_mixing(self, v_sg):
        """
        Perform the potential mixing.
 
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        v_sg        A python list containing the potentials to be mixed.
                    v_sg contains one potential for spin-paired calculation
                    and two potentials for spin polarized calculation.
        =========== ==========================================================
        """

        # If old_vt_gs has not been allocated yet, or the potential shape has changed
        if (len(self.old_v_sg) != len(v_sg)) or (self.old_v_sg[0].shape != v_sg[0].shape):
            # Create a copy from the orginal potential to be old_vt_g
            print "Updated potential-mixer!"
            self.old_v_sg = [ v_g.copy() for v_g in v_sg ]

        # Mix the potentials
        for v_g, old_v_g in zip(v_sg, self.old_v_sg):
            v_g[:] = self.mixing * v_g[:] + (1.0 - self.mixing) * old_v_g[:]
            old_v_g[:] = v_g[:]


    def set_reference_index(self, index):
        """Set the reference band index, which is set to HOMO/LUMO"""
        self.reference_index = index

    def get_reference_index(self):
        """Get the reference band index pointing to HOMO/LUMO"""
        return self.reference_index


    def is_ready(self):
        try:
            self.kpt_u[0].eps_n
        except AttributeError:
            print "Attribute error."
            return False
        return True

    def find_reference_level(self):
        if self.lumo_reference:
            c = -1
        else:
            c = 1

        reference_level_s= []
        # For each spin
        for s in range(0, self.nspins):
            level = -c*1000
            # For each k-point
            for kpt in self.kpt_u:
                if s == kpt.s:
                    eps = kpt.eps_n[self.reference_index[s]]
                    if c*level < c*eps:
                        level = eps

            level = c*self.kpt_comm.max(c*level)
            reference_level_s.append(level)

        return reference_level_s


    def update(self):
        # Locate the reference levels
        self.reference_level_s = self.find_reference_level()
        self.initialized = True

    def calculate_gllb(self, n_sg, v_sg, e_g):
        # Only spin-paired calculation supported
        assert(self.nspins == 1)

        if not self.is_ready():
            return

        # Calculate the Slater-part of exchange (and correlation) potential
        self.prepare_gga_xc()
        Exc = self.gga_xc.get_energy_and_potential_spinpaired(n_sg[0], self.v_g, e_g=self.e_g)
        v_sg[0] += 2 * self.e_g / (n_sg[0] + SMALL_NUMBER)
        e_g [:]= self.e_g.flat

        # Use the coarse grid for response part
        # Calculate the coarse response multiplied with density and the coarse density
        # and to the division at the end of the loop.
        self.vt_G[:] = 0.0

        # For each k-point, add the response part
        for kpt in self.kpt_u:
            w_n = self.get_weights_kpoint(kpt)
            for f, psit_G, w in zip(kpt.f_n, kpt.psit_nG, w_n):
                if w > 0:
                    if kpt.dtype == float:
                        self.vt_G += f * w * (psit_G **2)
                    else:
                        self.vt_G += f * w * (psit_G * num.conjugate(psit_G)).real

        # Communicate the coarse-response part
        self.kpt_comm.sum(self.vt_G)

        # Include the symmetry to the response part also
        if self.symmetry is not None:
            self.symmetry.symmetrize(self.vt_G, self.gd)

        # Interpolate the response part to fine grid
        self.vt_g[:] = 0.0 
        self.interpolate(self.vt_G, self.vt_g)

        # Add the response part to the potential
        v_sg[0] += self.vt_g / (n_sg[0] + SMALL_NUMBER)

    def pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation, kpt_comm, symmetry, fixdensity):
        """
        Important quanities is supplied to non-local functional using this method.

        Called from xc_functional::set_non_local_things method
        All the necessary classes and methods are passed through this method
        Not used in 1D-calculations.
        """

        self.kpt_u = kpt_u
        self.gd = gd
        self.finegd = finegd
        self.interpolate = interpolate
        self.nspins = nspins
        self.nuclei = nuclei
        self.occupation = occupation
        self.kpt_comm = kpt_comm
        self.symmetry = symmetry

        # Allocate stuff needed for potential calculation
        self.v_g = finegd.empty()
        self.e_g = finegd.empty()
        self.vt_G = gd.empty()
        self.vt_g = finegd.empty()
        if fixdensity:
            self.mixing = 0.0

    def get_gga_xc_name(self):
        if self.slater_xc_name is not None:
            return self.slater_xc_name

        xcname = EXCHANGE_FUNCTIONAL
        if self.correlation:
            xcname += '-' + CORRELATION_FUNCTIONAL
        else:
            xcname += '-None'
        print "Using GGA-functional ", xcname, " for screening part"
        return xcname

    def prepare_gga_xc(self):
        # Create the exchange and correlation functional for screening part (only once per calculation)
        if self.gga_xc == None:
            xcname = self.get_gga_xc_name()
            from gpaw.xc_functional import XCFunctional
            self.gga_xc = XC3DGrid(XCFunctional(xcname, 1), self.finegd, self.nspins)

    def prepare_gga_xc_1D(self, gd):
        # Do we have already XCRadialGrid object, if not, create one
        if self.gga_xc1D == None:
            xcname = self.get_gga_xc_name()
            from gpaw.xc_functional import XCFunctional, XCRadialGrid
            self.gga_xc1D = XCRadialGrid(XCFunctional(xcname, 1), gd)

    def get_slater1D(self, gd, n_g, u_j, f_j, l_j, vrho_xc):
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

        self.prepare_gga_xc_1D(gd)

        self.v_g[:] = 0.0
        self.e_g[:] = 0.0
        
        # Calculate B88-energy density
        self.gga_xc1D.get_energy_and_potential_spinpaired(n_g, self.v_g, e_g=self.e_g)

        # Calculate the exchange energy
        Exc = num.dot(self.e_g, gd.dv_g)

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
        w_j = [ self.gllb_weight(e, reference_level) for e in e_j ]
        return w_j

    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j,
                                             v_xc, iteration, njcore=None, density = None):
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
            Exc = self.get_slater1D(gd, n_g, u_j, f_j, l_j, v_xc)
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

        if iteration < 10:
            NUMBER = SMALL_NUMBER_ATOM
        else:
            NUMBER = SMALL_NUMBER

        if njcore == None:
        # Divide with the density, beware division by zero
            v_xc[1:] /= n_g[1:] + NUMBER

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
        v_xc = num.zeros(N, float)

        # Calculate the response part using wavefunctions, eigenvalues etc. from AllElectron calculator
        self.get_non_local_energy_and_potential1D(ae.rgd, ae.u_j, ae.f_j, ae.e_j, ae.l_j, v_xc, 200, 
                                                  njcore = ae.njcore)

        extra_xc_data['core_response'] = v_xc.copy()

        w_j = self.get_response_weights1D(ae.u_j[ae.njcore:], ae.f_j[ae.njcore:], ae.e_j[ae.njcore:])
        extra_xc_data['response_weights'] = w_j

        if self.relaxed_core_response:

            for nc in range(0, ae.njcore):
                # Add the response multiplied with density to potential
                orbital_density = construct_density1D(ae.rgd, ae.u_j[nc], ae.f_j[nc])
                extra_xc_data['core_orbital_density_'+str(nc)] = orbital_density
                extra_xc_data['core_eigenvalue_'+str(nc)] = [ ae.e_j[nc] ]

            extra_xc_data['njcore'] = [ ae.njcore ]

    def get_slater_part_paw_correction(self, rgd, n_g, a2_g, v_g, pseudo = True, ndenom_g=None):

        if ndenom_g == None:
            ndenom_g = n_g

        # TODO: This method needs more arguments to support arbitary slater part
        self.prepare_gga_xc_1D(rgd)
        N = len(n_g)
        # TODO: Allocate these only once
        vtemp_g = num.zeros(N, float)
        etemp_g = num.zeros(N, float)
        deda2temp_g = num.zeros(N, float)
        self.gga_xc1D.xcfunc.calculate_spinpaired(etemp_g, n_g, vtemp_g, a2_g, deda2temp_g)

        # Grr... When n_g = 0, B88 returns -0.03 for e_g!!!!!!!!!!!!!!!!!
        etemp_g[:] = num.where(abs(n_g) < SMALL_NUMBER, 0, etemp_g)

        v_g[:] = 2 * etemp_g / (ndenom_g + SMALL_NUMBER)

        return num.sum(etemp_g * rgd.dv_g)

    def calculate_non_local_paw_correction(self, D_sp, H_sp, sphere_nt, sphere_n, nucleus, extra_xc_data, a):
        N = sphere_n.get_slice_length()

        n_g = num.zeros(N, float) # Density
        v_g = num.zeros(N, float) # Potential
        a2_g = num.zeros(N, float) # Density gradient |\/n|^2
        resp_g = num.zeros(N, float) # Numerator of response pontial
        core_resp_g = num.zeros(N, float) # Numerator of core response potential
        deg = len(D_sp)

        for s, (D_p, H_p, Dresp_p) in enumerate(zip(D_sp, H_sp, nucleus.Dresp_sp)):
            H_p[:] = 0.0
            # If relaxed core response, calculate the core response explicitly
            if self.relaxed_core_response:
                njcore = extra_xc_data['njcore']
                for nc in range(0, njcore):
                    psi2_g = extra_xc_data['core_orbital_density_'+str(nc)]
                    epsilon = extra_xc_data['core_eigenvalue_'+str(nc)]
                    core_resp_g[:] += psi2_g * self.gllb_weight(epsilon, self.reference_levels[s])
            else:
                # Take the core response directly from setup
                core_resp_g[:] = extra_xc_data['core_response']

            n_iter = sphere_n.get_iterator(D_p)
            nt_iter = sphere_nt.get_iterator(D_p)
            resp_iter = sphere_n.get_iterator(Dresp_p, core=False, gradient=False)
            respt_iter = sphere_nt.get_iterator(Dresp_p, core=False, gradient=False)
            Exc = 0
            while n_iter.has_next():
                # Calculate true density, density gradient and numerator of response
                v_g[:] = 0.0
                n_iter.get_density(n_g)
                n_iter.get_gradient(a2_g)
                resp_iter.get_density(resp_g)

                # Calculate the slater potential
                Exc += n_iter.get_weight() * self.get_slater_part_paw_correction( \
                              n_iter.get_rgd(), n_g, a2_g, v_g, pseudo=False)

                # Calculate the response potential
                v_g[:] += (resp_g + core_resp_g) / (n_g + SMALL_NUMBER)

                # Integrate v_g over wave functions to get H_p
                n_iter.integrate(1.0, v_g, H_p)

                # Calculate pseudo density, density gradient and numerator of response
                nt_iter.get_density(n_g)
                nt_iter.get_gradient(a2_g)
                respt_iter.get_density(resp_g)
                v_g[:] = 0
                # Calculate the pseudo slater potential
                Exc -= n_iter.get_weight() * self.get_slater_part_paw_correction( \
                n_iter.get_rgd(), n_g, a2_g, v_g, pseudo=False)

                # Calculate the pseudo response potential
                v_g[:] += resp_g / (n_g + SMALL_NUMBER)

                # Integrate v_g over pseudo wave functions to get H_p. Substract.
                nt_iter.integrate(-1.0, v_g, H_p)

                # Increment the iterators
                n_iter.next()
                nt_iter.next()
                resp_iter.next()
                respt_iter.next()
        return Exc


    def get_weights_kpoint(self, kpt):
        w_n = num.zeros(len(kpt.eps_n), float)
        # Return a weight for each of eigenvalues of k-point
        for i, e in enumerate(kpt.eps_n):
            w_n[i] = self.gllb_weight(e, self.reference_level_s[kpt.s])
        return w_n

