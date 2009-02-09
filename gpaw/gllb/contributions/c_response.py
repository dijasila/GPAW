from gpaw.gllb.contributions.contribution import Contribution
from gpaw.gllb.contributions.contribution import Contribution
from gpaw.xc_functional import XCRadialGrid, XCFunctional, XC3DGrid
from gpaw.xc_correction import A_Liy
from gpaw.gllb import safe_sqr
from math import sqrt, pi

import numpy as npy

class C_Response(Contribution):
    def __init__(self, nlfunc, weight, coefficients):
        Contribution.__init__(self, nlfunc, weight)
        self.coefficients = coefficients
        
    # Initialize Response functional
    def initialize_1d(self):
        self.ae = self.nlfunc.ae

    # Calcualte the GLLB potential and energy 1d
    def add_xc_potential_and_energy_1d(self, v_g):
        w_i = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.u_j)
        v_g += self.weight * npy.dot(w_i, u2_j) / (npy.dot(self.ae.f_j, u2_j) +1e-10)
        return 0.0 # Response part does not contribute to energy

    def initialize(self):
        self.gd = self.nlfunc.gd
        self.finegd = self.nlfunc.finegd
        self.wfs = self.nlfunc.wfs
        self.kpt_u = self.wfs.kpt_u
        self.setups = self.wfs.setups
        self.density = self.nlfunc.density
        self.symmetry = self.wfs.symmetry
        self.nspins = self.nlfunc.nspins

        self.vt_sg = self.finegd.empty(self.nlfunc.nspins)
        self.vt_sG = self.gd.empty(self.nlfunc.nspins)
        print "Writing over vt_sG!"
        self.nt_sG = self.gd.empty(self.nlfunc.nspins)

        self.Dresp_asp = None
        
    def calculate_spinpaired(self, e_g, n_g, v_g):
        w_kn = self.coefficients.get_coefficients_by_kpt(self.kpt_u)
        f_kn = [ kpt.f_n for kpt in self.kpt_u ]

        if w_kn is not None:
            self.vt_sG[:] = 0.0
            self.nt_sG[:] = 0.0
            for kpt, w_n in zip(self.kpt_u, w_kn):
                self.wfs.add_to_density_from_k_point_with_occupation(self.vt_sG, kpt, w_n)
                self.wfs.add_to_density_from_k_point(self.nt_sG, kpt)

            if self.wfs.symmetry:
                for nt_G, vt_G in zip(self.nt_sG, self.vt_sG):
                    self.symmetry.symmetrize(nt_G, self.gd)
                    self.symmetry.symmetrize(vt_G, self.gd)


            self.wfs.calculate_atomic_density_matrices_with_occupation(
                self.Dresp_asp, w_kn)
            self.wfs.calculate_atomic_density_matrices_with_occupation(
                self.D_asp, f_kn)

            self.vt_sG /= self.nt_sG +1e-10
            print "Updating vt_sG"
        else:
            print "Reusing potential"
            
        self.density.interpolater.apply(self.vt_sG[0], self.vt_sg[0])
        v_g[:] += self.weight * self.vt_sg[0]
        return 0.0

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g, 
                                a2_g=None, aa2_g=None, ab2_g=None, deda2_g=None,
                                dedaa2_g=None, dedab2_g=None):
        raise NotImplementedError

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        # Get the XC-correction instance
        c = self.nlfunc.setups[a].xc_correction
        ncresp_g = self.nlfunc.setups[a].extra_xc_data['core_response']
        
        D_p = self.D_asp.get(a)[0]
        Dresp_p = self.Dresp_asp.get(a)[0]
        dEdD_p = H_sp[0][:]
        
        D_Lq = npy.dot(c.B_Lqp, D_p)
        n_Lg = npy.dot(D_Lq, c.n_qg) # Construct density
        n_Lg[0] += c.nc_g * sqrt(4 * pi)
        nt_Lg = npy.dot(D_Lq, c.nt_qg) # Construct smooth density (without smooth core)

        Dresp_Lq = npy.dot(c.B_Lqp, Dresp_p)
        nresp_Lg = npy.dot(Dresp_Lq, c.n_qg) # Construct 'response density'
        nrespt_Lg = npy.dot(Dresp_Lq, c.nt_qg) # Construct smooth 'response density' (w/o smooth core)

        for w, Y_L in zip(c.weights, c.Y_yL):
            nt_g = npy.dot(Y_L, nt_Lg)
            nrespt_g = npy.dot(Y_L, nrespt_Lg)
            x_g = nrespt_g / (nt_g + 1e-10)
            dEdD_p -= self.weight * w * npy.dot(npy.dot(c.B_pqL, Y_L),
                                                npy.dot(c.nt_qg, x_g * c.rgd.dv_g))

            n_g = npy.dot(Y_L, n_Lg)
            nresp_g = npy.dot(Y_L, nresp_Lg)
            x_g = (nresp_g+ncresp_g) / (n_g + 1e-10)
            
            dEdD_p += self.weight * w * npy.dot(npy.dot(c.B_pqL, Y_L),
                                                npy.dot(c.n_qg, x_g * c.rgd.dv_g))
            
        return 0.0
    
    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        w_ln = self.coefficients.get_coefficients_1d(smooth=True)
        v_g = npy.zeros(self.ae.N)
        n_g = npy.zeros(self.ae.N)
        for w_n, f_n, u_n in zip(w_ln, self.ae.f_ln, self.ae.s_ln): # For each angular momentum
            u2_n = safe_sqr(u_n)
            v_g += npy.dot(w_n, u2_n)
            n_g += npy.dot(f_n, u2_n)
                           
        vt_g += self.weight * v_g / (n_g + 1e-10)
        return 0.0 # Response part does not contribute to energy

    def initialize_from_atomic_orbitals(self, basis_functions):
        # Initiailze 'response-density' and density-matrices
        self.Dresp_asp = {}
        self.D_asp = {}
        
        for a in self.density.nct.my_atom_indices:
            ni = self.setups[a].ni
            self.Dresp_asp[a] = npy.zeros((self.nlfunc.nspins, ni * (ni + 1) // 2))
            self.D_asp[a] = npy.zeros((self.nlfunc.nspins, ni * (ni + 1) // 2))
            
        f_sM = npy.empty((self.nspins, basis_functions.Mmax))
        self.D_asp = {}
        f_asi = {}
        w_asi = {}

        assert self.nspins == 1  # Note: All initializations with magmom=0, hund=False and charge=0
        for a in basis_functions.atom_indices:
            w_j = self.setups[a].extra_xc_data['w_j']
            # Basis function coefficients based of response weights
            w_si = self.setups[a].calculate_initial_occupation_numbers(
                    0, False, charge=0, f_j = w_j)
            # Basis function coefficients based on density
            f_si = self.setups[a].calculate_initial_occupation_numbers(
                    0, False, charge=0)            
            if a in basis_functions.my_atom_indices:
                self.Dresp_asp[a] = self.setups[a].initialize_density_matrix(w_si)
                self.D_asp[a] = self.setups[a].initialize_density_matrix(f_si)
                
            f_asi[a] = f_si
            w_asi[a] = w_si

        self.nt_sG.fill(0.0)
        basis_functions.add_to_density(self.nt_sG, f_asi)
        self.vt_sG.fill(0.0)
        basis_functions.add_to_density(self.vt_sG, w_asi)
        # Update vt_sG to correspond atomic response potential. This will be
        # used until occupations and eigenvalues are available.
        self.vt_sG /= self.nt_sG + 1e-10

    def add_extra_setup_data(self, dict):
        ae = self.ae
        njcore = ae.njcore
        w_ln = self.coefficients.get_coefficients_1d(smooth=True)
        w_j = []
        for w_n in w_ln:
            for w in w_n:
                w_j.append(w)
        dict['w_j'] = w_j

        w_j = self.coefficients.get_coefficients_1d()
        x_g = npy.dot(w_j[:njcore], safe_sqr(ae.u_j[:njcore]))
        x_g[1:] /= ae.r[1:]**2 * 4*npy.pi
        x_g[0] = x_g[1]
        dict['core_response'] = x_g        

        # For debugging purposes
        w_j = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.u_j)
        v_g = self.weight * npy.dot(w_j, u2_j) / (npy.dot(self.ae.f_j, u2_j) +1e-10)
        v_g[0] = v_g[1]
        dict['all_electron_response'] = v_g

    def write(self, w):
        wfs = self.wfs
        world = wfs.world
        domain_comm = wfs.gd.comm
        kpt_comm = wfs.kpt_comm
        band_comm = wfs.band_comm
        
        master = (world.rank == 0)

        atoms = self.nlfunc.atoms
        natoms = len(atoms)

        nadm = 0
        for setup in wfs.setups:
            ni = setup.ni
            nadm += ni * (ni + 1) / 2

        # Not yet tested for parallerization
        assert world.size == 1

        # Write the pseudodensity on the coarse grid:
        if master:
            w.add('GLLBPseudoResponsePotential',
                  ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)
        if kpt_comm.rank == 0:
            for s in range(wfs.nspins):
                vt_sG = wfs.gd.collect(self.vt_sG[s])
                if master:
                    w.fill(vt_sG)

        print "Integration over vt_sG", npy.sum(self.vt_sG.ravel())
                
        if master:
            all_D_sp = npy.empty((wfs.nspins, nadm))
            all_Dresp_sp = npy.empty((wfs.nspins, nadm))
            p1 = 0
            for a in range(natoms):
                ni = wfs.setups[a].ni
                nii = ni * (ni + 1) / 2
                if a in self.D_asp:
                    D_sp = self.D_asp[a]
                    Dresp_sp = self.Dresp_asp[a]
                else:
                    D_sp = npy.empty((wfs.nspins, nii))
                    domain_comm.receive(D_sp, wfs.rank_a[a], 27)
                    Dresp_sp = npy.empty((wfs.nspins, nii))
                    domain_comm.receive(Dresp_sp, wfs.rank_a[a], 271)
                p2 = p1 + nii
                all_D_sp[:, p1:p2] = D_sp
                all_Dresp_sp[:, p1:p2] = Dresp_sp
                p1 = p2
            assert p2 == nadm
            w.add('GLLBAtomicDensityMatrices', ('nspins', 'nadm'), all_D_sp)
            w.add('GLLBAtomicResponseMatrices', ('nspins', 'nadm'), all_Dresp_sp)
        elif kpt_comm.rank == 0 and band_comm.rank == 0:
            for a in range(natoms):
                if a in self.density.D_asp:
                    domain_comm.send(self.D_asp[a], 0, 27)
                    domain_comm.send(self.Dresp_asp[a], 0, 271)

        #print "Wrote Dresp_asp", self.Dresp_asp
        
    def read(self, r):
        wfs = self.wfs
        world = wfs.world
        domain_comm = wfs.gd.comm
        kpt_comm = wfs.kpt_comm
        band_comm = wfs.band_comm
        
        self.vt_sG = wfs.gd.empty(wfs.nspins)
        print "Reading vt_sG"
        for s in range(wfs.nspins):
            self.gd.distribute(r.get('GLLBPseudoResponsePotential', s),
                              self.vt_sG[s])
        print "Integration over vt_sG", npy.sum(self.vt_sG.ravel())
        
        # Read atomic density matrices and non-local part of hamiltonian:
        D_sp = r.get('GLLBAtomicDensityMatrices')
        Dresp_sp = r.get('GLLBAtomicResponseMatrices')
        
        self.D_asp = {}
        self.Dresp_asp = {}
        p1 = 0
        for a, setup in enumerate(wfs.setups):
            ni = setup.ni
            p2 = p1 + ni * (ni + 1) // 2
            if domain_comm.rank == 0:
                # NOTE: Distrbibutes the matrices to more processors than necessary
                self.D_asp[a] = D_sp[:, p1:p2].copy()
                self.Dresp_asp[a] = Dresp_sp[:, p1:p2].copy()
            p1 = p2

        #print "Read Dresp_asp", self.Dresp_asp

