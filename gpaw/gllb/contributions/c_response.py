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

        self.vt_sg = self.finegd.empty(self.nlfunc.nspins)
        self.vt_sG = self.gd.empty(self.nlfunc.nspins)
        self.nt_sG = self.gd.empty(self.nlfunc.nspins)

        self.Dresp_asp = None
        
    def calculate_spinpaired(self, e_g, n_g, v_g):
        w_kn = self.coefficients.get_coefficients_by_kpt(self.kpt_u)
        f_kn = [ kpt.f_n for kpt in self.kpt_u ]
        if w_kn is None:
            return 0.0

        self.vt_sG[:] = 0.0
        self.nt_sG[:] = 0.0
        for kpt, w_n in zip(self.kpt_u, w_kn):
            self.wfs.add_to_density_from_k_point_with_occupation(self.vt_sG, kpt, w_n)
            self.wfs.add_to_density_from_k_point(self.nt_sG, kpt)

        if self.wfs.symmetry:
            for nt_G, vt_G in zip(self.nt_sG, self.vt_sG):
                self.symmetry.symmetrize(nt_G, self.gd)
                self.symmetry.symmetrize(vt_G, self.gd)

        if self.Dresp_asp is None:
            # Initiailze 'response-density' and density-matrices
            self.Dresp_asp = {}
            self.D_asp = {}
            for a in self.density.nct.my_atom_indices:
                ni = self.setups[a].ni
                self.Dresp_asp[a] = npy.zeros((self.nlfunc.nspins, ni * (ni + 1) // 2))
                self.D_asp[a] = npy.zeros((self.nlfunc.nspins, ni * (ni + 1) // 2))
            

        self.wfs.calculate_atomic_density_matrices_with_occupation(
            self.Dresp_asp, w_kn)
        self.wfs.calculate_atomic_density_matrices_with_occupation(
            self.D_asp, f_kn)

        self.vt_sG /= self.nt_sG[0] +1e-10
        self.density.interpolater.apply(self.vt_sG[0], self.vt_sg[0])
        v_g[:] += self.weight * self.vt_sg[0]
        return 0.0

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g, 
                                a2_g=None, aa2_g=None, ab2_g=None, deda2_g=None,
                                dedaa2_g=None, dedab2_g=None):
        raise NotImplementedError

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        if self.Dresp_asp is None:
            return 0.0
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
        w_j = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.s_j) # s_j, not u_j!
        vt_g += self.weight * npy.dot(w_j, u2_j) / (npy.dot(self.ae.f_j, u2_j) +1e-10)
        return 0.0 # Response part does not contribute to energy
        
    def add_extra_setup_data(self, dict):
        ae = self.ae
        njcore = ae.njcore
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
