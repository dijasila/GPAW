#from gpaw.xc.functional import XCFunctional
from gpaw.xc.gga import PurePythonGGAKernel, GGA, gga_vars
from gpaw.xc.gga import add_gradient_correction, radial_gga_vars
from gpaw.xc.gga import add_radial_gradient_correction
from gpaw.xc.gga import GGARadialCalculator, GGARadialExpansion
import numpy as np
#from ase.neighborlist import NeighborList
#from ase.units import Bohr
#import sys
from gpaw.lfc import LFC
from gpaw.spline import Spline
#from ase.parallel import parprint
from math import sqrt, pi
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.xc.pawcorrection import rnablaY_nLv
from gpaw.xc.gga import calculate_sigma

# TODO: User GGARadialCalcualtor from gga.py
class QNARadialCalculator: #(GGARadialCalculator):
    def __init__(self, kernel):
        self.kernel = kernel
        #GGARadialCalculator.__init__(self, kernel)

    def __call__(self, rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n, QNA):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg,
         b_vsg) = radial_gga_vars(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv)

        QNA.qna_kernel_calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)

        vv_sg = add_radial_gradient_correction(rgd, sigma_xg,
                                               dedsigma_xg, a_sg)
        return e_g, dedn_sg + vv_sg, b_vsg, dedsigma_xg


class QNA(GGA):
    def __init__(self, atoms, parameters, qna_setup_name='PBE', alpha=2.0, force_atoms=None):
        kernel = PurePythonGGAKernel('QNA', kappa=0.804, mu=np.nan, beta=np.nan)
        GGA.__init__(self, kernel)
        self.atoms = atoms
        self.parameters = parameters
        self.qna_setup_name = qna_setup_name
        self.alpha = alpha
        self.force_atoms = force_atoms
        self.orbital_dependent = False

    def todict(self):
        dct = dict(type='qna-gga',
                   name='QNA',
                   setup_name=self.qna_setup_name,
                   parameters=self.parameters,
                   alpha=self.alpha,
                   orbital_dependent=False)
        return dct

    def set_grid_descriptor(self, gd):
        GGA.set_grid_descriptor(self, gd)
        self.dedmu_g = gd.zeros()
        self.dedbeta_g = gd.zeros()
        # Create gaussian LFC
        l_lim = 1.0e-30
        rcut = 12
        points = 200
        r_i = np.linspace(0, rcut, points + 1)
        rcgauss = 1.2
        g_g = 2 / rcgauss**3 / np.pi * np.exp(-((r_i / rcgauss)**2)**self.alpha)
        # Values too close to zero can cause numerical problems especially with
        # forces (some parts of the mu and beta field can become negative),
        # so we clip them:
        g_g[ np.where( g_g < l_lim ) ] = l_lim
        # non-numpy version:
        #for i in range(len(g_g)):
        #    if g_g[i] < l_lim:
        #        g_g[i] = l_lim
        spline = Spline(l=0, rmax=rcut, f_g=g_g)
        spline_j = [[ spline ]] * len(self.atoms)
        self.Pa = LFC(gd, spline_j)

    def set_positions(self, spos_ac, atom_partition=None):
        self.Pa.set_positions(spos_ac)

    def calculate_spatial_parameters(self, atoms):
        mu_g = self.gd.zeros()
        beta_g = self.gd.zeros()
        denominator = self.gd.zeros()
        mu_a = {}
        beta_a = {}
        eye_a = {}
        for atom in atoms:
            mu, beta = self.parameters[atom.symbol]
            mu_a[atom.index] = [ mu ]
            beta_a[atom.index] = [ beta ]
            eye_a[atom.index] = 1.0
        self.Pa.add(mu_g, mu_a)
        self.Pa.add(beta_g, beta_a)
        self.Pa.add(denominator, eye_a)
        mu_g /= denominator
        beta_g /= denominator
        return mu_g, beta_g

    def qna_kernel_calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        if self.force_atoms is not None:
            atoms = self.force_atoms
            self.Pa.set_positions(atoms.get_scaled_positions() % 1.0)
        else:
            atoms = self.atoms

        if len(n_sg.shape) > 2:
            mu_g, beta_g = self.calculate_spatial_parameters(atoms)
            dedmu_g = self.dedmu_g
            dedbeta_g = self.dedbeta_g
        else:
            # For atoms, use always atomwise mu and beta parameters
            mu, beta = self.parameters[self.atoms[self.current_atom].symbol]
            mu_g = np.zeros_like(n_sg[0])
            beta_g = np.zeros_like(n_sg[0])
            mu_g[:] = mu
            beta_g[:] = beta
            dedmu_g = None
            dedbeta_g = None
 
        #Enable to use PBE always
        #mu_g[:] = 0.2195149727645171
        #beta_g[:] = 0.06672455060314922

        # Write mu and beta fields
        if 0:
            from ase.io import write
            write('mu_g.cube', atoms, data=mu_g)
            write('beta_g.cube', atoms, data=beta_g)
            asd

        return self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=mu_g,
                                     beta_g=beta_g, dedmu_g=dedmu_g, dedbeta_g=dedbeta_g)


    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, self.grad_v, n_sg)

        self.qna_kernel_calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

        add_gradient_correction(self.grad_v, gradn_svg, sigma_xg,
                                dedsigma_xg, v_sg)


    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        self.current_atom = a
        rcalc = QNARadialCalculator(self.kernel)
        expansion = GGARadialExpansion(rcalc, self)
        xcc = setup.xc_correction
        if xcc is None:
            return 0.0

        rgd = xcc.rgd
        nspins = len(D_sp)

        if addcoredensity:
            nc0_sg = rgd.empty(nspins)
            nct0_sg = rgd.empty(nspins)
            nc0_sg[:] = sqrt(4 * pi) / nspins * xcc.nc_g
            nct0_sg[:] = sqrt(4 * pi) / nspins * xcc.nct_g
            if xcc.nc_corehole_g is not None and nspins == 2:
                nc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
                nc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
        else:
            nc0_sg = 0
            nct0_sg = 0

        D_sLq = np.inner(D_sp, xcc.B_pqL.T)

        e, dEdD_sqL = expansion(rgd, D_sLq, xcc.n_qg, nc0_sg)
        et, dEtdD_sqL = expansion(rgd, D_sLq, xcc.nt_qg, nct0_sg)

        if dEdD_sp is not None:
            dEdD_sp += np.inner((dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
                                xcc.B_pqL.reshape((len(xcc.B_pqL), -1)))

        if addcoredensity:
            return e - et - xcc.e_xc0
        else:
            return e - et


    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        dndr_sg = np.empty_like(n_sg)
        for n_g, dndr_g in zip(n_sg, dndr_sg):
            rgd.derivative(n_g, dndr_g)
        if e_g is None:
            e_g = rgd.empty()

        rcalc = QNARadialCalculator(self.kernel)

        e_g[:], dedn_sg = rcalc(rgd, n_sg[:, np.newaxis],
                                [1.0],
                                dndr_sg[:, np.newaxis],
                                np.zeros((1, 3)), n=None, QNA=self)[:2]
        v_sg[:] = dedn_sg
        return rgd.integrate(e_g)


    def stress_tensor_contribution(self, n_sg):
        sigma_xg, gradn_svg = calculate_sigma(self.gd, self.grad_v, n_sg)
        nspins = len(n_sg)
        dedsigma_xg = self.gd.empty(nspins * 2 - 1)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()

        self.qna_kernel_calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        for v_g, n_g in zip(v_sg, n_sg):
            P -= integrate(v_g, n_g)
        for sigma_g, dedsigma_g in zip(sigma_xg, dedsigma_xg):
            P -= 2 * integrate(sigma_g, dedsigma_g)
        stress_vv = P * np.eye(3)
        for v1 in range(3):
            for v2 in range(3):
                stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                               gradn_svg[0, v2],
                                               dedsigma_xg[0]) * 2
                if nspins == 2:
                    stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                                   gradn_svg[1, v2],
                                                   dedsigma_xg[1]) * 2
                    stress_vv[v1, v2] -= integrate(gradn_svg[1, v1] *
                                                   gradn_svg[1, v2],
                                                   dedsigma_xg[2]) * 2
        return stress_vv

    def get_setup_name(self):
        return self.qna_setup_name
    
    def get_description(self):
        return "QNA Parameters: "+str(self.parameters)

    def add_forces(self, F_av):
        mu_g = self.gd.zeros()
        beta_g = self.gd.zeros()
        denominator = self.gd.zeros()
        mu_a = {}
        beta_a = {}
        eye_a = {}
        for atom in self.atoms:
            mu, beta = self.parameters[atom.symbol]
            mu_a[atom.index] = [ mu ]
            beta_a[atom.index] = [ beta ]
            eye_a[atom.index] = 1.0
        self.Pa.add(mu_g, mu_a)
        self.Pa.add(beta_g, beta_a)
        self.Pa.add(denominator, eye_a)
        mu_g /= denominator
        beta_g /= denominator

        # mu
        part1 = -self.dedmu_g / denominator
        part2 = -part1 * mu_g
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part1, c_axiv)
        old = F_av.copy()
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:] * mu_a[atom.index][0]
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part2, c_axiv)
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:]

        # beta
        part1 = -self.dedbeta_g / denominator
        part2 = -part1 * beta_g
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part1, c_axiv)
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:] * beta_a[atom.index][0]
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part2, c_axiv)
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:]

