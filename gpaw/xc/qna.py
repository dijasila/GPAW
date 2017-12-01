#from gpaw.xc.functional import XCFunctional
from gpaw.xc.gga import PurePythonGGAKernel, GGA
import numpy as np
#from ase.neighborlist import NeighborList
#from ase.units import Bohr
#import sys
from gpaw.lfc import LFC
from gpaw.spline import Spline
#from ase.parallel import parprint

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
        if 0:
            from ase.io import write
            #write('mu_raw.cube', atoms, data=mu_g)
            write('denominator.cube', atoms, data=denominator)
            asd
        return mu_g, beta_g

    def calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
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

        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=mu_g, beta_g=beta_g, dedmu_g=dedmu_g, dedbeta_g=dedbeta_g)

        # Integrate dedmu_g
        #print(self.gd.integrate(dedmu_g))
        #asd

        # Write mu and beta fields
        if 0:
            from ase.io import write
            write('mu_g.cube', atoms, data=mu_g)
            write('beta_g.cube', atoms, data=beta_g)
            asd

        # Write exc
        if 0:
            from ase.io import write
            write('exc.cube', self.atoms, data=e_g)
            asd

        # Write reduced gradient s
        if 0:
            from ase.io import write
            C0I = 0.238732414637843
            C2 = 0.26053088059892404
            rs = (C0I / n_sg[0])**(1 / 3.)
            red_grad = np.sqrt(sigma_xg[0])*(C2*rs/n_sg[0])
            write('s.cube', self.atoms, data=red_grad)
            asd
        
        # Write XC potential
        #from ase.io import write
        #write('XC_potential.cube', self.atoms, data=v_sg[0])
        #asd

        # Write dFxc_ds
        if 0:
            dx = 1.0e-3 
            e2_g = np.zeros_like(e_g)
            e3_g = np.zeros_like(e_g)
            exLDA = np.zeros_like(e_g)
            muLDA = np.zeros_like(mu_g)
            betaLDA = np.zeros_like(beta_g)
            muLDA[:] = 0.000001
            betaLDA[:] = 0.000001
            self.kernel.calculate(e2_g, n_sg, v_sg, sigma_xg + dx, dedsigma_xg, mu_g=mu_g, beta_g=beta_g)
            self.kernel.calculate(e3_g, n_sg, v_sg, sigma_xg - dx, dedsigma_xg, mu_g=mu_g, beta_g=beta_g)
            self.kernel.calculate(exLDA, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=muLDA, beta_g=betaLDA)
            from ase.io import write
            write('dFxc_dsigma.cube', self.atoms, data=(e2_g-e3_g)/(2*dx)/exLDA)
            asd
            
        # Write the ratio of two XCs
        if 0:
            mu2_g = np.zeros_like(mu_g)
            mu3_g = np.zeros_like(mu_g)
            mu2_g[:] = 0.219515
            mu3_g[:] = 0.000001
            beta2_g = np.zeros_like(beta_g)
            beta3_g = np.zeros_like(beta_g)
            beta2_g[:] = 0.066725
            beta3_g[:] = 0.000001
            e2_g = np.zeros_like(e_g)
            e3_g = np.zeros_like(e_g)
            self.kernel.calculate(e2_g, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=mu2_g, beta_g=beta2_g)
            self.kernel.calculate(e3_g, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=mu3_g, beta_g=beta3_g)
            from ase.io import write
            write('exc_ratio.cube', self.atoms, data=e2_g/e3_g)
            asd
        
        # Calculate XC potential numerically
        if 0:
            e2_g = np.zeros_like(e_g)
            e3_g = np.zeros_like(e_g)
            self.kernel.calculate(e2_g, n_sg + 1.0e-6, v_sg, sigma_xg, dedsigma_xg, mu_g=mu_g, beta_g=beta_g)
            self.kernel.calculate(e3_g, n_sg - 1.0e-6, v_sg, sigma_xg, dedsigma_xg, mu_g=mu_g, beta_g=beta_g)
            from ase.io import write
            write('vxc_numerical.cube', self.atoms, data=(e2_g-e3_g)/2e-6)
            asd

        if 0:
            e2_g = np.zeros_like(e_g)
            e3_g = np.zeros_like(e_g)
            self.kernel.calculate(e2_g, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=mu_g+1e-6, beta_g=beta_g)
            self.kernel.calculate(e3_g, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=mu_g-1e-6, beta_g=beta_g)
            from ase.io import write
            #write('dedmu_analytic.cube', self.atoms, data=dedmu_g)
            #write('dedmu_numerical.cube', self.atoms, data=(e2_g-e3_g)/2e-6)
            #write('dedmu_div.cube', self.atoms, data=(e2_g-e3_g)/2e-6 / dedmu_g)
            #print(dedmu_g)
            asd

        if 0:
            e2_g = np.zeros_like(e_g)
            e3_g = np.zeros_like(e_g)
            self.kernel.calculate(e2_g, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=mu_g, beta_g=beta_g+1e-4)
            self.kernel.calculate(e3_g, n_sg, v_sg, sigma_xg, dedsigma_xg, mu_g=mu_g, beta_g=beta_g-1e-4)
            from ase.io import write
            write('dedbeta_analytic.cube', self.atoms, data=dedbeta_g)
            write('dedbeta_numerical.cube', self.atoms, data=(e2_g-e3_g)/2e-4)
            write('dedbeta_div.cube', self.atoms, data=(e2_g-e3_g)/2e-4 / dedbeta_g)
            asd
        

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        self.current_atom = a
        return GGA.calculate_paw_correction(self, setup, D_sp, dEdD_sp=dEdD_sp,
                                             addcoredensity=addcoredensity, a=a)

    def calculate_gga_qna_radial(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        return self.calculate_gga(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

    calculate_gga_radial = calculate_gga_qna_radial

    def get_setup_name(self):
        return self.qna_setup_name
    
    def get_description(self):
        return "QNA Parameters"+str(self.parameters)

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
        #print "mu Force before",F_av[0][0]
        old = F_av.copy()
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:] * mu_a[atom.index][0]
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part2, c_axiv)
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:]
        #print "mu Force after",F_av[0][0]

        # beta
        part1 = -self.dedbeta_g / denominator
        part2 = -part1 * beta_g
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part1, c_axiv)
        #print "beta Force before",F_av[0][0]
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:] * beta_a[atom.index][0]
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part2, c_axiv)
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:]
        #print "beta Force after",F_av[0][0]

        #from ase.units import Hartree, Bohr
        #print "Analytic Correction", (F_av[0][0] - old[0][0])* (Hartree / Bohr)
