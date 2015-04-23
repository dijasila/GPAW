from gpaw.xc.mgga import MGGA
from gpaw.xc.libxc import LibXC
from gpaw.fd_operators import Laplace


class TB09Kernel:
    name = 'TB09'
    type = 'MGGA'
    
    def __init__(self):
        self.tb09 = LibXC('MGGA_X_TB09').xc.tb09
        self.ldac = LibXC('LDA_C_PW')
        
    def calculate(self, e_g, n_sg, dedn_sg, sigma_xg,
                  dedsigma_xg, tau_sg, dedtau_sg):
        assert len(n_sg) == 1
        if n_sg.ndim == 4:
            lapl_g = self.gd.empty()
            self.lapl.apply(n_sg[0], lapl_g)
            self.n_Lg = None
        else:
            lapl_g = self.rgd.zeros()
            l = 0
            m = 0
            for Y, n_g in zip(self.Y_L, self.n_Lg):
                nrm2_g = n_g.copy()
                nrm2_g[1:] /= self.rgd.r_g[1:]**2
                lapl_g += Y * (self.rgd.laplace(n_g) - l * (l + 1) * nrm2_g)
                m += 1
                if m == 2 * l + 1:
                    l += 1
                    m = 0
            lapl_g[0] = 0.0
        dedn_sg[:] = 0.0
        n_sg[n_sg < 1e-6] = 1e-6
        sigma_xg[sigma_xg < 1e-10] = 1e-10
        tau_sg[tau_sg < 1e-10] = 1e-10
        self.tb09(1.2, n_sg.ravel(), sigma_xg, lapl_g, tau_sg, dedn_sg,
                  dedsigma_xg)
        self.ldac.calculate(e_g, n_sg, dedn_sg)
        e_g[:] = 0.0
        dedsigma_xg[:] = 0.0
        dedtau_sg[:] = 0.0

        
class TB09(MGGA):
    def __init__(self):
        MGGA.__init__(self, TB09Kernel())

    def get_setup_name(self):
        return 'LDA'

    def initialize(self, dens, ham, wfs, occ):
        MGGA.initialize(self, dens, ham, wfs, occ)
        self.kernel.rgd = wfs.setups[0].xc_correction.rgd
        self.kernel.gd = dens.finegd
        self.kernel.lapl = Laplace(dens.finegd)
        
    def calculate_radial(self, rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv):
        self.kernel.n_Lg = n_sLg[0]
        self.kernel.Y_L = Y_L
        return MGGA.calculate_radial(self, rgd, n_sLg, Y_L, dndr_sLg,
                                     rnablaY_Lv)
        
    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG, dH_asp):
        pass
