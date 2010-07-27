from gpaw.xc.lda import LDA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient


class GGA(LDA):
    def set_grid_descriptor(self, gd):
        LDA.set_grid_descriptor(self, gd)
        self.grad_v = [Gradient(gd, v, allocate=not False).apply
                       for v in range(3)]

    def _calculate(self, e_g, n_sg, v_sg):
        gradn_vg = self.gd.empty(3)
        sigma_xg = self.gd.zeros(1)
        dedsigma_xg = self.gd.empty(1)
        for v in range(3):
            self.grad_v[v](n_sg[0], gradn_vg[v])
            axpy(1.0, gradn_vg[v]**2, sigma_xg[0])
        self.xckernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        tmp_g = gradn_vg[0]
        for v in range(3):
            self.grad_v[v](dedsigma_xg[0] * gradn_vg[v], tmp_g)
            axpy(-2.0, tmp_g, v_sg[0])       

    def calculate_radial(self, rgd, n_sg, v_sg=None, e_g=None):
        if e_g is None:
            e_g = rgd.empty()
        if v_sg is None:
            v_sg = np.zeros_like(n_sg)
        self.xckernel.calculate(e_g, n_sg, v_sg)
        return rgd.integrate(e_g)
