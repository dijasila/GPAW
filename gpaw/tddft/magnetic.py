import numpy as np

from gpaw.io import Reader, Writer

from gpaw import debug
from gpaw.utilities import is_contiguous
from gpaw.analyse.observers import Observer
from gpaw.transformers import Transformer
from gpaw.tddft.units import attosec_to_autime, eV_to_aufrequency
from gpaw.fd_operators import Gradient


class MagneticMomentObserver(Observer):
    def __init__(self, filename):
        Observer.__init__(self)
        self.filename = filename
        self.f = open(filename,'w')

    def initialize(self, paw):
        assert hasattr(paw, 'time') and hasattr(paw, 'niter'), 'Use TDDFT!'
        self.time = paw.time
        self.niter = paw.niter
        self.wfs = paw.wfs

        # Attach to PAW-type object
        paw.attach(self, self.interval)

    def update(self):


        grad = []
        dtype = np.float
        for c in range(3):
            grad.append(Gradient(self.wfs.gd, c, dtype=dtype, n=2))
        
        grad_psit_G = [self.wfs.gd.empty(), self.wfs.gd.empty(),
                        self.wfs.gd.empty()]

        
  
        for kpt in calc.wfs.kpt_u:

            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):

            # Gradients
                for c in range(3):
                     grad[c].apply(psit_G, grad_psit_G[c],kpt.phase_cd)

            # <psi1|r x grad|psi2>
            #    i  j  k
            #    x  y  z   = (y pz - z py)i + (z px - x pz)j + (x py - y px)
            #    px py pz

            rxnabla_g = np.zeros(3)
            rxnabla_g[0] = self.wfs.gd.integrate(psit_G *
                                                      (r_cG[1] * grad_psit_G[2] -
                                                       r_cG[2] * grad_psit_G[1]))
            rxnabla_g[1] = self.wfs.gd.integrate(psit_G *
                                                      (r_cG[2] * grad_psit_G[0] -
                                                       r_cG[0] * grad_psit_G[2]))
            rxnabla_g[2] = self.wfs.gd.integrate(psit_G *
                                                      (r_cG[0] * grad_psit_G[1] -
                                                       r_cG[1] * grad_psit_G[0]))

             
            print >> self.f, "%.16f %16.f %16.f" % rxnabla
            self.f.flush()
 

            """

            # augmentation contributions to magnetic moment
            # <psi1| r x nabla |psi2> = <psi1| (r-Ra+Ra) x nabla |psi2>
            #                         = <psi1| (r-Ra) x nabla |psi2> + Ra x <psi1| nabla |psi2>
            rxnabla_a = np.zeros(3)
            # <psi1| (r-Ra) x nabla |psi2>
            for a, P_ni in self.calc.wfs.kpt_u[self.kpt_ind].P_ani.items():
                Pi_i = P_ni[kss_ip.occ_ind]
                Pp_i = P_ni[kss_ip.unocc_ind]
                rxnabla_iiv = self.calc.wfs.setups[a].rxnabla_iiv
                for c in range(3):
                    for i1, Pi in enumerate(Pi_i):
                        for i2, Pp in enumerate(Pp_i):
                            rxnabla_a[c] += Pi * Pp * rxnabla_iiv[i1, i2, c]

            self.calc.wfs.gd.comm.sum(rxnabla_a) # sum up from different procs


            # Ra x <psi1| nabla |psi2>
            Rxnabla_a = np.zeros(3)
            for a, P_ni in self.calc.wfs.kpt_u[self.kpt_ind].P_ani.items():
                Pi_i = P_ni[kss_ip.occ_ind]
                Pp_i = P_ni[kss_ip.unocc_ind]
                nabla_iiv = self.calc.wfs.setups[a].nabla_iiv
                Ra = (self.calc.atoms[a].position / ase.units.Bohr) - R0
                for i1, Pi in enumerate(Pi_i):
                    for i2, Pp in enumerate(Pp_i):
                        # (y pz - z py)i + (z px - x pz)j + (x py - y px)k
                        Rxnabla_a[0] += Pi * Pp * ( Ra[1] * nabla_iiv[i1, i2, 2] - Ra[2] * nabla_iiv[i1, i2, 1] )
                        Rxnabla_a[1] += Pi * Pp * ( Ra[2] * nabla_iiv[i1, i2, 0] - Ra[0] * nabla_iiv[i1, i2, 2] )
                        Rxnabla_a[2] += Pi * Pp * ( Ra[0] * nabla_iiv[i1, i2, 1] - Ra[1] * nabla_iiv[i1, i2, 0] )


            self.calc.wfs.gd.comm.sum(Rxnabla_a) # sum up from different procs

            #print (kss_ip.occ_ind, kss_ip.unocc_ind), kss_ip.dip_mom_r, rxnabla_g, rxnabla_a, Rxnabla_a

            # m_ip = -1/2c <i|r x p|p> = i/2c <i|r x nabla|p>
            # just imaginary part!!!
            kss_ip.magn_mom = ase.units.alpha / 2. * (rxnabla_g + rxnabla_a + Rxnabla_a)"""


