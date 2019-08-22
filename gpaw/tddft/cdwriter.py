import numpy as np

from gpaw.mpi import world
from ase.units import Hartree, alpha, Bohr
from gpaw.utilities.tools import coordinates
from gpaw.fd_operators import Gradient
from gpaw.lcaotddft.observer import TDDFTObserver
# from gpaw.lcaotddft.dipolemomentwriter import repr


def convert_repr(r):
    # Integer
    try:
        return int(r)
    except ValueError:
        pass
    # Boolean
    b = {repr(False): False, repr(True): True}.get(r, None)
    if b is not None:
        return b
    # String
    s = r[1:-1]
    if repr(s) == r:
        return s
    raise RuntimeError('Unknown value: %s' % r)

class CDWriter(TDDFTObserver):
    version=1

    def __init__(self, paw, filename, center=False,
                 interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.master = paw.world.rank == 0
        if paw.niter == 0:
            # Initialize
            self.do_center = center
            if self.master:
                self.fd = open(filename, 'w')
        else:
            # Read and continue
            self.read_header(filename)
            if self.master:
                self.fd = open(filename, 'a')

        # Create Gradient operator
        gd = paw.wfs.gd
        grad = []
        for c in range(3):
            grad.append(Gradient(gd, c, dtype=complex, n=2))
        self.grad = grad
        self.timer=paw.timer

    def _write(self, line):
        if self.master:
            self.fd.write(line)
            self.fd.flush()

    def _write_header(self, paw):
        if paw.niter != 0:
            return
        line = '# %s[version=%s]' % (self.__class__.__name__, self.version)
        line += ('(center=%s)\n' %
                 (repr(self.do_center)))
        line += ('# %15s %22s %22s %22s\n' %
                 ('time', 'cmx', 'cmy', 'cmz'))
        self._write(line)

    def read_header(self, filename):
        with open(filename, 'r') as f:
            line = f.readline()
        m_i = re.split("[^a-zA-Z0-9_=']+", line[2:])
        assert m_i.pop(0) == self.__class__.__name__
        for m in m_i:
            if '=' not in m:
                continue
            k, v = m.split('=')
            v = convert_repr(v)
            if k == 'version':
                assert v == self.version
                continue
            # Translate key
            k = {'center': 'do_center'}[k]
            setattr(self, k, v)
   
    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def calculate_cd_moment(self, paw, center=True):
        grad = self.grad
        gd = paw.wfs.gd
               
        self.timer.start('CD')


        grad_psit_G = gd.empty(3, dtype=complex)
        psit_G=gd.empty(dtype=complex)
        rxnabla_a = np.zeros(3,dtype=complex)
        # Ra x <psi1| nabla |psi2>
        Rxnabla_a = np.zeros(3, dtype=complex)
        Rxnabla_a2 = np.zeros(3, dtype=complex)

        rxnabla_g = np.zeros(3, dtype=complex)
        R0 = 0.5 * np.diag(paw.wfs.gd.cell_cv) # + [0.0, 0.0, 2.0]

        print(R0)
        r_cG, r2_G = coordinates(paw.wfs.gd, origin=R0)

        for kpt in paw.wfs.kpt_u:
            #paw.wfs.atomic_correction.calculate_projections(paw.wfs, kpt)

            for n, (f, psit_G) in enumerate(zip(kpt.f_n, kpt.psit_nG)):
                #psit_G[:] = 0.0
                #paw.wfs.basis_functions.lcao_to_grid(C_M, psit_G, kpt.q)
                for c in range(3): 
                     grad[c].apply(psit_G, grad_psit_G[c],kpt.phase_cd)

                rxnabla_g[0] += -1j*f*paw.wfs.gd.integrate(psit_G.conjugate() *
                                                        (r_cG[1] * grad_psit_G[2] -
                                                         r_cG[2] * grad_psit_G[1]))
                rxnabla_g[1] += -1j*f*paw.wfs.gd.integrate(psit_G.conjugate() *
                                                        (r_cG[2] * grad_psit_G[0] -
                                                         r_cG[0] * grad_psit_G[2]))
                rxnabla_g[2] += -1j*f*paw.wfs.gd.integrate(psit_G.conjugate() *
                                                        (r_cG[0] * grad_psit_G[1] -
                                                         r_cG[1] * grad_psit_G[0]))

                # augmentation contributions to magnetic moment
                # <psi1| r x nabla |psi2> = <psi1| (r-Ra+Ra) x nabla |psi2>
                #                         = <psi1| (r-Ra) x nabla |psi2> + Ra x <psi1| nabla |psi2>


                # <psi1| (r-Ra) x nabla |psi2>
                for a, P_ni in kpt.P_ani.items():
                    P_i = P_ni[n]
                    def skew(a):
                        return (a-a.T)/2

                    #print(world.rank,a,n)
           
                    rxnabla_iiv = paw.wfs.setups[a].rxnabla_iiv.copy()
                    nabla_iiv = paw.wfs.setups[a].nabla_iiv.copy()

                    for c in range(3):
                        rxnabla_iiv[:,:,c]=skew(rxnabla_iiv[:,:,c])
                        nabla_iiv[:,:,c]=skew(nabla_iiv[:,:,c])     



                    Rxnabla_a2[0] += np.dot(P_i,np.dot(nabla_iiv[:,:,0],P_i.T.conjugate()))
                    Ra = (paw.atoms[a].position /Bohr) - R0
                    for i1, P1 in enumerate(P_i):
                        for i2, P2 in enumerate(P_i):
                            for c in range(3):
                                rxnabla_a[c] += -1j*f*P1.conjugate() * P2 * rxnabla_iiv[i1, i2, c]

                            # (y pz - z py)i + (z px - x pz)j + (x py - y px)k
                            Rxnabla_a[0] +=-1j*f* P1.conjugate() * P2 * ( Ra[1] * nabla_iiv[i1, i2, 2] - Ra[2] * nabla_iiv[i1, i2, 1] )
                            Rxnabla_a[1] +=-1j*f* P1.conjugate() * P2 * ( Ra[2] * nabla_iiv[i1, i2, 0] - Ra[0] * nabla_iiv[i1, i2, 2] )
                            Rxnabla_a[2] +=-1j*f* P1.conjugate() * P2 * ( Ra[0] * nabla_iiv[i1, i2, 1] - Ra[1] * nabla_iiv[i1, i2, 0] )
                               
        paw.wfs.bd.comm.sum(rxnabla_a) # sum up from different procs  
        paw.wfs.gd.comm.sum(rxnabla_a) # sum up from different procs  

        paw.wfs.bd.comm.sum(Rxnabla_a) # sum up from different procs
        paw.wfs.gd.comm.sum(Rxnabla_a) # sum up from different procs

        paw.wfs.bd.comm.sum(rxnabla_g) # sum up from different procs

        self.timer.stop('CD')

        rtots = rxnabla_g + Rxnabla_a + rxnabla_a #  summaa eri r-termit ja tarvittaessa PAW off
        
        # paw.wfs.gd.comm.sum(rtots)

        # return rxnabla_g + Rxnabla_a + rxnabla_a # 

        return rtots


    def _write_cd(self, paw):
        time = paw.time

        cd = self.calculate_cd_moment(paw,center=self.do_center)
        #line = ('%20.8lf %22.12le %22.12le %22.12le %22.12le %22.12le %22.12le\n' %
        # (time, rxnabla_g[0].real, Rxnabla_a[0].real, rxnabla_a[0].real, rxnabla_g[0].imag, Rxnabla_a[0].imag, rxnabla_a[0].imag))
        

        line = ('%20.8lf %22.12le %22.12le %22.12le\n' %
          (time, cd[0].real, cd[1].real, cd[2].real))


        self._write(line)

    def _update(self, paw):
        #if paw.action == 'init':
        #    self._write_header(paw)
        #elif paw.action == 'kick':
        #    self._write_kick(paw)
        self._write_cd(paw)

    def __del__(self):
        if self.master:
            self.fd.close()
        TDDFTObserver.__del__(self)










