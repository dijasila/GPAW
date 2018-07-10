import numpy as np

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

    print(1)

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
   
            print (2) 

    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def calculate_cd_moment(self, paw, center=True):
        gd=paw.wfs.gd
        grad = []
        dtype = np.float
        for c in range(3):
            grad.append(Gradient(gd, c, dtype=dtype, n=2))

        grad_psit_G = [gd.empty(), gd.empty(),
                      gd.empty()]


        rxnabla_g = np.zeros(3)
        
        R0 = 0.5 * np.diag(paw.wfs.gd.cell_cv)

        r_cG, r2_G = coordinates(paw.wfs.gd, origin=R0)

        
        print (3) 

        for kpt in paw.wfs.kpt_u:
            for f, C_M in zip(kpt.f_n, kpt.C_nM):
                psit_G=gd.empty()
                paw.wfs.basis_functions.lcao_to_grid(C_M, psit_G, kpt.q)
                
 
                # Gradients
                for c in range(3):
                     grad[c].apply(psit_G, grad_psit_G[c],kpt.phase_cd)

                # <psi1|r x grad|psi2>
                #    i  j  k
                #    x  y  z   = (y pz - z py)i + (z px - x pz)j + (x py - y px)
                #    px py pz

                rxnabla_g[0] += paw.wfs.gd.integrate(psit_G *
                                                      (r_cG[1] * grad_psit_G[2] -
                                                       r_cG[2] * grad_psit_G[1]))
                rxnabla_g[1] += paw.wfs.gd.integrate(psit_G *
                                                      (r_cG[2] * grad_psit_G[0] -
                                                       r_cG[0] * grad_psit_G[2]))
                rxnabla_g[2] += paw.wfs.gd.integrate(psit_G *
                                                      (r_cG[0] * grad_psit_G[1] -
                                                       r_cG[1] * grad_psit_G[0]))

        print(4)    
        return rxnabla_g


       

    def _write_cd(self, paw):
        time = paw.time

        cd = self.calculate_cd_moment(paw,center=self.do_center)
        line = ('%20.8lf %22.12le %22.12le %22.12le\n' %
                (time, cd[0], cd[1], cd[2]))
        self._write(line)

    def _update(self, paw):
        if paw.action == 'init':
            self._write_header(paw)
        elif paw.action == 'kick':
            self._write_kick(paw)
        self._write_cd(paw)

    def __del__(self):
        if self.master:
            self.fd.close()
        TDDFTObserver.__del__(self)
