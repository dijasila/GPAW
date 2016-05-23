import numpy as np
from ase.units import Bohr

from gpaw.fftw import get_efficient_fft_size, FFTPlan
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LFC
from gpaw.utilities import h2gpts


class Interpolator:
    def __init__(self, n_c, N_c):
        self.tmp_r = np.empty(n_c, dtype=complex)
        self.tmp_R = np.empty(N_c, dtype=complex)
        self.fftplan = FFTPlan(self.tmp_r, self.tmp_r, -1)
        self.ifftplan = FFTPlan(self.tmp_R, self.tmp_R, 1)
        self.slice = tuple(slice(N // 2 - n // 2, N // 2 + n // 2)
                           for n, N in zip(n_c, N_c))
        print(self.slice)

    def interpolate(self, a_r):
        print(a_r.sum())
        b_r = self.tmp_r
        b_R = self.tmp_R
        b_r[:] = a_r
        self.fftplan.execute()
        b_R[:] = 0.0
        #b_R[self.slice] = np.fft.fftshift(b_r) / b_r.size
        n_c = np.array(b_r.shape)
        N_c = np.array(b_R.shape)
        a0, a1, a2 = N_c // 2 - n_c // 2
        b0, b1, b2 = n_c + (a0, a1, a2)
        print(a0,a1,a2)
        print(b0,b1,b2)
        print(b_r.sum()/b_r.size)
        c_r = np.fft.fftshift(b_r) / b_r.size
        print(c_r.sum())
        c_r[0]*=0.5
        c_r[:,0]*=0.5
        c_r[:,:,0]*=0.5
        
        b_R[a0:b0, a1:b1, a2:b2] = c_r
        
        b_R[b0, a1:b1, a2:b2] = b_R[a0, a1:b1, a2:b2]
        b_R[a0:b0+1, b1, a2:b2] = b_R[a0:b0+1, a1, a2:b2]
        b_R[a0:b0+1, a1:b1+1, b2] = b_R[a0:b0+1, a1:b1+1, a2]
        
        b_R[b0, a1+1:b1, a2+1:b2] = b_R[a0, a1+1:b1, a2+1:b2][::-1,::-1]
        b_R[a0+1:b0, b1, a2+1:b2] = b_R[a0+1:b0, a1, a2+1:b2][::-1,::-1]
        b_R[a0+1:b0, a1+1:b1, b2] = b_R[a0+1:b0, a1+1:b1, a2][::-1,::-1]
        print(b_R.sum())
        
        if 0:
            b_R[a0, a1:b1, a2:b2] *= 0.5
            b_R[b0, a1:b1, a2:b2] = b_R[a0, b1-1:a1-1, b2-1:a2-1].conj()
            b0 += 1
        if 0:
            b_R[a0:b0, a1, a2:b2] *= 0.5
            b_R[a0:b0, b1, a2:b2] = b_R[a0:b0, a1, a2:b2]
            b1 += 1
        if 0:
            if 1:
                print(b_R[a0:b0, a1:b1, a2])
                b_R[a0:b0, a1:b1, a2] *= 0.5
                b_R[a0:b0, a1:b1, b2] = b_R[a0:b0, a1:b1, a2]
        
        b_R[:] = np.fft.ifftshift(b_R)
        #print(b_r[0,0]/b_r.size)
        #print(b_R[0,0])
        self.ifftplan.execute()
        b_R = b_R.copy()
        if a_r.dtype == float:
            print(b_R.imag.ptp())
            b_R = b_R.real.copy()
        print(b_R.sum())
        return b_R
        

class PS2AE:
    """Transform PS to AE wave functions.
    
    Interpolates PS wave functions to a fine grid and adds PAW
    corrections in order to obtain true AE wave functions.
    """
    def __init__(self, calc, h=0.05, n=2):
        """Create transformation object.
        
        calc: GPAW calculator object
            The calcalator that has the wave functions.
        h: float
            Desired grid-spacing in Angstrom.
        n: int
            Force number of points to be a mulitiple of n.
        """
        self.calc = calc
        gd = calc.wfs.gd
        
        # Descriptor for the final grid:
        N_c = h2gpts(h / Bohr, gd.cell_cv)
        N_c = np.array([get_efficient_fft_size(N, n) for N in N_c])
        self.gd = GridDescriptor(N_c, gd.cell_cv)
        self.interpolator = Interpolator(gd.N_c, N_c)
        print(gd.N_c, N_c)
        self.dphi = None  # PAW correction (will be initialize when needed)

    def _initialize_corrections(self):
        if self.dphi is not None:
            return
        splines = {}
        dphi_aj = []
        for setup in self.calc.wfs.setups:
            dphi_j = splines.get(setup)
            if dphi_j is None:
                rcut = max(setup.rcut_j) * 1.1
                gcut = setup.rgd.ceil(rcut)
                dphi_j = []
                for l, phi_g, phit_g in zip(setup.l_j,
                                            setup.data.phi_jg,
                                            setup.data.phit_jg):
                    dphi_g = (phi_g - phit_g)[:gcut]
                    dphi_j.append(setup.rgd.spline(dphi_g, rcut, l,
                                                   points=200))
            dphi_aj.append(dphi_j)
            
        self.dphi = LFC(self.gd, dphi_aj, kd=self.calc.wfs.kd,
                        dtype=self.calc.wfs.dtype)
        print(self.calc.wfs.dtype)
        self.dphi.set_positions(self.calc.atoms.get_scaled_positions())
        
    def get_wave_function(self, n, k=0, s=0, ae=True):
        """Interpolate wave function.
        
        n: int
            Band index.
        k: int
            K-point index.
        s: int
            Spin index.
        ae: bool
            Add PAW correction to get an all-electron wave function.
        """
        psi_r = self.calc.get_pseudo_wave_function(n, k, s, pad=True,
                                                   periodic=True).real
        psi_R = self.interpolator.interpolate(psi_r * Bohr**1.5)
        if ae:
            self._initialize_corrections()
            wfs = self.calc.wfs
            kpt_rank, u = wfs.kd.get_rank_and_index(s, k)
            band_rank, n = wfs.bd.who_has(n)
            assert kpt_rank == 0 and band_rank == 0
            P_ai = dict((a, P_ni[n]) for a, P_ni in wfs.kpt_u[u].P_ani.items())
            self.dphi.add(psi_R, P_ai, k)
        return psi_R
