import numpy as np
from ase.units import Bohr

from gpaw.fftw import get_efficient_fft_size
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LFC
from gpaw.utilities import h2gpts
from gpaw.wavefunctions.pw import PWDescriptor


class PS2AE:
    """Transform PS to AE wave functions.
    
    Interpolates PS wave functions to a fine grid and adds PAW
    corrections in order to obtain true AE wave functions.
    """
    def __init__(self, calc, h=0.05, n=2, grid = None):
        """Create transformation object.
        
        calc: GPAW calculator object
            The calcalator that has the wave functions.
        h: float
            Desired grid-spacing in Angstrom.
        n: int
            Force number of points to be a mulitiple of n.
        """
        self.calc = calc
        self. grid = grid
        # Create plane-wave descriptor for starting grid:
        gd0 = GridDescriptor(calc.wfs.gd.N_c, calc.wfs.gd.cell_cv)
        self.pd0 = PWDescriptor(ecut=None, gd=gd0)
        
        # ... and a descriptor for the final gris:
        if self.grid is None:
            N_c = h2gpts(h / Bohr, gd0.cell_cv, n)
            N_c = np.array([get_efficient_fft_size(N) for N in N_c])
        else:
            N_c = self.grid.shape
        self.gd = GridDescriptor(N_c, gd0.cell_cv)
        self.pd = PWDescriptor(ecut=None, gd=self.gd)
        
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
            
        self.dphi = LFC(self.gd, dphi_aj)
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
        psi_r = self.calc.get_pseudo_wave_function(n, k, s, pad=True)
        psi_r *= Bohr**1.5
        psi_R, _ = self.pd0.interpolate(psi_r, self.pd)
        if ae:
            self._initialize_corrections()
            wfs = self.calc.wfs
            kpt_rank, u = wfs.kd.get_rank_and_index(s, k)
            band_rank, n = wfs.bd.who_has(n)
            assert kpt_rank == 0 and band_rank == 0
            P_ai = dict((a, P_ni[n]) for a, P_ni in wfs.kpt_u[u].P_ani.items())
            self.dphi.add(psi_R, P_ai)
        return psi_R
