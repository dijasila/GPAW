from gpaw import GPAW, Davidson
from ase import Atoms
from gpaw.xc.libvdwxc import VDWDF
from gpaw.utilities import h2gpts
import numpy as np
from gpaw.mpi import rank
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD

from sys import argv

import matplotlib.pyplot as plt

h = float(argv[1])
vacuum = float(argv[2])

class XX(VDWDF):
    def __init__(self):
        VDWDF.__init__(self)

    def calculate_nonlocal(self, n_g, sigma_g, v_g, dedsigma_g):
        energy = VDWDF.calculate_nonlocal(self, n_g, sigma_g, v_g, dedsigma_g)
        self.Enlc = energy
        # Recalculate gradients
        sigma_xg, gradn_svg = self.calculate_sigma(n_g[np.newaxis,:])
        ref_v_g = v_g.copy()
        # Add gradient correction
        self.add_gradient_correction(gradn_svg, sigma_xg, dedsigma_g[np.newaxis,:], ref_v_g[np.newaxis,:])
        # Store one radial slice
        self.ref_v_g = ref_v_g[0,0,:]
        self.ref_n_g = n_g[0,0,:]
        print self.ref_v_g
        print self.ref_n_g
        return energy

for atom in ['He','Be','Ne']:
    system = Atoms(atom)
    system.pbc = 1
    system.center(vacuum=vacuum)
    system.positions[:] = 0.0

    xc=XX()
    calc = GPAW(xc=xc,
                eigensolver=Davidson(10), 
                gpts=h2gpts(h, system.get_cell(), idiv=8))

    system.set_calculator(calc)
    system.get_potential_energy()

    N = len(xc.ref_v_g)
    xc.ref_v_g[N//2:] = 0.0
    xc.ref_n_g[N//2:] = 0.0
    h = calc.density.finegd.h_cv[0][0]
    r = np.arange(N) * h
    plt.plot(r, xc.ref_v_g, label="3d reference")

    rgd = RGD(h, N)
    radial_v_g = np.zeros_like(xc.ref_n_g)
    Exc = xc.calculate_spherical(rgd, xc.ref_n_g.reshape((1,-1)) , radial_v_g.reshape((1,-1)), add_gga=False)
    print "Exc radial", h,vacuum, Exc
    print "Exc grid ", xc.Enlc
    print "ratio    ", xc.Enlc / Exc
    
    if 1:
        plt.plot(r,radial_v_g, label='radial')
        plt.legend()
        plt.show()

    for i in range(2,N//2-3):
        assert abs(xc.ref_v_g[i]-radial_v_g[i])<1e-4


