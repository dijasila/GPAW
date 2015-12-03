from gpaw import GPAW, Davidson
from ase import Atoms
from gpaw.xc.libvdwxc import VDWDF
from gpaw.utilities import h2gpts
import numpy as np
from gpaw.mpi import rank
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD

import matplotlib.pyplot as plt

vacuum = 4
h = 0.1
#Exc radial 0.00108681358977
#Exc grid 0.0378374748211
#Exc ratio 0.0287232061576
#Exc 1/ratio 34.8150549251
#h=0.15
#Exc radial 0.00661914049258
#Exc grid 0.0384622768556 <-- check
#Exc ratio 0.172094348898
#Exc 1/ratio 5.81076605017
#h=0.075
#vdw-energy 0.0220264473 
#Exc radial 0.0220264472795
#Exc grid 0.038462815859
#Exc ratio 0.572668609605
#Exc 1/ratio 1.74621060632

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

system = Atoms('Ne')
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
print "Exc radial", Exc # XXX
print "Exc grid", xc.Enlc
for i in range(2,N//2-3):
    assert abs(xc.ref_v_g[i]-radial_v_g[i])<1e-4

if 1:
    plt.plot(r,radial_v_g, label='radial')
    plt.legend()
    plt.show()

