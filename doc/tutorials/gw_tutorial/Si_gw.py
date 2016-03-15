import pickle
import numpy as np
from gpaw.response.g0w0 import G0W0

gw = G0W0(calc='Si_groundstate.gpw',
          nbands=30,                # number of bands for calculation of self-energy
          bands=(3,5),               # here: all valence bands and lowest conduction bands
          ecut=20.,                  # plane wave cutoff for self-energy
          filename='Si-g0w0',
          savepckl=True              # a .pckl is saved with e.g. the resulting QP energies.
          )

result = gw.calculate()


