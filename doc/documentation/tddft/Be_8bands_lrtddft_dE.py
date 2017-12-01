from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT

c = GPAW('Be_gs_8bands.gpw')

dE = 2.5 # maximal Kohn-Sham transition energy to consider
lr = LrTDDFT(c, xc='LDA', energy_range=dE)

lr.write('lr_dE.dat.gz')
