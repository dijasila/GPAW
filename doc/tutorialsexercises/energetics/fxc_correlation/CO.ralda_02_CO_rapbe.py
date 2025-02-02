from ase.parallel import paropen
from ase.units import Hartree
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation

fxc0 = FXCCorrelation('CO.ralda.pbe_wfcs_CO.gpw',
                      xc='rAPBE',
                      ecut=400,
                      nblocks=8,
                      txt='CO.ralda_02_CO_rapbe.txt')

E0_i = fxc0.calculate()

f = paropen('CO.ralda_rapbe_CO.dat', 'w')
for ecut, E0 in zip(fxc0.rpa.ecut_i, E0_i):
    print(ecut * Hartree, E0, file=f)
f.close()

rpa0 = RPACorrelation('CO.ralda.pbe_wfcs_CO.gpw',
                      ecut=400,
                      nblocks=8,
                      txt='CO.ralda_02_CO_rpa.txt')

E0_i = rpa0.calculate()

f = paropen('CO.ralda_rpa_CO.dat', 'w')
for ecut, E0 in zip(rpa0.ecut_i, E0_i):
    print(ecut * Hartree, E0, file=f)
f.close()
