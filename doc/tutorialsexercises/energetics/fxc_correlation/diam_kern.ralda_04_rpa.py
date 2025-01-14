from gpaw.xc.fxc import FXCCorrelation
from ase.parallel import paropen

fxc = FXCCorrelation('diam_kern.ralda.lda_wfcs.gpw', xc='RPA',
                     ecut=[131.072, 80.0],
                     txt='diam_kern.ralda_08_rpa.txt')
E_i = fxc.calculate()

resultfile = paropen('diam_kern.ralda_kernel_comparison.dat', 'a')
resultfile.write(str(E_i[-1]) + '\n')
resultfile.close()
