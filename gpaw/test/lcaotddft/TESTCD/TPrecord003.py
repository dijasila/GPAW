from ase.io import read
from gpaw import GPAW
from gpaw import PoissonSolver
from gpaw.occupations import FermiDirac
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter
from gpaw.lcaotddft.cdwriter import CDWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum
from gpaw import setup_paths


# Time propagation
td_calc = LCAOTDDFT('methyl-oxirane.gpw')

DipoleMomentWriter(td_calc, 'rmeo003.dm')
CDWriter(td_calc,'cd_live003.dat')
#WaveFunctionWriter(td_calc, 'wf003.ulm')


td_calc.absorption_kick([1e-5, 0.0, 0.0])
td_calc.propagate(10, 1000)



td_calc.write('td0.gpw', mode='all')
