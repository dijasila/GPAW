import os
from gpaw.test.big.setups.g2_1.fix.pbe_nwchem_def2_qzvppd_analyse import tag
os.system("NWCHEM_COMMAND='mpiexec -np 4 nwchem' ase nwchem molecule g2-1 -t " + tag + " --atomize -l -p geometry='noautosym nocenter',task='gradient',xc='PBE',smear=0.0,grid='nodisk',tolerances='tight',basis='def2-qzvppd',basispar='spherical',dftcontrol='direct\nnoio'")
