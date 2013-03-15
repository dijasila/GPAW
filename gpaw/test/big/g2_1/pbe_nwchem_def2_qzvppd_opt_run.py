import os
from gpaw.test.big.g2_1.pbe_nwchem_def2_qzvppd_opt_analyse import tag
os.system("ASE_NWCHEM_COMMAND='mpiexec -np 4 nwchem PREFIX.nw > PREFIX.out' NWCHEM_COMMAND='mpiexec -np 4 nwchem' ase nwchem molecule g2_1 -t " + tag + " --atomize -l -R 0.01 -p geometry='noautosym nocenter noautoz',task='gradient',xc='PBE',width=0.0,grid='nodisk',tolerances='tight',basis='def2-qzvppd',basispar='spherical',direct='noio'")
