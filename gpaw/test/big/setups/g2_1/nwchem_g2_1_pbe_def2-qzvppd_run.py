import os
os.system("NWCHEM_COMMAND='mpiexec -np 4 nwchem' ase nwchem molecule g2-1 -t g2_1_pbe_def2-qzvppd --atomize -l -p geometry='noautosym',task='gradient',xc='PBE',smear=0.0,grid='nodisk',tolerances='tight',basis='def2-qzvppd',basispar='spherical',dftcontrol='direct\nnoio'")
