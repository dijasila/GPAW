import os
os.system("NWCHEM_COMMAND='mpiexec -np 4 nwchem' ase nwchem molecule g2-1 -t 95g2_1_pbe_def2_qzvppd --modify='system.set_positions(system.get_positions()*0.95);system.center()' --atomize -l -p geometry='noautosym',task='gradient',xc='PBE',smear=0.0,grid='nodisk',tolerances='tight',basis='def2-qzvppd',basispar='spherical',dftcontrol='direct\nnoio'")
