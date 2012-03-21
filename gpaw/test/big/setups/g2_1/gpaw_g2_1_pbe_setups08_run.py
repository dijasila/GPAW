import os
os.system("mpiexec -np 4 gpaw-python `which ase` gpaw molecule g2-1 -t g2_1_pbe_setups08 --unit-cell='16.00,16.01,16.02' --atomize -l -p xc='PBE',h=0.13,width=0.0,fixmom=True,nbands=-5")
