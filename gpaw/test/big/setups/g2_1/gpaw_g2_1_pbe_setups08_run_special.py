import os
os.system("mpiexec -np 4 gpaw-python `which ase` gpaw molecule CH NO Si2 ClO C Cl S Si -t g2_1_pbe_setups08 --unit-cell='16.00,16.01,16.02' -l -p xc='PBE',h=0.13,width=0.0,fixmom=True,nbands=-5,basis=dzp")
