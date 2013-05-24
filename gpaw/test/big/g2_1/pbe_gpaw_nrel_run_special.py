import os
from gpaw.test.big.g2_1.pbe_gpaw_nrel_analyse import tag
os.system("""export GPAW_SETUP_PATH=.&& mpiexec -np 1 gpaw-python `which ase` gpaw molecule OH C Si -t """ + tag + """ --atomize --unit-cell='17.00,17.01,17.02' -l -p xc='PBE',mode='PW(1100)',width=0.0,fixmom=True,nbands=-2,setups=nrel,mixer='Mixer(0.05,2),maxiter=400'""")
