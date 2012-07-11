import os
from gpaw.test.big.setups.g2_1.fix.pbe_gpaw_nrel08_analyse import tag
os.system("""export GPAW_SETUP_PATH=.&& mpiexec -np 1 gpaw-python `which ase` gpaw molecule C ClO OH Si -t """ + tag + """ --atomize --unit-cell='17.00,17.01,17.02' -l -p xc='PBE',mode='PW(1100)',width=0.0,fixmom=True,nbands=-2,setups='{None:"nrel08"}',mixer='Mixer(0.05,2)'""")
