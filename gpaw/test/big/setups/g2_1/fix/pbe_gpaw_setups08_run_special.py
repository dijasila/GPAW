import os
from gpaw.test.big.setups.g2_1.fix.pbe_gpaw_setups08_analyse import tag
os.system("""export GPAW_SETUP_PATH=.&& mpiexec -np 4 gpaw-python `which ase` gpaw molecule CH NO OH Si2 ClO C Cl S Si -t """ + tag + """ --unit-cell='17.00,17.01,17.02' -l -p xc='PBE',h=0.12,width=0.0,fixmom=True,nbands=-5,setups='{None:"setups08"}',basis=dzp""")
