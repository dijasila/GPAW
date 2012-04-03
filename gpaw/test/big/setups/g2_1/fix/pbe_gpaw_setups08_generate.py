import os
import gpaw

elements='H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar'
os.system("gpaw-setup -n --name=setups08 -f PBE " + elements)
os.system("gpaw-basis -t dzp " + elements)
