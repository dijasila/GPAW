from gpaw.xc.libvdwxc import VDWDF, VDWDF2, VDWDFCX
from gpaw.atom.generator import Generator
from gpaw.atom.all_electron import AllElectron

gen = AllElectron('He', xc='new_vdW-DF')
gen.run()

