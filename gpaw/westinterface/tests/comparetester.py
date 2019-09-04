from ase import Atoms
from gpaw import GPAW, PW
import numpy as np
from gpaw import Mixer
import argparse
from ase.parallel import parprint
from ase.visualize import view

Ry2eV = 13.6056981

parser = argparse.ArgumentParser()

add = parser.add_argument

add("-x", "--xc", type=str, default="PBE")
add("-c", "--cutoff", type=float, default=45*Ry2eV)
add("-n", "--nbands", type=int, default=50)
add("-f", "--filename", type=str, default="calc")
args = parser.parse_args()

xc = args.xc
cutoff = args.cutoff
nbands = args.nbands
fname = args.filename

parprint("-----------")
parprint("Settings:")
parprint("xc: {}".format(xc))
parprint("cutoff: {}".format(cutoff))
parprint("nbands: {}".format(nbands)) 
parprint("-----------")

bohr2A = 0.529177
lattice_constant = bohr2A * 47.24323#25.0000413052

atoms = Atoms("SiH4", scaled_positions=[[0.5, 0.5, 0.5],
                                 [0.534179, 0.465824, 0.534176],
                                 [0.465824, 0.534176, 0.534176],
                                 [0.465824, 0.465824, 0.465824],
                                 [0.534176, 0.534176, 0.465824]], cell=lattice_constant*np.identity(3))

atoms.center()


calc = GPAW(mode=PW(cutoff), h=0.235, symmetry="off", xc=xc, mixer=Mixer(0.02, 5, 100), nbands=nbands, setups="sg15")#, txt=fname+".txt")
# TODO Do diagonalize_full_hamiltonian(nbands=??) in server

atoms.set_calculator(calc)
atoms.get_potential_energy()

if not fname.endswith(".gpw"):
    fname = fname + ".gpw"

calc.write(fname, mode="all")
