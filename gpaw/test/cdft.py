from ase import Atoms
from gpaw import GPAW
from gpaw.cdft import CDFT

distance = 2.5
sys = Atoms('He2', positions=([0, 0, 0], [0, 0, distance]))
sys.center(4)
calc = GPAW(charge=1, xc='PBE', txt='he2.txt')
sys.calc = CDFT(calc, [[0]], [1])
e = sys.get_potential_energy()
