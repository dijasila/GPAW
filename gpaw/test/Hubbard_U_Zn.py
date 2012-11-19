from ase import Atom
from ase.units import Hartree

from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster
from gpaw.test import equal

h =.3
box = 4.
energy_tolerance = 0.0004

a=0                 # Zn is the 0 index atom 
n=3                 # third orbital
l=2                 # d-orbitals
U_ev=3              # U in eV
U_au=U_ev / Hartree # U in atomic units
scale=1             # scale (to be investigated what the influence is!)

HubU_dict = {a:{
                n:{
                   l:{'U':U_au}
                   }},
            'scale': scale,
            }

s = Cluster([Atom('Zn')])
s.minimal_box(box, h=h)

E = {}
E_U = {}

for spin in [0, 1]:
    c = GPAW(h=h, spinpol=spin,
             eigensolver='cg',
             charge=1, occupations=FermiDirac(width=0.1, fixmagmom=spin)
             )
    s.set_calculator(c)
    E[spin] = s.get_potential_energy()
    c.hamiltonian.set_hubbard_u(HubU_dict=HubU_dict)    
    c.scf.reset()
    E_U[spin] = s.get_potential_energy()

print "E=", E
equal(E[0], E[1], energy_tolerance)
print "E_U=", E_U
equal(E_U[0], E_U[1], energy_tolerance)