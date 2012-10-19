from ase import Atoms
from ase.units import Hartree
from ase.data.tmfp06d import data
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.hubu import HubU

molecule = 'ScH'

energy_tolerance = 0.0004
h =.2
box = 3.0

# applying U on the 3d orbital 
alpha = 0.05/Hartree    # potential shift  

# Find U0 on 3d and 4s sites. Both spin independent 
#HubU_IO_dict = {a:{n:{l:1}}
HubU_IO_dict = {0:{3:{2:1},#3d, on both electrons
                   }}
                
# Specifications of procedure
scale = 1       # Should the projections be normalized
NbP = 1         # Use the non-occupied orbitals when available
background = 1  # Have a background term in the calculation of Hubbard


########################
gpw_filename = molecule+'.gpw'


data_mol = data[molecule]
sys = Cluster(Atoms(
                  data_mol['symbols'],
                  positions=data_mol['positions'],
                  magmoms=data_mol['magmoms'])
            )
sys.minimal_box(box,h=h)
c = GPAW(h=h,
         xc = 'PBE')
sys.set_calculator(c)
sys.get_potential_energy()

Hubu = HubU(c)
HubU_dict = Hubu.get_MS_Usc(
                   HubU_IO_dict, 
                   background = background,
                   alpha = alpha,
                   scale = scale,
                   NbP = NbP,
                   factors = [0.6, 0.8, 1.0],
                   )
orbital = ['s','p','d','f']

for a in HubU_dict:
    for n in HubU_dict[a]:
        for l in HubU_dict[a][n]:
            print a, str(n)+orbital[l], 
            print'U:',HubU_dict[a][n][l][0]['U']*Hartree
