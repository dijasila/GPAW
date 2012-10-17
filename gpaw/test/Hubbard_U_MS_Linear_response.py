from ase import Atoms
from ase.units import Hartree
from ase.data.tmfp06d import data
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.hubu import HubU
import numpy as np
molecule = 'ScH'

energy_tolerance = 0.0004
h =.16
box = 3.5

# applying U on the 3d orbital 
alpha = 0.05/Hartree    # potential shift  

# Find U0 on 3d and 4s sites. Both spin independent 
#HubU_IO_dict = {a:{n:{l:1}}
HubU_IO_dict = {0:{4:{0:1}, #4s
                   3:{2:1},#3d
                   
                   }}
                
# Second atom 
scale = 1       #
NbP = 1         #
background = 1  #


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
U0_au, X0, Xks = Hubu.get_MS_linear_response_U0(HubU_IO_dict,
                                 background=background,
                                 scale = scale, 
                                 NbP = NbP, 
                                 alpha = alpha)
U0 = np.array(U0_au)*Hartree

orbital = ['s','p','d','f']
jj = 0
for a in HubU_IO_dict:
    for n in HubU_IO_dict[a]:
        for l in HubU_IO_dict[a][n]:
            print a, str(n)+orbital[l], 'U:',U0[jj,jj]
            jj+=1
