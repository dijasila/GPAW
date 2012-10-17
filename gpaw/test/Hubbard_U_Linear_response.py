from ase import Atoms
from ase.units import Hartree
from ase.data.tmfp06d import data
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.hubu import HubU

molecule = 'ScF'


energy_tolerance = 0.0004
h =.2
box = 3.

# applying U on the 3d orbital 
alpha = 0.05/Hartree    # potential shift  
s = 0           # spin 0,non-spinpolerized 
a = 0           # atom index
n = 3           # quantum number
l = 2           # angular momentum
scale = 1       #
NbP = 1         #

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
U0 = Hubu.get_linear_response_U0(a,n,l,s,scale = scale, 
                                 NbP = NbP, alpha = alpha)
print 'U in eV:', U0*Hartree