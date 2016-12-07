from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac
from ase.io import read
from ase.visualize import view


structure = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                size=(1, 1, 1), vacuum=5)
structure.pbc = (1, 1, 1)

#structure = read('/home/niflheim2/s093017/MoS2/WS2_18x18x1_PBErelaxed.gpw')
setup_list = ['paw', 'Mo_1', 'Mo_2', 'Mo_nc_1', 'Mo_nc_2']
Ecut = 1000

for setup in setup_list:
    calc = GPAW(mode=PW(Ecut),
                xc='PBE',
                kpts={'size': (18,18,1), 'gamma': True},
                setups={'Mo': setup},			
                occupations=FermiDirac(0.01),
                txt=setup + '_out_gs.txt')

    structure.set_calculator(calc)
    structure.get_potential_energy()
    calc.write('gs_' + setup  + '.gpw', 'all')

    calc.diagonalize_full_hamiltonian(nbands=2000,expert=True)
    calc.write(setup + '_fulldiag.gpw', 'all')

