from gpaw import GPAW

calc = GPAW('Pt_gs.gpw', txt=None)

calc = calc.fix_density(
    kpts={'path': 'GXWLGKX', 'npoints': 200},
    symmetry='off',
    txt='Pt_bands.txt')
calc.write('Pt_bands.gpw')
