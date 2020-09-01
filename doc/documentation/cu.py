from ase.build import bulk
from gpaw import GPAW, PW

cu = bulk('Cu', 'fcc', a=3.6)

for smearing in [{'name': 'tetrahedron-method'},
                 {'name': 'improved-tetrahedron-method'},
                 {'name': 'marzari-vanderbilt', 'width': 0.2},
                 {'name': 'fermi-dirac', 'width': 0.05}]:
    name = ''.join(word[0].upper() for word in smearing['name'].split('-'))
    width = smearing.get('width')
    if width:
        name += f'-{width}'

    # for n in range(6, 21):
    for n in range(6, 9):
        cu.calc = GPAW(
            mode=PW(400),
            kpts=(n, n, n),
            occupations=smearing,
            txt=f'Cu-{name}-{n}.txt')
        e = cu.get_potential_energy()
