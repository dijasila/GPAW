# web-page: si-gaps.csv
from ase.build import bulk
from ase.parallel import paropen
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw import GPAW, PW

a = 5.43
si = bulk('Si', 'diamond', a)

fd = paropen('si-gaps.csv', 'w')

for k in range(2, 9, 2):
    name = f'Si-{k}'
    si.calc = GPAW(kpts={'size': (k, k, k), 'gamma': True},
                   mode=PW(200),
                   xc='PBE',
                   convergence={'bands': 5},
                   txt=name + '.txt')
    si.get_potential_energy()
    si.calc.write(name + '.gpw', mode='all')

    # Range of eigenvalues:
    n1 = 3
    n2 = 5

    ibzkpts = si.calc.get_ibz_k_points()
    kpt_indices = []
    for kpt in [(0, 0, 0), (0.5, 0.5, 0)]:  # Gamma and X
        # Find k-point index:
        i = abs(ibzkpts - kpt).sum(1).argmin()
        kpt_indices.append(i)

    # Do PBE0 calculation:
    epbe, vpbe, vpbe0 = non_self_consistent_eigenvalues(
        name + '.gpw',
        'PBE0',
        n1, n2,
        kpt_indices,
        snapshot=name + '.json')

    epbe0 = epbe - vpbe + vpbe0

    gg = epbe[0, 0, 1] - epbe[0, 0, 0]
    gx = epbe[0, 1, 1] - epbe[0, 0, 0]
    gg0 = epbe0[0, 0, 1] - epbe0[0, 0, 0]
    gx0 = epbe0[0, 1, 1] - epbe0[0, 0, 0]

    print(f'{k}, {gg:.3f}, {gx:.3f}, {gg0:.3f}, {gx0:.3f}', file=fd)
    fd.flush()

assert abs(gg - 2.559) < 0.01
assert abs(gx - 0.707) < 0.01
assert abs(gg0 - 3.873) < 0.01
assert abs(gx0 - 1.828) < 0.01
