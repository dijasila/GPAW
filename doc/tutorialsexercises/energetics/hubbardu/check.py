# web-page: gaps.csv
from ase.io import read


with open('gaps.csv', 'w') as fd:
    gaps = []
    for name in ['no_u', 'normalized_u', 'not_normalized_u']:
        n = read(name + '.txt')
        gap = n.calc.get_eigenvalues(spin=1)[1] - \
            n.calc.get_eigenvalues(spin=0)[1]
        gaps.append(gap)
        print(f"{name.replace('_', ' ').replace('u', 'U')}, {gap:.3f}",
              file=fd)

assert abs(gaps[1] - gaps[0] - 6.0) < 0.8
