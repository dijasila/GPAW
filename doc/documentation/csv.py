# creates: ug.csv, ugf.csv, pw.csv, pwe.csv
from gpaw.core import UniformGrid, PlaneWaves
from gpaw.core.plane_waves import PlaneWaveExpansions
from gpaw.core.uniform_grid import UniformGridFunctions


for cls in [UniformGrid,
            PlaneWaves,
            UniformGridFunctions,
            PlaneWaveExpansions]:
    name = ''.join(x for x in cls.__name__ if x.isupper()).lower()
    mod = cls.__module__
    if name == 'ug':
        mod = mod.replace('.uniform_grid', '')
    elif name == 'pw':
        mod = mod.replace('.plane_waves', '')
    print(name, mod)
    with open(f'{name}.csv', 'w') as fd:
        for name, meth in cls.__dict__.items():
            if name[0] != '_':
                try:
                    doc = meth.__doc__.splitlines()[0]
                except AttributeError:
                    doc = '...'
                print(f':meth:`~{mod}.{cls.__name__}.{name}`, "{doc}"',
                      file=fd)
