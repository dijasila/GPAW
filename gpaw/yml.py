import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils.plugins import ExternalIOFormat

gpaw_yaml = ExternalIOFormat(
    desc='GPAW-yaml output',
    code='+B',
    module='gpaw.yml',
    magic=b'#   __  _  _',
    glob=['*.yaml', '*.yml', '*.txt'])


def read_gpaw_yaml(fd, index):
    import yaml
    configs = []
    for dct in yaml.safe_load_all(fd):
        if 'atoms' in dct:
            atoms = dict2atoms(dct)
            configs.append(atoms)
    return configs[index]


def dict2atoms(dct) -> Atoms:
    symbols = []
    positions = []
    magmoms = []
    for symbol, position, (_, _, magmom) in dct['atoms']:
        symbols.append(symbol)
        positions.append(position)
        magmoms.append(magmom)
    cell = dct['cell']
    pbc = dct['periodic']
    atoms = Atoms(symbols,
                  positions,
                  cell=cell,
                  pbc=pbc)
    if 'energies' in dct:
        energy = dct['energies']['extrapolated']
        if 'forces' in dct:
            forces = dct['forces']
        else:
            forces = None
        if 'stress tensor' in dct:
            stress = np.array(dct['stress tensor']).ravel()[0, 4, 8, 5, 2, 1]
        else:
            stress = None
        atoms.calc = SinglePointCalculator(energy=energy,
                                           forces=forces,
                                           stress=stress,
                                           atoms=atoms)
    return atoms


if __name__ == '__main__':
    import sys

    import yaml
    for dct in yaml.safe_load_all(open(sys.argv[1])):
        print(dct)
