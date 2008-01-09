"""Module for building atomic structures"""

from ase import *
from math import sqrt

def bcc100(symbol, a, layers, h):
    """Build a bcc(100) surface

    symbol: chemical symbol ('H', 'Li', ...)
    a     : lattice constant
    layers: number of layers
    h     : height of unit cell"""

    # Distance between layers:
    z = a / 2

    assert h > layers * z, 'unit cell too small for ' + str(layers) + ' layers'
    
    # Start with an empty Atoms object with an orthorhombic unit cell:
    atoms = Atoms(pbc=(True, True, False), cell=(a, a, h))
    
    # Fill in the atoms:
    for n in range(layers):
        position = [a / 2 * (n % 2),
                    a / 2 * (n % 2),
                    h / 2 + (0.5 * (layers - 1) - n) * z]
        atoms.append(Atom(symbol, position))
        
    return atoms


def fcc100(symbol, a, layers, h):
    """Build a fcc(100) surface

    symbol: chmical symbol ('H', 'Li', ...)
    a     : lattice constant
    layers: number of layers
    h     : height of unit cell"""

    # Distance between atoms:
    d = a / sqrt(2)

    # Distance between layers:
    z = a / sqrt(3)

    assert h > layers * z, 'unit cell too small for ' + str(layers) + ' layers'
    
    # Start with an empty Atoms object:
    atoms = Atoms(cell=(d, d, h), pbc=(1, 1, 0))
    
    # Fill in the atoms:
    for n in range(layers):
        position = [d / 2 * (n % 2),
                    d / 2 * (n % 2),
                    h / 2 + (0.5 * (layers - 1) - n) * z]
        atoms.append(Atom(symbol, position))

    return atoms


if __name__ == '__main__':
    bcc = bcc100('Al', 4.0, 4, 15.0)
    from ase import view
    view(bcc * (4, 4, 2))
