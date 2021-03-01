from gpaw.grid_descriptor import GridDescriptor
from gpaw.external import (ConstantElectricField, PointChargePotential,
                           PotentialCollection)


def test_collection():
    a = 4.0
    N = 48
    gd = GridDescriptor((N, N, N), (a, a, a), 0)

    ext1 = ConstantElectricField(1)
    ext2 = PointChargePotential([1, -5], positions=((0, 0, -10), (0, 0, 10)))
    collection = PotentialCollection([ext1, ext2])
    
    ext1.calculate_potential(gd)
    ext2.calculate_potential(gd)
    collection.calculate_potential(gd)

    assert (collection.vext_g == ext1.vext_g + ext2.vext_g).all()

    assert len(collection.todict()['potentials']) == 2
    print(collection)
