def test_diamond():
    from ase.build import bulk
    from gpaw import GPAW

    k = 4
    atoms = bulk('C', 'diamond')
    atoms.calc = GPAW(kpts={'size': (k, k, k), 'gamma': True},
                      mode='lcao', basis='dzp',
                      xc='PBE')
    atoms.get_potential_energy()

if __name__ == "__main__":
    test_diamond()
