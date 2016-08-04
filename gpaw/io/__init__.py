def Reader(filename):
    import ase.io.aff as aff
    return aff.Reader(filename)

    
def Writer(filename, world):
    import ase.io.aff as aff
    if world.rank == 0:
        return aff.Writer(filename, tag='GPAW')
    return aff.DummyWriter()
