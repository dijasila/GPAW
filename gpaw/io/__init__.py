def Reader(filename):
    import ase.io.aff as aff
    try:
        return aff.Reader(filename)
    except aff.InvalidAFFError:
        pass
    from gpaw.io.old import wrap_old_gpw_reader
    return wrap_old_gpw_reader(filename)

    
def Writer(filename, world, tag='GPAW'):
    import ase.io.aff as aff
    if world.rank == 0:
        return aff.Writer(filename, tag=tag)
    return aff.DummyWriter()
