from gpaw.io import read as base_read

# XXX defining write here is not straightforward,
# since gpaw.io.write closes the file handle

def read(paw, reader):
    base_read(paw, reader)
    paw.timer.start('Read')
    hamiltonian = paw.hamiltonian
    r = reader
    try:
        hamiltonian.Eel = r['SolvationEel']
        hamiltonian.Erep = r['SolvationErep']
        hamiltonian.Edis = r['SolvationEdis']
        hamiltonian.Ecav = r['SolvationEcav']
        hamiltonian.Etm = r['SolvationEtm']
        hamiltonian.Acav = r['SolvationAcav']
        hamiltonian.Vcav = r['SolvationVcav']
    except (AttributeError, KeyError):
        # read conventional restart file
        hamiltonian.Eel = None
        hamiltonian.Erep = None
        hamiltonian.Edis = None
        hamiltonian.Ecav = None
        hamiltonian.Etm = None
        hamiltonian.Acav = None
        hamiltonian.Vcav = None
    paw.timer.stop('Read')
