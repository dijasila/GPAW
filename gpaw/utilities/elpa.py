import _gpaw

class Elpa:
    def __init__(self):
        handle = np.empty(1, np.intp)
        _gpaw.elpa_init(handle)
        self._handle = handle

    def solve(self):
        pass

    def __del__(self):
        if hasattr(self, '_handle'):
            _gpaw.elpa_free(self._handle)


def elpa_general_diagonalize(desc, A, S, C, eps):
    bg = desc.blacsgrid
    comm = bg.comm
    for arr in [A, C, S, eps]:
        assert A.dtype == float
        assert A.flags.contiguous

    na = len(A)
    assert A.shape == (na, na)
    assert A.shape == S.shape
    nev = len(eps)
    assert nev <= na
    assert eps.shape == (nev,)
    assert C.shape == (na, nev)  # or transpose?
    print('eps0', eps)
    print('na nev', na, nev)
    print('C')
    print(C)
    print('ARRGH A')
    print(A)
    assert desc.mb == desc.nb
    print(desc)
    _gpaw.elpa_general_diagonalize(
        comm.get_c_object(),
        bg.context,
        A, S, C, eps,
        na, nev,
        (bg.npcol, bg.nprow, bg.mycol, bg.myrow),
        desc.shape, desc.mb)
    print('eps1', eps)
