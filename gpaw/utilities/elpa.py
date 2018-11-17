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


def elpa_diagonalize(desc, A, C, eps):
    bg = desc.blacsgrid
    na = desc.gshape[0]
    nev = len(eps)
    code = _gpaw.elpa_general_diagonalize(
        bg.comm.get_c_object(),
        bg.context,
        A, A.copy(), C, eps,
        desc.gshape[0], nev,
        (bg.npcol, bg.nprow, bg.mycol, bg.myrow),
        desc.shape, desc.mb)
    return code

def elpa_general_diagonalize(desc, A, S, C, eps):
    bg = desc.blacsgrid
    comm = bg.comm

    for arr in [A, C, S, eps]:
        assert arr.dtype == float
        assert arr.flags.contiguous

    #for arr in [A, C, S]:
    #    desc.checkassert(arr)

    na = desc.gshape[0]
    nev = len(eps)
    assert nev <= na
    assert eps.shape == (nev,)
    assert desc.mb == desc.nb
    for arr in [A, S, C]:
        assert arr.shape == desc.shape

    code = _gpaw.elpa_general_diagonalize(
        comm.get_c_object(),
        bg.context,
        A, S, C, eps,
        na, nev,
        # Tricky/important: row and col definition apparently swapped
        # between GPAW and ScaLAPACK/Elpa
        #(bg.nprow, bg.npcol, bg.myrow, bg.mycol),
        (bg.npcol, bg.nprow, bg.mycol, bg.myrow),
        desc.shape, desc.mb)
    if code != 0:
        raise RuntimeError('Elpa general diagonalization failed with code '
                           '{}'.format(code))
    #print('epsout', eps)
