import numpy as np
import _gpaw


class LibElpa:
    def __init__(self, desc, **kwargs):
        ptr = np.zeros(1, np.intp)

        if desc.nb != desc.mb:
            raise ValueError('Row and column block size must be '
                             'identical to support Elpa')

        if not hasattr(_gpaw, 'pyelpa_allocate'):
            raise ImportError('GPAW is not running in parallel or otherwise '
                              'not compiled with Elpa support')
        _gpaw.pyelpa_allocate(ptr)
        self._ptr = ptr
        _gpaw.pyelpa_set_comm(ptr, desc.blacsgrid.comm)
        self._parameters = {}

        bg = desc.blacsgrid
        self.elpa_set(na=desc.gshape[0],
                      local_ncols=desc.shape[0],
                      local_nrows=desc.shape[1],
                      nblk=desc.mb,
                      process_col=bg.myrow,  # XXX interchanged
                      process_row=bg.mycol,
                      blacs_context=bg.context)
        # remember: nev
        self.elpa_set(**kwargs)
        self._desc = desc
        _gpaw.pyelpa_setup(self._ptr)


    def diagonalize(self, A, C, eps):
        assert self._parameters.get('nev') == len(eps), 'bad "nev"'
        self._desc.checkassert(A)
        _gpaw.pyelpa_diagonalize(self._ptr, A, C, eps)

    def elpa_set(self, **kwargs):
        for key, value in kwargs.items():
            print('pyelpa_set {}={}'.format(key, value))
            _gpaw.pyelpa_set(self._ptr, key, value)
            self._parameters[key] = value

    def __repr__(self):
        return 'LibElpa({})'.format(self._parameters)

    def __del__(self):
        if hasattr(self, '_ptr'):
            _gpaw.pyelpa_deallocate(self._ptr)
            self._ptr[0] = 0

def elpa_diagonalize(desc, A, C, eps):
    bg = desc.blacsgrid
    na = desc.gshape[0]
    nev = len(eps)
    #print('THE FSCKING MATRIX')
    #print(A)
    code = _gpaw.pyelpa_diagonalize(
        bg.comm.get_c_object(),
        bg.context,
        A, C, eps,
        desc.gshape[0], nev,
        (bg.npcol, bg.nprow, bg.myrow, bg.mycol),
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
