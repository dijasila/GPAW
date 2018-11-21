import numpy as np
import _gpaw

def _elpaconstants():
    consts = _gpaw.pyelpa_constants()
    return {'elpa_ok': consts[0],
            '1stage': consts[1],
            '2stage': consts[2]}


class LibElpa:
    @staticmethod
    def have_elpa():
        return hasattr(_gpaw, 'pyelpa_allocate')

    def __init__(self, desc, nev=None, solver='1stage'):
        if nev is None:
            nev = desc.gshape[0]

        ptr = np.zeros(1, np.intp)

        if not self.have_elpa():
            raise ImportError('GPAW is not running in parallel or otherwise '
                              'not compiled with Elpa support')

        if desc.nb != desc.mb:
            raise ValueError('Row and column block size must be '
                             'identical to support Elpa')

        _gpaw.pyelpa_allocate(ptr)
        self._ptr = ptr
        _gpaw.pyelpa_set_comm(ptr, desc.blacsgrid.comm)
        self._parameters = {}

        self._consts = _elpaconstants()
        elpasolver = self._consts[solver]

        bg = desc.blacsgrid
        self.elpa_set(na=desc.gshape[0],
                      local_ncols=desc.shape[0],
                      local_nrows=desc.shape[1],
                      nblk=desc.mb,
                      process_col=bg.myrow,  # XXX interchanged
                      process_row=bg.mycol,
                      blacs_context=bg.context)
        # remember: nev
        self.elpa_set(nev=nev, solver=elpasolver)
        self.desc = desc

        _gpaw.pyelpa_setup(self._ptr)

    @property
    def description(self):
        solver = self._parameters['solver']
        if solver == self._consts['1stage']:
            pretty = 'Elpa one-stage solver'
        else:
            assert solver == self._consts['2stage']
            pretty = 'Elpa two-stage solver'
        return pretty

    @property
    def nev(self):
        return self._parameters['nev']

    def diagonalize(self, A, C, eps):
        assert self.nev == len(eps)
        self.desc.checkassert(A)
        self.desc.checkassert(C)
        err = _gpaw.pyelpa_diagonalize(self._ptr, A, C, eps)
        assert err == 0

    def general_diagonalize(self, A, S, C, eps):
        assert self.nev == len(eps)
        self.desc.checkassert(A)
        self.desc.checkassert(S)
        self.desc.checkassert(C)
        err = _gpaw.pyelpa_general_diagonalize(self._ptr, A, S, C, eps)
        assert err == 0

    def elpa_set(self, **kwargs):
        for key, value in kwargs.items():
            # print('pyelpa_set {}={}'.format(key, value))
            _gpaw.pyelpa_set(self._ptr, key, value)
            self._parameters[key] = value

    def __repr__(self):
        return 'LibElpa({})'.format(self._parameters)

    def __del__(self):
        if hasattr(self, '_ptr'):
            _gpaw.pyelpa_deallocate(self._ptr)
            self._ptr[0] = 0
