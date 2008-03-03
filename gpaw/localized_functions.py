# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Atomic-centered localized functions.
"""

from math import pi, cos, sin

import numpy as npy

from gpaw import debug
from gpaw.utilities import is_contiguous
from gpaw.mpi import run
import _gpaw


MASTER = 0


def create_localized_functions(functions, gd, spos_c,
                               dtype=float, cut=False,
                               forces=True, lfbc=None):
    """Create `LocFuncs` object.

    From a list of splines, a grid-descriptor and a scaled position,
    create a `LocFuncs` object.  If this domain does not contribute to
    the localized functions, ``None`` is returned.

    ============= ======================== ===================================
    keyword       type
    ============= ======================== ===================================
    ``dtype``     ``float`` or ``complex`` Type of arrays to operate on.
    ``cut``       ``bool``                 Allow functions to cut boundaries
                                           when not periodic.
    ``forces``    ``bool``                 Calculate derivatives.
    ``lfbc``      `LocFuncBroadcaster`     Parallelization ...
    ============= ======================== ===================================
    """

    lfs = LocFuncs(functions, gd, spos_c,
                    dtype, cut, forces, lfbc)

    if len(lfs.box_b) > 0:
        return lfs
    else:
        # No boxes in this domain:
        return None


class LocFuncs:
    """Class to handle atomic-centered localized functions."""
    def __init__(self, functions, gd, spos_c,
                 dtype, cut, forces, lfbc):
        """Create `LocFuncs` object.

        Use `create_localized_functions()` to create this object."""

        # We assume that all functions have the same cut-off:
        rcut = functions[0].get_cutoff()

        box_b = gd.get_boxes(spos_c, rcut, cut)

        self.box_b = []
        self.sdisp_bc = npy.zeros((len(box_b), 3))
        b = 0
        for beg_c, end_c, sdisp_c in box_b:
            box = LocalizedFunctions(functions, beg_c, end_c,
                                     spos_c, sdisp_c, gd,
                                     dtype, forces, lfbc)
            self.box_b.append(box)
            self.sdisp_bc[b] = sdisp_c
            b += 1
        
        self.ni = 0
        for radial in functions:
            l = radial.get_angular_momentum_number()
            assert l <= 4, 'C-code only does l <= 4.'
            self.ni += 2 * l + 1

        self.dtype = dtype
        self.comm = gd.comm
        self.set_ranks([0], 0)
        self.phase_kb = None

    def set_ranks(self, ranks, root):
        """Set MPI-communicator and master CPU."""
        self.ranks = ranks
        self.root = root

    def set_phase_factors(self, k_kc):
        self.phase_kb = npy.exp(2j * pi * npy.inner(k_kc, self.sdisp_bc))
        
    def add(self, a_xg, coef_xi, k=None, communicate=False):
        """Add localized functions to extended arrays.

        Add the product of coef_xi and the localized functions to
        a_xg.  With Bloch boundary-condtions, k is used to
        index the phase-factors.  If communicate is True,
        coef_xi will be broadcasted from the root-CPU."""
        
        run(self.iadd(a_xg, coef_xi, k, communicate))
        
    def iadd(self, a_xg, coef_xi, k=None, communicate=False):
        """Iterator for adding localized functions to extended arrays."""
        if communicate:
            if len(self.ranks) == 1:
                # Nothing to do:
                yield None
            else:
                # Send coefficients to other ranks:
                if self.comm.rank == self.root:
                    requests = []
                    for rank in self.ranks:
                        if rank == self.root:
                            continue
                        request = self.comm.send(coef_xi, rank, 1329, False)
                        requests.append(request)
                    yield None
                    for request in requests:
                        self.comm.wait(request)
                else:
                    # Get coefficients from root:
                    shape = a_xg.shape[:-3] + (self.ni,)
                    coef_xi = npy.zeros(shape, self.dtype)
                    request = self.comm.receive(coef_xi, self.root,
                                                1329, False)

                    yield None
                    self.comm.wait(request)
        yield None

        if k is None or self.phase_kb is None:
            # No k-points:
            for box in self.box_b:
                box.add(coef_xi, a_xg)
        else:
            # K-points:
            for box, phase in zip(self.box_b, self.phase_kb[k]):
                box.add(coef_xi / phase, a_xg)
                
        yield None

    def integrate(self, a_xg, result_xi, k=None):
        run(self.iintegrate(a_xg, result_xi, k))
        
    def iintegrate(self, a_xg, result_xi, k=None):
        """Iterator for projecting extended arrays onto localized functions."""
        shape = a_xg.shape[:-3] + (self.ni,)
        tmp_xi = npy.zeros(shape, self.dtype)

        if result_xi is None:
            result_xi = npy.zeros(shape, self.dtype)
            
        isum = self.isum(result_xi)
        isum.next()
        yield None
                    
        if k is None or self.phase_kb is None:
            # No k-points:
            for box in self.box_b:
                box.integrate(a_xg, tmp_xi)
                result_xi += tmp_xi                
        else:
            # K-points:
            for box, phase in zip(self.box_b, self.phase_kb[k]):
                box.integrate(a_xg, tmp_xi)
                result_xi += phase * tmp_xi


        isum.next()
        yield None
        isum.next()
        yield None

    def sum(self, a_x, broadcast=False):
        """Sum up array.

        The default behavior is to let the owner-node return the
        result in a_x.  With broadcast=True, all nodes will return the
        result in a_x."""
        
        run(self.isum(a_x, broadcast))
        
    def isum(self, a_x, broadcast=False):
        """Iterator for adding arrays.

        There are three steps:

        1. Root-node starts receiving.
        2. Non-root nodes start sending.
        3. Wait.  Then root does sum.

        If broadcast is True, there will be two more steps:

        4. Root-node sends and non-root nodes receive.
        5. Wait.
        """

        ndomains = len(self.ranks)
        if ndomains > 1:
            if self.comm.rank == self.root:
                requests = []
                a_dx = npy.empty((ndomains - 1,) + a_x.shape, self.dtype)
                d = 0
                for rank in self.ranks:
                    if rank == self.root:
                        continue
                    request = self.comm.receive(a_dx[d:d + 1], rank,
                                                1330, False)
                    requests.append(request)
                    d += 1

                yield None
                yield None

                for request in requests:
                    self.comm.wait(request)

                a_x += a_dx.sum(0)

                if broadcast:
                    yield None
                    requests = []
                    for rank in self.ranks:
                        if rank == self.root:
                            continue
                        request = self.comm.send(a_x, rank, 1331, False)
                        requests.append(request)
                    yield None
                    for request in requests:
                        self.comm.wait(request)
            else:
                yield None
                request = self.comm.send(a_x, self.root, 1330, False)
                yield None
                self.comm.wait(request)
                if broadcast:
                    yield None
                    request = self.comm.receive(a_x, self.root, 1331, False)
                    yield None
                    self.comm.wait(request)

        else:
            yield None
            yield None
            if broadcast:
                yield None
                yield None

        yield None
            
    def derivative(self, a_xg, result_xic, k=None):
        """Calculate derivatives of localized integrals.

        Return the *x*- *y*- and *z*-derivatives of the integral of
        extended arrays times localized functions in ``result_xi``.
        Correct phase-factors are used if the **k**-point index ``k``
        is not ``None`` (Block boundary-condtions)."""
        
        run(self.iderivative(a_xg, result_xic, k))
        
    def iderivative(self, a_xg, result_xic, k=None):
        shape = a_xg.shape[:-3] + (self.ni, 3)
        tmp_xic = npy.zeros(shape, self.dtype)
        if result_xic is None:
            result_xic = npy.zeros(shape, self.dtype)
            
        isum = self.isum(result_xic)
        isum.next()
        yield None

        if k is None or self.phase_kb is None:
            # No k-points:
            for box in self.box_b:
                box.derivative(a_xg, tmp_xic)
                result_xic += tmp_xic                
        else:
            # K-points:
            for box, phase in zip(self.box_b, self.phase_kb[k]):
                box.derivative(a_xg, tmp_xic)
                result_xic += phase * tmp_xic
               
        isum.next()
        yield None
        isum.next()
        yield None

    def add_density(self, n_G, f_i):
        """Add atomic electron density to extended density array.

        Special method for adding the atomic electron density
        calculated from atomic orbitals and occupation numbers
        ``f_i``."""
        for box in self.box_b:
            box.add_density(n_G, f_i)

    def add_density2(self, n_G, D_p):
        """Add atomic electron density to extended density array.
        
        Special method for adding the atomic electron density
        calculated from all cross products of atomic orbitals
        weighted using the density matrix ``D_p``.

        The method returns the integral of the atomic electron density
        """
        assert self.comm.size == 1
        I = 0.0
        for box in self.box_b:
            I += box.add_density2(n_G, D_p)
        return I

    def norm(self):
        """Calculate norm of localized functions."""

        I_i = npy.zeros(self.ni)
        for box in self.box_b:
            box.norm(I_i)
        self.sum(I_i, broadcast=True)
        return I_i
        
    def normalize(self, I0):
        """Normalize localized functions.
        
        The integral of the first function (shperically symmetric, l =
        0) is normalized to the value ``I0`` and the following
        functions (l > 0) are adjusted so that they integrate to
        zero."""

        I_i = self.norm()
        for box in self.box_b:
            box.normalize(I0, I_i)

        
class LocalizedFunctionsWrapper:
    """Python wrapper class for C-extension: ``LocalizedFunctions``.

    This class is used for construction of the C-object and for adding
    type-checking to the C-methods."""
    
    def __init__(self, functions, beg_c, end_c, spos_c, sdisp_c, gd,
                 dtype, forces, locfuncbcaster):
        """Construct a ``LocalizedFunctions`` C-object.

        Evaluate function values from a list of splines
        (``functions``) inside a box between grid points ``beg_c``
        (included) to ``end_c`` (not included).  The functions are
        centered at the scaled position ``spos_c`` displaced by
        ``sdisp_c`` (in units of lattice vectors), and ``gd`` is the
        grid-descriptor.

        Derivatives are calculated when ``forces=True``."""

        assert dtype in [float, complex]

        # Who evaluates the function values?
        if locfuncbcaster is None:
            # I do!
            compute = True
        else:
            # One of the CPU's in the k-point communicator does it,
            # and will later broadcast to the others:
            compute = locfuncbcaster.next()
            
        size_c = end_c - beg_c
        corner_c = beg_c - gd.beg_c
        pos_c = (beg_c - (spos_c - sdisp_c) * gd.N_c) * gd.h_c

        self.lfs = _gpaw.LocalizedFunctions(
            [function.spline for function in functions],
            size_c, gd.n_c, corner_c, gd.h_c, pos_c,
            dtype == float, forces, compute)
        
        if locfuncbcaster is not None:
            locfuncbcaster.add(self.lfs)

        self.ni = 0   # number of functions
        for function in functions:
            l = function.get_angular_momentum_number()
            self.ni += 2 * l + 1; 

        self.shape = tuple(gd.n_c)
        self.dtype = dtype
        self.forces = forces
        
    def integrate(self, a_xg, result_xi):
        """Calculate integrals of arrays times localized functions.

        Return the integral of extended arrays times localized
        functions in ``result_xi``."""
        
        assert is_contiguous(a_xg, self.dtype)
        assert is_contiguous(result_xi, self.dtype)
        assert a_xg.shape[:-3] == result_xi.shape[:-1]
        assert a_xg.shape[-3:] == self.shape
        assert result_xi.shape[-1] == self.ni
        self.lfs.integrate(a_xg, result_xi)

    def derivative(self, a_xg, result_xic):
        """Calculate x-, y-, z-derivatives of localized integrals.

        Return the *x*- *y*- and *z*-derivatives of the integral of
        extended arrays times localized functions in
        ``result_xic``."""

        assert self.forces
        assert is_contiguous(a_xg, self.dtype)
        assert is_contiguous(result_xic, self.dtype)
        assert a_xg.shape[:-3] == result_xic.shape[:-2]
        assert a_xg.shape[-3:] == self.shape
        assert result_xic.shape[-2:] == (self.ni, 3)
        self.lfs.derivative(a_xg, result_xic)

    def add(self, coef_xi, a_xg):
        """Add localized functions to extended arrays.

        Add the product of ``coef_xi`` and the localized functions to
        ``a_xg``."""
        
        assert is_contiguous(a_xg, self.dtype)
        assert is_contiguous(coef_xi, self.dtype)
        assert a_xg.shape[:-3] == coef_xi.shape[:-1]
        assert a_xg.shape[-3:] == self.shape
        assert coef_xi.shape[-1] == self.ni
        self.lfs.add(coef_xi, a_xg)

    def add_density(self, n_G, f_i):
        """Add atomic electron density to extended density array.

        Special method for adding the atomic electron density
        calculated from atomic orbitals and occupation numbers
        ``f_i``."""
        
        assert is_contiguous(n_G, float)
        assert is_contiguous(f_i, float)
        assert n_G.shape == self.shape
        assert f_i.shape == (self.ni,)
        self.lfs.add_density(n_G, f_i)

    def add_density2(self, n_G, D_p):
        """Add atomic electron density to extended density array.
        
        Special method for adding the atomic electron density
        calculated from all cross products of atomic orbitals
        weighted using the density matrix ``D_p``.

        The method returns the integral of the atomic electron density
        """
        
        assert is_contiguous(n_G, float)
        assert is_contiguous(D_p, float)
        assert n_G.shape == self.shape
        assert D_p.shape == (self.ni * (self.ni + 1) / 2,)
        return self.lfs.add_density2(n_G, D_p)

    def norm(self, I_i):
        """Integrate functions."""
        assert is_contiguous(I_i, float)
        assert I_i.shape == (self.ni,)
        return self.lfs.norm(I_i)

    def normalize(self, I0, I_i):
        """Normalize functions."""
        assert is_contiguous(I_i, float)
        assert I_i.shape == (self.ni,)
        return self.lfs.normalize(I0, I_i)

if debug:
    # Add type and sanity checks:
    LocalizedFunctions = LocalizedFunctionsWrapper
else:
    # Just use the bare C-object for efficiency:
    def LocalizedFunctions(functions, beg_c, end_c, spos_c, sdisp_c, gd,
                           dtype, forces, locfuncbcaster):
        return LocalizedFunctionsWrapper(functions, beg_c, end_c, spos_c,
                                         sdisp_c, gd,
                                         dtype, forces, locfuncbcaster).lfs


class LocFuncBroadcaster:
    """..."""
    def __init__(self, comm):
        if debug:
            comm = comm.comm
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.reset()

    def reset(self):
        self.lfs = []
        self.root = 0

    def next(self):
        compute = (self.root == self.rank)
        self.root = (self.root + 1) % self.size
        return compute

    def add(self, lf):
        self.lfs.append(lf)
    
    def broadcast(self):
        if self.size > 1:
            for root, lf in enumerate(self.lfs):
                lf.broadcast(self.comm, root % self.size)
        self.reset()
