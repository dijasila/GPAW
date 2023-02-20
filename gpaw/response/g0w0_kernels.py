from ase.units import Ha
import gpaw.mpi as mpi
import numpy as np
from gpaw.xc.fxc import KernelWave, XCFlags, FXCCache

from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.pw.descriptor import PWMapping


class G0W0Kernel:
    def __init__(self, xc, context, **kwargs):
        self.xc = xc
        self.context = context
        self.xcflags = XCFlags(xc)
        self._kwargs = kwargs

    def calculate(self, qpd):
        return calculate_kernel(
            qpd=qpd,
            xcflags=self.xcflags,
            context=self.context,
            **self._kwargs)


def actually_calculate_kernel(*, gs, qd, xcflags, q_empty, cache, ecut_max,
                              context):
    ibzq_qc = qd.ibzk_kc

    kernel = KernelWave(
        gs=gs,
        xc=xcflags.xc,
        ibzq_qc=ibzq_qc,
        q_empty=q_empty,
        ecut=ecut_max,
        cache=cache,
        context=context)

    kernel.calculate_fhxc()


def calculate_kernel(*, ecut, xcflags, gs, qd, ns, qpd, context):
    xc = xcflags.xc

    # Get iq
    ibzq_qc = qd.ibzk_kc
    iq = np.argmin(np.linalg.norm(ibzq_qc - qpd.q_c[np.newaxis], axis=1))
    assert np.allclose(ibzq_qc[iq], qpd.q_c)

    ecut_max = ecut * Ha  # XXX very ugly this
    q_empty = None

    # If we want a reduced plane-wave description, create qpd mapping
    if qpd.ecut < ecut:
        # Recreate nonreduced plane-wave description corresponding to ecut_max
        qpdnr = SingleQPWDescriptor.from_q(qpd.q_c, ecut, qpd.gd,
                                           gammacentered=qpd.gammacentered)
        pw_map = PWMapping(qpd, qpdnr)
        G2_G1 = pw_map.G2_G1
    else:
        G2_G1 = None

    cache = FXCCache(tag=gs.atoms.get_chemical_formula(mode='hill'),
                     xc=xc, ecut=ecut_max)
    handle = cache.handle(iq)

    if not handle.exists():
        q_empty = iq

    if xc == 'RPA':
        return np.eye(qpd.ngmax)

    assert xcflags.spin_kernel

    if q_empty is not None:
        actually_calculate_kernel(q_empty=q_empty, qd=qd,
                                  cache=cache,
                                  xcflags=xcflags,
                                  ecut_max=ecut_max, gs=gs,
                                  context=context)

    mpi.world.barrier()

    fv = handle.read()

    if G2_G1 is not None:
        cut_sG = np.tile(G2_G1, ns)
        cut_sG[len(G2_G1):] += len(fv) // ns
        fv = fv.take(cut_sG, 0).take(cut_sG, 1)

    return fv
