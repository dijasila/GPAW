from ase.units import Ha
import os
import gpaw.mpi as mpi
import numpy as np
from gpaw.xc.fxc import KernelWave, XCFlags
from ase.io.aff import affopen

from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.pw.descriptor import PWMapping


class G0W0Kernel:
    def __init__(self, xc, context, **kwargs):
        self.xc = xc
        self.context = context
        self.xcflags = XCFlags(xc)
        self._kwargs = kwargs

    def calculate(self, pd):
        return calculate_kernel(
            pd=pd,
            xcflags=self.xcflags,
            context=self.context,
            **self._kwargs)


def actually_calculate_kernel(*, gs, qd, xcflags, q_empty, tag, ecut_max,
                              context):
    ibzq_qc = qd.ibzk_kc

    l_l = np.array([1.0])

    if xcflags.linear_kernel:
        l_l = None

    kernel = KernelWave(
        l_l=l_l,
        gs=gs,
        xc=xcflags.xc,
        ibzq_qc=ibzq_qc,
        q_empty=q_empty,
        ecut=ecut_max,
        tag=tag,
        context=context)

    kernel.calculate_fhxc()


def calculate_kernel(*, ecut, xcflags, gs, qd, ns, pd, context):
    xc = xcflags.xc
    tag = gs.atoms.get_chemical_formula(mode='hill')

    # Get iq
    ibzq_qc = qd.ibzk_kc
    iq = np.argmin(np.linalg.norm(ibzq_qc - pd.q_c[np.newaxis], axis=1))
    assert np.allclose(ibzq_qc[iq], pd.q_c)

    ecut_max = ecut * Ha  # XXX very ugly this
    q_empty = None

    # If we want a reduced plane-wave description, create pd mapping
    if pd.ecut < ecut:
        # Recreate nonreduced plane-wave description corresponding to ecut_max
        pdnr = SingleQPWDescriptor.from_q(pd.q_c, ecut, pd.gd,
                                          gammacentered=pd.gammacentered)
        pw_map = PWMapping(pd, pdnr)
        G2_G1 = pw_map.G2_G1
    else:
        G2_G1 = None

    filename = 'fhxc_%s_%s_%s_%s.ulm' % (tag, xc, ecut_max, iq)

    if not os.path.isfile(filename):
        q_empty = iq

    if xc != 'RPA':
        if q_empty is not None:
            actually_calculate_kernel(q_empty=q_empty, qd=qd, tag=tag,
                                      xcflags=xcflags,
                                      ecut_max=ecut_max, gs=gs,
                                      context=context)
            # (This creates the ulm file above.  Probably.)

        mpi.world.barrier()

        if xcflags.spin_kernel:
            with affopen(filename) as r:
                fv = r.fhxc_sGsG

            if G2_G1 is not None:
                cut_sG = np.tile(G2_G1, ns)
                cut_sG[len(G2_G1):] += len(fv) // ns
                fv = fv.take(cut_sG, 0).take(cut_sG, 1)

        else:
            if xc == 'RPA':
                fv = np.eye(pd.ngmax)
            elif xc == 'range_RPA':
                raise NotImplementedError
#                    fv = np.exp(-0.25 * (G_G * self.range_rc) ** 2.0)

            elif xcflags.linear_kernel:
                with affopen(filename) as r:
                    fv = r.fhxc_sGsG

                if G2_G1 is not None:
                    fv = fv.take(G2_G1, 0).take(G2_G1, 1)

            else:
                # static kernel which does not scale with lambda
                with affopen(filename) as r:
                    fv = r.fhxc_lGG

                if G2_G1 is not None:
                    fv = fv.take(G2_G1, 1).take(G2_G1, 2)

    else:
        fv = np.eye(pd.ngmax)

    return fv
