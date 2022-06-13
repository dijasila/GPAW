from ase.units import Ha
import os
import gpaw.mpi as mpi
import numpy as np
from gpaw.xc.fxc import KernelWave
from ase.io.aff import affopen


def actually_calculate_kernel(*, calc, xcflags, q_empty, tag, ecut_max, fd,
                              timer, Eg, wd):
    kd = calc.wfs.kd
    bzq_qc = kd.get_bz_q_points(first=True)
    U_scc = kd.symmetry.op_scc
    ibzq_qc = kd.get_ibz_q_points(bzq_qc, U_scc)[0]

    l_l = np.array([1.0])

    if xcflags.linear_kernel:
        l_l = None
        omega_w = None
    elif not xcflags.dyn_kernel:
        omega_w = None
    else:
        omega_w = wd.omega_w

    kernel = KernelWave(
        l_l=l_l,
        omega_w=omega_w,
        calc=calc,
        xc=xcflags.xc,
        ibzq_qc=ibzq_qc,
        fd=fd,
        q_empty=q_empty,
        Eg=Eg,
        ecut=ecut_max,
        tag=tag,
        timer=timer)

    kernel.calculate_fhxc()


def calculate_kernel(*, ecut, xcflags, calc, nG, ns, iq, cut_G=None,
                     timer, fd, wd, Eg):
    xc = xcflags.xc
    tag = calc.atoms.get_chemical_formula(mode='hill')

    ecut_max = ecut * Ha  # XXX very ugly this
    q_empty = None

    filename = 'fhxc_%s_%s_%s_%s.ulm' % (tag, xc, ecut_max, iq)

    if not os.path.isfile(filename):
        q_empty = iq

    if xc != 'RPA':
        if q_empty is not None:
            actually_calculate_kernel(q_empty=q_empty, tag=tag,
                                      xcflags=xcflags, Eg=Eg,
                                      ecut_max=ecut_max, calc=calc,
                                      fd=fd, timer=timer,
                                      wd=wd)
            # (This creates the ulm file above.  Probably.)

        mpi.world.barrier()

        if xcflags.spin_kernel:
            with affopen(filename) as r:
                fv = r.fhxc_sGsG

            if cut_G is not None:
                cut_sG = np.tile(cut_G, ns)
                cut_sG[len(cut_G):] += len(fv) // ns
                fv = fv.take(cut_sG, 0).take(cut_sG, 1)

        else:
            if xc == 'RPA':
                fv = np.eye(nG)
            elif xc == 'range_RPA':
                raise NotImplementedError
#                    fv = np.exp(-0.25 * (G_G * self.range_rc) ** 2.0)

            elif xcflags.linear_kernel:
                with affopen(filename) as r:
                    fv = r.fhxc_sGsG

                if cut_G is not None:
                    fv = fv.take(cut_G, 0).take(cut_G, 1)

            elif not xcflags.dyn_kernel:
                # static kernel which does not scale with lambda
                with affopen(filename) as r:
                    fv = r.fhxc_lGG

                if cut_G is not None:
                    fv = fv.take(cut_G, 1).take(cut_G, 2)

            else:  # dynamical kernel
                with affopen(filename) as r:
                    fv = r.fhxc_lwGG

                if cut_G is not None:
                    fv = fv.take(cut_G, 2).take(cut_G, 3)
    else:
        fv = np.eye(nG)

    return fv
