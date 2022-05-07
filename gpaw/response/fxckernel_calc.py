from ase.units import Ha
import os
import gpaw.mpi as mpi
import numpy as np
from gpaw.xc.fxc import KernelWave
from ase.io.aff import affopen


def actually_calculate_kernel(self, q_empty, tag, ecut_max):
    kd = self.calc.wfs.kd
    bzq_qc = kd.get_bz_q_points(first=True)
    U_scc = kd.symmetry.op_scc
    ibzq_qc = kd.get_ibz_q_points(bzq_qc, U_scc)[0]

    l_l = np.array([1.0])

    if self.linear_kernel:
        l_l = None
        omega_w = None
    elif not self.dyn_kernel:
        omega_w = None
    else:
        omega_w = self.wd.omega_w

    kernel = KernelWave(
        l_l=l_l,
        omega_w=omega_w,
        calc=self.calc,
        xc=self.xc,
        ibzq_qc=ibzq_qc,
        fd=self.fd,
        q_empty=q_empty,
        Eg=self.Eg,
        ecut=ecut_max,
        tag=tag,
        timer=self.timer)

    kernel.calculate_fhxc()


def calculate_kernel(self, nG, ns, iq, cut_G=None):
    tag = self.calc.atoms.get_chemical_formula(mode='hill')

    if self.av_scheme is not None:
        tag += '_' + self.av_scheme + '_nspins' + str(self.nspins)

    ecut = self.ecut * Ha
    if isinstance(ecut, (float, int)):
        ecut_max = ecut
    else:
        ecut_max = max(ecut)

    q_empty = None

    filename = 'fhxc_%s_%s_%s_%s.ulm' % (tag, self.xc, ecut_max, iq)

    if not os.path.isfile(filename):
        q_empty = iq

    if self.xc not in ('RPA'):
        if q_empty is not None:
            actually_calculate_kernel(self, q_empty, tag, ecut_max)
            # (This creates the ulm file above.  Probably.)

        mpi.world.barrier()

        if self.spin_kernel:
            with affopen(filename) as r:
                fv = r.fhxc_sGsG

            if cut_G is not None:
                cut_sG = np.tile(cut_G, ns)
                cut_sG[len(cut_G):] += len(fv) // ns
                fv = fv.take(cut_sG, 0).take(cut_sG, 1)

        else:
            if self.xc == 'RPA':
                fv = np.eye(nG)
            elif self.xc == 'range_RPA':
                raise NotImplementedError
#                    fv = np.exp(-0.25 * (G_G * self.range_rc) ** 2.0)

            elif self.linear_kernel:
                with affopen(filename) as r:
                    fv = r.fhxc_sGsG

                if cut_G is not None:
                    fv = fv.take(cut_G, 0).take(cut_G, 1)

            elif not self.dyn_kernel:
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
