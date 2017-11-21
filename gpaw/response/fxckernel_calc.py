from ase.units import Ha
import os
import gpaw.mpi as mpi
import numpy as np
from gpaw.xc.fxc import KernelDens
from gpaw.xc.fxc import KernelWave
from ase.io.aff import affopen

def calculate_kernel(self, nG, ns, iq, cut_G=None):
        self.unit_cells = self.calc.wfs.kd.N_c
        self.tag = self.calc.atoms.get_chemical_formula(mode='hill')

        if self.av_scheme is not None:
            self.tag += '_' + self.av_scheme + '_nspins' + str(self.nspins)

        kd = self.calc.wfs.kd
        self.bzq_qc = kd.get_bz_q_points(first=True)
        U_scc = kd.symmetry.op_scc
        self.ibzq_qc = kd.get_ibz_q_points(self.bzq_qc, U_scc)[0]

        ecut = self.ecut * Ha
        if isinstance(ecut, (float, int)):
            self.ecut_max = ecut
        else:
            self.ecut_max = max(ecut)

        q_empty = None

        if not os.path.isfile('fhxc_%s_%s_%s_%s.ulm'
                              % (self.tag, self.xc,
                                 self.ecut_max, iq)):
            q_empty = iq

        if self.xc not in ('RPA'):
            if q_empty is not None:
                if self.av_scheme == 'wavevector':

                    self.l_l = np.array([1.0])
                    if self.linear_kernel:
                        kernel = KernelWave(self.calc,
                                             self.xc,
                                             self.ibzq_qc,
                                             self.fd,
                                             None,
                                             q_empty,
                                             None,
                                             self.Eg,
                                             self.ecut_max,
                                             self.tag,
                                             self.timer)

                    elif not self.dyn_kernel:
                        kernel = KernelWave(self.calc,
                                             self.xc,
                                             self.ibzq_qc,
                                             self.fd,
                                             self.l_l,
                                             q_empty,
                                             None,
                                             self.Eg,
                                             self.ecut_max,
                                             self.tag,
                                             self.timer)

                    else:
                        kernel = KernelWave(self.calc,
                                             self.xc,
                                             self.ibzq_qc,
                                             self.fd,
                                             self.l_l,
                                             q_empty,
                                             self.omega_w,
                                             self.Eg,
                                             self.ecut_max,
                                             self.tag,
                                             self.timer)
                else:
                    kernel = KernelDens(self.calc,
                                        self.xc,
                                        self.ibzq_qc,
                                        self.fd,
                                        self.unit_cells,
                                        self.density_cut,
                                        self.ecut_max,
                                        self.tag,
                                        self.timer)

                kernel.calculate_fhxc()
                del kernel
            #else:
            #    print('%s kernel already calculated' % self.xc, file=self.fd)
            #    print(file=self.fd)

            mpi.world.barrier()

            #for iq in range(len(self.ibzq_qc)):
            #    if np.array_equal(self.ibzq_qc[iq], q_c):
            #        thisq = iq
            #        break

#            cut_G = G2G#None

            if self.spin_kernel:
                r = affopen('fhxc_%s_%s_%s_%s.ulm' %
                            (self.tag, self.xc, self.ecut_max, iq))
                fv = r.fhxc_sGsG

                if cut_G is not None:
                    cut_sG = np.tile(cut_G, ns)
                    cut_sG[len(cut_G):] += len(fv) // ns
                    fv = fv.take(cut_sG, 0).take(cut_sG, 1)

            # the spin-polarized kernel constructed from wavevector average
            # is already multiplied by |q+G| |q+G'|/4pi, and doesn't require
            # special treatment of the head and wings.  However not true for
            # density average:

                if self.av_scheme == 'density':
                    for s1 in range(ns):
                        for s2 in range(ns):
                            m1 = s1 * nG
                            n1 = (s1 + 1) * nG
                            m2 = s2 * nG
                            n2 = (s2 + 1) * nG
                            fv[m1:n1, m2:n2] *= (G_G * G_G[:, np.newaxis] /
                                                 (4 * np.pi))

                            if np.prod(self.unit_cells) > 1 and pd.kd.gamma:
                                m1 = s1 * nG
                                n1 = (s1 + 1) * nG
                                m2 = s2 * nG
                                n2 = (s2 + 1) * nG
                                fv[m1, m2:n2] = 0.0
                                fv[m1:n1, m2] = 0.0
                                fv[m1, m2] = 1.0
                                
            else:
                if self.xc == 'RPA':

                    fv = np.eye(nG)

                elif self.xc == 'range_RPA':

                    fv = np.exp(-0.25 * (G_G * self.range_rc) ** 2.0)

                elif self.linear_kernel:
                    r = affopen('fhxc_%s_%s_%s_%s.ulm' %
                                (self.tag, self.xc, self.ecut_max, iq))
                    fv = r.fhxc_sGsG

                    if cut_G is not None:
                        fv = fv.take(cut_G, 0).take(cut_G, 1)

                elif not self.dyn_kernel:
                # static kernel which does not scale with lambda

                    r = affopen('fhxc_%s_%s_%s_%s.ulm' %
                                (self.tag, self.xc, self.ecut_max, iq))
                    fv = r.fhxc_lGG

                    if cut_G is not None:
                        fv = fv.take(cut_G, 1).take(cut_G, 2)

                else:  # dynamical kernel
                    r = affopen('fhxc_%s_%s_%s_%s.ulm' %
                                (self.tag, self.xc, self.ecut_max, iq))
                    fv = r.fhxc_lwGG

                    if cut_G is not None:
                        fv = fv.take(cut_G, 2).take(cut_G, 3)

        else:
            fv =  np.eye(nG)

        return fv

def set_flags(self):
        """ Based on chosen fxc and av. scheme set up true-false flags """

        if self.xc not in ('RPA',
                           'range_RPA',  # range separated RPA a la Bruneval
                           'rALDA',      # renormalized kernels
                           'rAPBE',
                           'range_rALDA',
                           'rALDAns',    # no spin (ns)
                           'rAPBEns',
                           'rALDAc',     # rALDA + correlation
                           'CP',         # Constantin Pitarke
                           'CP_dyn',     # Dynamical form of CP
                           'CDOP',       # Corradini et al
                           'CDOPs',      # CDOP without local term
                           'JGMs',       # simplified jellium-with-gap kernel
                           'JGMsx',      # simplified jellium-with-gap kernel,
                                         # constructed with exchange part only
                                         # so that it scales linearly with l
                           'ALDA'        # standard ALDA
                           ):
            raise RuntimeError('%s kernel not recognized' % self.xc)

        if (self.xc == 'rALDA' or self.xc == 'rAPBE' or self.xc == 'ALDA'):

            if self.av_scheme is None:
                self.av_scheme = 'density'
                # Two-point scheme default for rALDA and rAPBE

            self.spin_kernel = True
            # rALDA/rAPBE are the only kernels which have spin-dependent forms

        else:
            self.spin_kernel = False

        if self.av_scheme == 'density':
            assert (self.xc == 'rALDA' or self.xc == 'rAPBE' or self.xc == 'ALDA'
                    ), ('Two-point density average ' +
                        'only implemented for rALDA and rAPBE')

        elif self.xc not in ('RPA', 'range_RPA'):
            self.av_scheme = 'wavevector'
        else:
            self.av_scheme = None

        if self.xc in ('rALDAns', 'rAPBEns', 'range_RPA', 'JGMsx',
                       'RPA', 'rALDA', 'rAPBE', 'range_rALDA','ALDA'):
            self.linear_kernel = True  # Scales linearly with coupling constant
        else:
            self.linear_kernel = False

        if self.xc == 'CP_dyn':
            self.dyn_kernel = True
        else:
            self.dyn_kernel = False

        if self.xc == 'JGMs' or self.xc == 'JGMsx':
            assert (self.Eg is not None), 'JGMs kernel requires a band gap!'
            self.Eg /= Ha # Convert from eV
        else:
            self.Eg = None
