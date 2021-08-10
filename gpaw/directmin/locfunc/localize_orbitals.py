from gpaw.directmin.locfunc.dirmin import DirectMinLocalize
from gpaw.directmin.locfunc.er import ERlocalization as ERL
from gpaw.directmin.odd.fdpw.pz import PzCorrections as PZpwfd
from gpaw.directmin.odd.fdpw.zero import ZeroCorrections as KSpwfd
from gpaw.directmin.odd.lcao import PzCorrectionsLcao as PZlcao
from gpaw.pipekmezey.pipek_mezey_wannier import PipekMezey
from gpaw.pipekmezey.wannier_basic import WannierLocalization
import numpy as np


def localize_orbitals(wfs, dens, ham, log, localizationtype):

    io = localizationtype
    if io is None:
        return
    elif io == 'PM':
        tol = 1.0e-6
    else:
        tol = 1.0e-10
    locnames = io.split('-')

    log("Initial Localization: ...", flush=True)
    wfs.timer.start('Initial Localization')
    for name in locnames:
        if name == 'ER':
            if wfs.mode == 'lcao':
                log('Edmiston-Ruedenberg loc. is not supproted in LCAO',
                    flush=True)
                continue
            log('Edmiston-Ruedenberg localization started',
                flush=True)
            dm = DirectMinLocalize(
                ERL(wfs, dens, ham), wfs,
                maxiter=200, g_tol=5.0e-5, randval=0.1)
            dm.run(wfs, dens)
            log('Edmiston-Ruedenberg localization finished',
                flush=True)
            del dm
        elif name == 'PZ':
            log('Perdew-Zunger localization started',
                flush=True)
            if wfs.mode == 'lcao':
                PZC = PZlcao
            else:
                PZC = PZpwfd
            dm = DirectMinLocalize(
                PZC(wfs, dens, ham), wfs,
                maxiter=200, g_tol=5.0e-4, randval=0.1)
            dm.run(wfs, dens, log)
            log('Perdew-Zunger localization finished',
                flush=True)
        elif name == 'KS':
            print('============')
            print('I am here!\n')
            print('============', flush=True)
            log('ETDM minimization using occupied and virtual orbitals',
                flush=True)
            if wfs.mode == 'lcao':
                raise NotImplementedError
            else:
                KS = KSpwfd
            dm = DirectMinLocalize(
                KS(wfs, dens, ham), wfs,
                maxiter=200, g_tol=5.0e-4, randval=0)
            dm.run(wfs, dens, log, ham=ham)
            log('ETDM minimization finished', flush=True)
        else:
            for kpt in wfs.kpt_u:
                if sum(kpt.f_n) < 1.0e-3:
                    continue
                if name == 'PM':
                    log('Pipek-Mezey localization started',
                        flush=True)
                    lf_obj = PipekMezey(
                        wfs=wfs, spin=kpt.s, dtype=wfs.dtype)
                    lf_obj.localize(tolerance=tol)
                    log('Pipek-Mezey localization finished',
                        flush=True)
                    U = np.ascontiguousarray(
                        lf_obj.W_k[kpt.q].T)
                elif name == 'FB':
                    log('Foster-Boys localization started',
                        flush=True)
                    lf_obj = WannierLocalization(
                        wfs=wfs, spin=kpt.s)
                    lf_obj.localize(tolerance=tol)
                    log('Foster-Boys localization finsihed',
                        flush=True)
                    U = np.ascontiguousarray(
                        lf_obj.U_kww[kpt.q].T)
                    if wfs.dtype == float:
                        U = U.real
                else:
                    raise ValueError('Check localization type.')
                wfs.gd.comm.broadcast(U, 0)
                dim = U.shape[0]
                if wfs.mode == 'fd':
                    kpt.psit_nG[:dim] = np.einsum(
                        'ij,jkml->ikml', U, kpt.psit_nG[:dim])
                elif wfs.mode == 'pw':
                    kpt.psit_nG[:dim] = U @ kpt.psit_nG[:dim]
                else:
                    kpt.C_nM[:dim] = U @ kpt.C_nM[:dim]

                del lf_obj

    wfs.timer.stop('Initial Localization')
    log("Done", flush=True)
