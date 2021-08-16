import numpy as np
from ase.units import Hartree
from gpaw.directmin.lcao.tools import d_matrix


class KSLCAO:
    """
    this class described derivartve of the KS functional
    w.r.t rotation parameters
    """

    def __init__(self, wfs, dens, ham, **kwargs):
        self.name = 'Zero'
        self.n_kps = wfs.kd.nibzkpts
        self.dtype = wfs.dtype
        self.nvalence = wfs.nvalence

    def get_gradients(self, h_mm, c_nm, f_n,
                      evec, evals, kpt, wfs, timer, matrix_exp,
                      repr_name, ind_up, occupied_only=False):

        timer.start('Construct Gradient Matrix')
        if repr_name in ['sparse', 'u_invar'] and not matrix_exp == 'egdecomp':
            if repr_name == 'u_invar':
                uind1 = np.unique(ind_up[0])
                uind2 = np.unique(ind_up[1])
                h_mm = c_nm[uind1].conj() @ h_mm.conj() @ c_nm[uind2].T
                h_mm = h_mm.ravel()
            else:
                occ = f_n > 1.0e-10
                n_occ = sum(occ)
                ch_nm = c_nm[:n_occ].conj() @ h_mm.conj()
                h1_mm = ch_nm @ c_nm[:n_occ].T
                h2_mm = ch_nm @ c_nm[n_occ:].T
                indl_oo = np.tril_indices(h1_mm.shape[0])
                h1_mm[indl_oo] = np.inf
                h_mm = np.concatenate((h1_mm, h2_mm), axis=1)
                h_mm = h_mm.ravel()
                h_mm = h_mm[h_mm != np.inf]
           
            error = 0.0
            ones_mat = np.ones(shape=(len(f_n), len(f_n)))
            anti_occ = f_n * ones_mat - f_n[:, np.newaxis] * ones_mat
            anti_occ = anti_occ[ind_up]
            grad = np.ascontiguousarray(anti_occ * h_mm)
        else:

            hc_mn = h_mm.conj() @ c_nm.T
            h_mm = c_nm.conj() @ hc_mn

            # let's also calculate residual here.
            # it's extra calculation though, maybe it's better to use
            # norm of grad as convergence criteria..
            
            timer.start('Residual')
            occ = f_n > 1.0e-10
            hc_mn = hc_mn[:, occ] - \
                kpt.S_MM.conj() @ c_nm[occ].T @ h_mm[occ][:, occ]
            norm = sum(hc_mn.conj() * hc_mn * kpt.f_n[occ])
            error = sum(norm.real) * Hartree ** 2 / self.nvalence
            del hc_mn, norm
            timer.stop('Residual')

            # continue with gradients
            h_mm = f_n * h_mm - f_n[:, np.newaxis] * h_mm
           
            if matrix_exp == 'egdecomp':
                timer.start('Use Eigendecomposition')
                grad = evec.T.conj() @ h_mm @ evec
                grad = grad * d_matrix(evals)
                grad = evec @ grad @ evec.T.conj()
                for i in range(grad.shape[0]):
                    grad[i][i] *= 0.5
                timer.stop('Use Eigendecomposition')
            else:
                grad = np.ascontiguousarray(h_mm)
           
            if repr_name in ['sparse', 'u_invar']:
                grad = grad[ind_up]

        if self.dtype == float:
            grad = grad.real
        
        timer.stop('Construct Gradient Matrix')

        return 2.0 * grad, error
