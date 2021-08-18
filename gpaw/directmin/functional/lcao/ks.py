import numpy as np
from ase.units import Hartree
from gpaw.directmin.lcao.tools import d_matrix


class KSLCAO:

    """
    this class described derivartve of the KS functional
    w.r.t rotation parameters
    """

    def __init__(self, wfs, dens, ham, **kwargs):
        self.name = 'ks'
        self.n_kps = wfs.kd.nibzkpts
        self.dtype = wfs.dtype
        self.nvalence = wfs.nvalence

    def get_gradients(self, h_mm, c_nm, f_n,
                      evec, evals, kpt, wfs, timer, matrix_exp,
                      repr_name, ind_up, occupied_only=False):

        with timer('Construct Gradient Matrix'):
            use_egdecomp = matrix_exp == 'egdecomp'
            
            hc_mn, h_ij, h_ia = self.get_ham_in_mol_orb_representation(
                h_mm, c_nm, f_n, repr_name, use_egdecomp)
            
            with timer('Residual'):
                error = self.get_residual_error(
                    hc_mn, kpt.S_MM, c_nm, h_ij, f_n)

            if repr_name in ['sparse', 'u_invar'] and not use_egdecomp:
                if repr_name == 'sparse':
                    indl_oo = np.tril_indices(h_ij.shape[0])
                    h_ij[indl_oo] = np.inf
                    h_ij = np.concatenate((h_ij, h_ia), axis=1)
                else:
                    h_ij = h_ia
                h_ij = h_ij.ravel()
                h_ij = h_ij[h_ij != np.inf]
        
                ones_mat = np.ones(shape=(len(f_n), len(f_n)))
                anti_occ = f_n * ones_mat - f_n[:, np.newaxis] * ones_mat
                anti_occ = anti_occ[ind_up]
                grad = np.ascontiguousarray(anti_occ * h_ij)

            else:
                h_ij = f_n * h_ij - f_n[:, np.newaxis] * h_ij
                grad = np.ascontiguousarray(h_ij)

                if use_egdecomp:
                    with timer('Use Eigendecomposition'):
                        grad = self.get_exact_gradient_matrix(
                            grad, evec, evals)

                if repr_name in ['sparse', 'u_invar']:
                    grad = grad[ind_up]

        if self.dtype == float:
            grad = grad.real

        return 2.0 * grad, error

    def get_ham_in_mol_orb_representation(self, h_mm, c_nm,
                                          f_n, matrix_rep, full_ham):

        """
        H = (C_nM @ H_MM @ C_nM.T.conj()).conj()
        for sparse and u_inv representation we calculate
        H_ij and H_ia, where i,j -- occupied and a - virtual
        H_ij is really needed only to calculate the residual later
        but it is not needed for u_inver representation.
        When matrix exp calcualted using egdecomp method
        we need the whole matrix H though

        :return: H@C_nM[:occ].T, H_ij, H_ia or
                 H@C_nM.T, H, H_ia
        """

        occ = sum(f_n > 1.0e-10)
        if matrix_rep in ['sparse', 'u_invar'] and not full_ham:
            hc_mn = h_mm.conj() @ c_nm[:occ].T
            h_ij = hc_mn.T.conj() @ c_nm[:occ].T
            h_ia = hc_mn.T.conj() @ c_nm[occ:].T
        else:
            hc_mn = h_mm.conj() @ c_nm.T
            h_ij = c_nm.conj() @ hc_mn
            h_ia = h_ij[:occ][:, occ:]

        return hc_mn, h_ij, h_ia
    
    def get_residual_error(self, hc_mn, S_MM, c_nm, h_ij, f_n):
        """
        calculate residual error of KS equations
        """

        occ = sum(f_n > 1.0e-10)
        hc_mn = hc_mn[:, :occ] - \
            S_MM.conj() @ c_nm[:occ].T @ h_ij[:occ, :occ]
        norm = sum(hc_mn.conj() * hc_mn * f_n[:occ])
        error = sum(norm.real) * Hartree ** 2 / self.nvalence
        
        return error

    def get_exact_gradient_matrix(self, h_ij, evec, evals):
 
        """
        given eigendecomposition of A
        calculate exact gradient matrix
        eq.(14), (39)-(41) from
        arXiv:2101.12597 [physics.comp-ph]
        Comput. Phys. Commun. 267, 108047 (2021).
        :return: gradient matrix
        """

        grad = evec.T.conj() @ h_ij @ evec
        grad = grad * d_matrix(evals)
        grad = evec @ grad @ evec.T.conj()
        for i in range(grad.shape[0]):
            grad[i][i] *= 0.5
        
        return grad
