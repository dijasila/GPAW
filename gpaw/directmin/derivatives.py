import numpy as np
from gpaw.directmin.etdm import random_a


class Derivatives:

    def __init__(self, etdm, wfs, c_ref=None, a_vec_u=None,
                 update_c_ref=False, eps=1.0e-7, random_amat=False):
        """
        :param etdm:
        :param wfs:
        :param c_ref: reference orbitals C_ref
        :param a_vec_u: skew-Hermitian matrix A
        :param update_c_ref: if True update reference orbitals
        :param eps: finite difference displacement
        :param random_amat: if True, use random matrix A
        """

        self.eps = eps

        # initialize vectors of elements matrix A
        if a_vec_u is None:
            self.a_vec_u = {u: np.zeros_like(v) for u,
                                                    v in etdm.a_vec_u.items()}

        if random_amat:
            for kpt in wfs.kpt_u:
                u = etdm.kpointval(kpt)
                a = random_a(etdm.a_vec_u[u].shape, wfs.dtype)
                wfs.gd.comm.broadcast(a, 0)
                self.a_vec_u[u] = a

        # initialize orbitals:
        if c_ref is None:
            self.c_ref = etdm.dm_helper.reference_orbitals
        else:
            self.c_ref = c_ref

        # update ref orbitals if needed
        if update_c_ref:
            etdm.rotate_wavefunctions(wfs, self.a_vec_u, etdm.n_dim,
                                      self.c_ref)
            etdm.dm_helper.set_reference_orbitals(wfs, etdm.n_dim)
            self.c_ref = etdm.dm_helper.reference_orbitals
            self.a_vec_u = {u: np.zeros_like(v) for u,
                                                    v in etdm.a_vec_u.items()}

    def get_analytical_derivatives(self, etdm, ham, wfs, dens,
                                   what2calc='gradient'):
        """
           Calculate analytical gradient or approximation to the Hessian
           with respect to the elements of a skew-Hermitian matrix

        :param etdm:
        :param ham:
        :param wfs:
        :param dens:
        :param what2calc: calculate gradient or Hessian
        :return: analytical gradient or Hessian
        """

        assert what2calc in ['gradient', 'hessian']

        if what2calc == 'gradient':
            # calculate analytical gradient
            analytical_der = etdm.get_energy_and_gradients(self.a_vec_u,
                                                           etdm.n_dim,
                                                           ham, wfs, dens,
                                                           self.c_ref)[1]
        else:
            # Calculate analytical approximation to hessian
            analytical_der = np.hstack([etdm.get_hessian(kpt).copy()
                                        for kpt in wfs.kpt_u])
            analytical_der = construct_real_hessian(analytical_der)
            analytical_der = np.diag(analytical_der)

        return analytical_der

    def get_numerical_derivatives(self, etdm, ham, wfs, dens,
                                  what2calc='gradient'):
        """
           Calculate numerical gradient or Hessian with respect to
           the elements of a skew-Hermitian matrix using central finite
           differences

        :param etdm:
        :param ham:
        :param wfs:
        :param dens:
        :param what2calc: calculate gradient or Hessian
        :return: numerical gradient or Hessian
        """

        assert what2calc in ['gradient', 'hessian']

        # total dimensionality if matrices are real
        dim = sum([len(a) for a in self.a_vec_u.values()])
        steps = [1.0, 1.0j] if etdm.dtype == complex else [1.0]
        use_energy_or_gradient = {'gradient': 0, 'hessian': 1}

        matrix_exp = etdm.matrix_exp
        if what2calc == 'gradient':
            numerical_der = {u: np.zeros_like(v) for u,
                                                     v in self.a_vec_u.items()}
        else:
            numerical_der = np.zeros(shape=(len(steps) * dim,
                                            len(steps) * dim))
            # have to use exact gradient when Hessian is calculated
            etdm.matrix_exp = 'egdecomp'

        row = 0
        f = use_energy_or_gradient[what2calc]
        for step in steps:
            for kpt in wfs.kpt_u:
                u = etdm.kpointval(kpt)
                for i in range(len(self.a_vec_u[u])):
                    a = self.a_vec_u[u][i]

                    self.a_vec_u[u][i] = a + step * self.eps
                    fplus = etdm.get_energy_and_gradients(
                        self.a_vec_u, etdm.n_dim, ham, wfs, dens,
                        self.c_ref)[f]

                    self.a_vec_u[u][i] = a - step * self.eps
                    fminus = etdm.get_energy_and_gradients(
                        self.a_vec_u, etdm.n_dim, ham, wfs, dens,
                        self.c_ref)[f]

                    derf = apply_central_finite_difference_approx(
                        fplus, fminus, self.eps)

                    if what2calc == 'gradient':
                        numerical_der[u][i] += step * derf
                    else:
                        numerical_der[row] = construct_real_hessian(derf)

                    row += 1
                    self.a_vec_u[u][i] = a

        if what2calc == 'hessian':
            etdm.matrix_exp = matrix_exp

        return numerical_der


def construct_real_hessian(hess):

    if hess.dtype == complex:
        hess_real = np.hstack((np.real(hess), np.imag(hess)))
    else:
        hess_real = hess

    return hess_real


def apply_central_finite_difference_approx(fplus, fminus, eps):

    if isinstance(fplus, dict) and isinstance(fminus, dict):
        assert (len(fplus) == len(fminus))
        derf = np.hstack([(fplus[k] - fminus[k]) * 0.5 / eps
                          for k in fplus.keys()])
    elif isinstance(fplus, float) and isinstance(fminus, float):
        derf = (fplus - fminus) * 0.5 / eps
    else:
        raise ValueError()

    return derf
