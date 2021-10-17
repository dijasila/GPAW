import numpy as np


def get_analytical_derivatives(etdm, ham, wfs, dens, c_nm_ref=None,
                               a_mat_u=None, update_c_nm_ref=False,
                               what2calc='gradient'):
    """
       Calculate analytical gradient or approximation to the Hessian
       with respect to the elements of a skew-Hermitian matrix

    :param ham:
    :param wfs:
    :param dens:
    :param c_nm_ref: reference orbitals
    :param a_mat_u: skew-hermitian matrix
    :param update_c_nm_ref: if True update reference orbitals
    :param what2calc: calculate gradient or Hessian
    :return: analytical gradient or Hessian
    """

    assert what2calc in ['gradient', 'hessian']

    if what2calc == 'gradient':
        a_mat_u, c_nm_ref = init_calc_derivatives(etdm, wfs, c_nm_ref, a_mat_u,
                                                  update_c_nm_ref)
        # Calculate analytical gradient
        analytical_der = etdm.get_energy_and_gradients(a_mat_u, etdm.n_dim,
                                                       ham, wfs, dens,
                                                       c_nm_ref)[1]
    else:
        # Calculate analytical approximation to hessian
        analytical_der = np.hstack([etdm.get_hessian(kpt).copy()
                                    for kpt in wfs.kpt_u])
        analytical_der = construct_real_hessian(analytical_der)
        analytical_der = np.diag(analytical_der)

    return analytical_der


def get_numerical_derivatives(etdm, ham, wfs, dens, c_nm_ref=None,
                              eps=1.0e-7, a_mat_u=None,
                              update_c_nm_ref=False,
                              what2calc='gradient'):
    """
       Calculate numerical gradient or Hessian with respect to
       the elements of a skew-Hermitian matrix using central finite
       differences

    :param ham:
    :param wfs:
    :param dens:
    :param c_nm_ref: reference orbitals
    :param eps: finite difference displacement
    :param a_mat_u: skew-Hermitian matrix
    :param update_c_nm_ref: if True update reference orbitals
    :param what2calc: calculate gradient or Hessian
    :return: numerical gradient or Hessian
    """

    assert what2calc in ['gradient', 'hessian']

    a_mat_u, c_nm_ref = init_calc_derivatives(etdm, wfs, c_nm_ref, a_mat_u,
                                              update_c_nm_ref)

    # total dimensionality if matrices are real
    dim = sum([len(a) for a in a_mat_u.values()])
    steps = [1.0, 1.0j] if etdm.dtype == complex else [1.0]
    use_energy_or_gradient = {'gradient': 0, 'hessian': 1}

    matrix_exp = etdm.matrix_exp
    if what2calc == 'gradient':
        numerical_der = {u: np.zeros_like(v) for u, v in a_mat_u.items()}
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
            for i in range(len(a_mat_u[u])):
                a = a_mat_u[u][i]

                a_mat_u[u][i] = a + step * eps
                fplus = etdm.get_energy_and_gradients(
                    a_mat_u, etdm.n_dim, ham, wfs, dens, c_nm_ref)[f]

                a_mat_u[u][i] = a - step * eps
                fminus = etdm.get_energy_and_gradients(
                    a_mat_u, etdm.n_dim, ham, wfs, dens, c_nm_ref)[f]

                derf = apply_central_finite_difference_approx(
                    fplus, fminus, eps)

                if what2calc == 'gradient':
                    numerical_der[u][i] += step * derf
                else:
                    numerical_der[row] = construct_real_hessian(derf)

                row += 1
                a_mat_u[u][i] = a

    if what2calc == 'hessian':
        etdm.matrix_exp = matrix_exp

    return numerical_der


def init_calc_derivatives(etdm, wfs, c_nm_ref=None,
                          a_mat_u=None, update_c_nm_ref=False):
    """
    Initialize skew-Hermitian and reference coefficient matrices
    for calculation of analytical or numerical derivatives
    """

    if c_nm_ref is None:
        c_nm_ref = etdm.dm_helper.reference_orbitals

    if a_mat_u is None:
        a_mat_u = {u: np.zeros_like(v) for u, v in etdm.a_mat_u.items()}

    # update ref orbitals if needed
    if update_c_nm_ref:
        etdm.rotate_wavefunctions(wfs, a_mat_u, etdm.n_dim, c_nm_ref)
        etdm.dm_helper.set_reference_orbitals(wfs, etdm.n_dim)
        c_nm_ref = etdm.dm_helper.reference_orbitals
        a_mat_u = {u: np.zeros_like(v) for u, v in etdm.a_mat_u.items()}

    return a_mat_u, c_nm_ref


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
