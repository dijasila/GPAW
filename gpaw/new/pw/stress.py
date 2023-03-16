from gpaw.gpu import synchronize
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.calculation import DFTState
from gpaw.typing import Array2D


def calculate_stress(pot_calc,
                     state: DFTState,
                     vt_g,
                     nt_g) -> Array2D:
    xc = pot_calc.xc

    if xc.xc.orbital_dependent and xc.type != 'MGGA':
        raise NotImplementedError('Calculation of stress tensor is not ' +
                                  'implemented for orbital-dependent ' +
                                  'XC functionals such as ' + xc.name)
    assert xc.type != 'MGGA'
    assert not xc.no_forces

    s_vv = get_kinetic_stress(state.ibzwfs)

    vHt_h = state.vHt_x
    pw = vHt_h.desc
    G_Gv = pw.G_plus_k_Gv
    for v1 in range(3):
        for v2 in range(3):
            s_vv[v1, v2] += pd.integrate(p_G, dens.rhot_q *
                                         G_Gv[:, v1] * G_Gv[:, v2])
    s_vv += dens.ghat.stress_tensor_contribution(ham.vHt_q, dens.Q_aL)

    s_vv -= np.eye(3) * ham.estress
    s_vv += ham.vbar.stress_tensor_contribution(dens.nt_Q)
    s_vv += dens.nct.stress_tensor_contribution(ham.vt_Q)

    s0 = 0.0
    s0_vv = 0.0
    for kpt in wfs.kpt_u:
        a_ani = {}
        for a, P_ni in kpt.P_ani.items():
            Pf_ni = P_ni * kpt.f_n[:, None]
            dH_ii = unpack(ham.dH_asp[a][kpt.s])
            dS_ii = ham.setups[a].dO_ii
            a_ni = (np.dot(Pf_ni, dH_ii) -
                    np.dot(Pf_ni * kpt.eps_n[:, None], dS_ii))
            s0 += np.vdot(P_ni, a_ni)
            a_ani[a] = 2 * a_ni.conj()
        s0_vv += wfs.pt.stress_tensor_contribution(kpt.psit_nG, a_ani,
                                                   q=kpt.q)
    s0_vv -= dens.gd.comm.sum(s0.real) * np.eye(3)
    s0_vv /= dens.gd.comm.size
    wfs.world.sum(s0_vv)
    s_vv += s0_vv

    s_vv += wfs.dedepsilon * np.eye(3)

    vol = calc.atoms.get_volume() / units.Bohr**3
    s_vv = 0.5 / vol * (s_vv + s_vv.T)

    # Symmetrize:
    sigma_vv = np.zeros((3, 3))
    cell_cv = wfs.gd.cell_cv
    for U_cc in wfs.kd.symmetry.op_scc:
        M_vv = np.dot(np.linalg.inv(cell_cv),
                      np.dot(U_cc, cell_cv)).T
        sigma_vv += np.dot(np.dot(M_vv.T, s_vv), M_vv)
    sigma_vv /= len(wfs.kd.symmetry.op_scc)

    # Make sure all agree on the result (redundant calculation on
    # different cores involving BLAS might give slightly different
    # results):
    wfs.world.broadcast(sigma_vv, 0)

    calc.log('Stress tensor:')
    for sigma_v in sigma_vv:
        calc.log('{:13.6f}{:13.6f}{:13.6f}'
                 .format(*(units.Ha / units.Bohr**3 * sigma_v)))

    calc.timer.stop('Stress tensor')

    return sigma_vv


def get_kinetic_stress(ibzwfs: IBZWaveFunctions):
    xp = ibzwfs.xp
    sigma_vv = xp.zeros((3, 3))
    for wfs in ibzwfs:
        sigma_vv += get_kinetic_stress_single(wfs)
    synchronize()
    ibzwfs.kpt_comm.sum(sigma_vv)
    return sigma_vv


def get_kinetic_stress_single(wfs: PWFDWaveFunctions):
    occ_n = wfs.weight * wfs.spin_degeneracy * wfs.myocc_n
    psit_nG = wfs.psit_nX
    pw = psit_nG.desc
    xp = psit_nG.xp
    psit_nGz = psit_nG.data.view(float).reshape(psit_nG.data.shape + (2,))
    psit2_G = xp.einsum('n, nGz, nGz -> G', occ_n, psit_nGz, psit_nGz)
    Gk_Gv = xp.asarray(pw.G_plus_k_Gv)
    sigma_vv = xp.einsum('G, Gv, Gw -> vw', psit2_G, Gk_Gv, Gk_Gv)
    x = pw.dv
    if pw.dtype == float:
        x *= 2
    return -x * sigma_vv
