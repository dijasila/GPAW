# import enum
from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

from gpaw import GPAW
from gpaw.atom.aeatom import AllElectronAtom
from gpaw.spherical_harmonics import yLL
from gpaw.gaunt import gaunt
from gpaw.utilities.progressbar import ProgressBar

from asr.core import read_json, write_json

from ase.dft.kpoints import ibz_points, kpoint_convert




def get_atoms_matrix_elements(symbol):

    if not isinstance(symbol,str):
        symbol = int(symbol)
    
    aeatom = AllElectronAtom(symbol=symbol,xc='LDA',ee_interaction=True)
    # for l in range(4):
    #     for n in range(l+1,8):
    #         aeatom.add(n = n, l = l, df=0)
    aeatom.run()

    r = aeatom.rgd.r_g
    Vr = aeatom.vr_sg[0]
    dVrdr = aeatom.rgd.derivative(Vr)
    r2dVdr = r*dVrdr - Vr

    states = []
    n_j = []
    l_j = []

    for ch in aeatom.channels:
        l = ch.l
        for n, phi_g in enumerate(ch.phi_ng):
            for m in range(2*l+1):
                f = ch.f_n[n]
            # if f > 0:# and ch.s == s:
                states.append((n+l+1, l, m, f * aeatom.nspins / 2.0 / (2 * l + 1),
                                phi_g))
            l_j.append(l)
            n_j.append(n+l+1)
            # if l > lmax:
            #         lmax = l

    ni = len(states)

    G_LLL = gaunt(3)

    M_iiv = np.zeros((ni,ni,3))

    for i, (n1, l1, m1, f1, phi1_g) in enumerate(states):
        for j, (n2, l2, m2, f2, phi2_g) in enumerate(states):
            G = G_LLL[l1**2 + m1,
                      l2**2 + m2,
                      [3,1,2]]
            r_int = aeatom.rgd.integrate(phi1_g * phi2_g * r2dVdr, n=-2) / (4*pi) / (sqrt(3./4/pi))
            M_iiv[i,j,:] = f1 * f2 * G * r_int
            # M_iiv[i,j,:] = G * r_int

    return M_iiv,l_j,n_j


def get_elph_nesting(calc,
                     bands = None, 
                     q_qc = None,
                     include_nesting = False,
                     results_file = None):

    if isinstance(calc,str):
        calc = GPAW(calc)

    Z_a = calc.atoms.get_atomic_numbers()
    M_a = calc.atoms.get_masses()
    kd = calc.wfs.kd

    if q_qc is None:
        q_qc = kd.get_bz_q_points(first=True)
    else:
        q_qc = np.array(q_qc)
    nq = q_qc.shape[0]

    nesting_q = np.zeros((nq), dtype=complex)
    
    elph_qav = np.zeros((nq,len(Z_a),3),dtype=complex)
    elph_nesting_qav = np.zeros((nq,len(Z_a),3),dtype=complex)

    elph2_q = np.zeros((nq),dtype=complex)
    elph2_nesting_q = np.zeros((nq),dtype=complex)

    eigs_q = np.zeros((nq,len(Z_a)*3),dtype=complex)
    eigs_nesting_q = np.zeros((nq,len(Z_a)*3),dtype=complex)

    M_aiiv = {a: None for a in range(len(Z_a))}
    l_aj = {a: None for a in range(len(Z_a))}
    n_aj = {a: None for a in range(len(Z_a))}
    for a,Z in enumerate(Z_a):
        M_iiv,l_j,n_j = get_atoms_matrix_elements(Z)
        M_aiiv[a] = M_iiv
        l_aj[a] = l_j
        n_aj[a] = n_j
    
    if bands is None:
        bands = np.arange(calc.get_number_of_bands())
    elif isinstance(bands,float):
        bands = np.arange(calc.get_number_of_bands())[(np.abs(np.array([calc.get_eigenvalues(k) for k in range(kd.nibzkpts)]) - calc.get_fermi_level()) < bands).any(0)]
    elif isinstance(bands,int):
        bands = np.array([bands])
    elif isinstance(bands,list):
        bands = np.array(bands)
    print('bands = {}'.format(bands))
    
    # Find sets of states included in wfs setups and isolated atoms, and the matches between the two sets
    setup_states_i = [[(n,l,m) for n,l in zip(s.n_j,s.l_j) for m in range(-l,l+1)] for s in calc.wfs.setups]
    atom_states_i = [[(n,l,m) for n,l in  zip(n_aj[a],l_aj[a]) for m in range(-l,l+1)] for a in range(len(Z_a))]
    match_states_i = [[s for s in setup_states_i[a] if s in atom_states_i[a]] for a in range(len(Z_a))]
    match_atom_states = [[s in match_states_i[a] for s in atom_states_i[a]] for a in range(len(Z_a))]
    match_setup_states = [[s in match_states_i[a] for s in setup_states_i[a]] for a in range(len(Z_a))]

    P_kani = [[calc.wfs.kpt_u[ik].P_ani[a][bands][:,match_setup_states[a]] for a in range(len(Z_a))] for ik in range(kd.nibzkpts)]
    for a in range(len(Z_a)):
        M_aiiv[a] = M_aiiv[a][match_atom_states[a]][:,match_atom_states[a]]

    eta = 0.01
    pb = ProgressBar()
    for q, q_c in enumerate(q_qc):
        # print(q, q_c)
        kplusq_k = kd.find_k_plus_q(q_c)

        nesting_nm = 0
        elph2 = 0
        elph2_nesting = 0
        elph_av = np.zeros((len(Z_a),3),dtype=complex)
        elph_nesting_av = np.zeros((len(Z_a),3),dtype=complex)
        elph_mat = np.zeros((3*len(Z_a),3*len(Z_a)),dtype=complex)
        elph_nesting_mat = np.zeros((3*len(Z_a),3*len(Z_a)),dtype=complex)

        for k in range(kd.nbzkpts):
            M_anmv = np.zeros((len(Z_a),len(bands),len(bands),3),dtype=complex)

            bz2ibzmap = calc.get_bz_to_ibz_map()

            ik = bz2ibzmap[k]
            ikplusq = bz2ibzmap[kplusq_k[k]]

            wk = calc.get_k_point_weights()[ik]
            wkplusq = calc.get_k_point_weights()[ikplusq]

            f_n = calc.get_occupation_numbers(ikplusq) / wkplusq
            f_m = calc.get_occupation_numbers(ik) / wk
            # f_n = (calc.get_occupation_numbers(ikplusq) > 1e-10).astype(float)
            # f_m = (calc.get_occupation_numbers(ik) > 1e-10).astype(float)

            ek_m = calc.get_eigenvalues(ik)
            ekplusq_n = calc.get_eigenvalues(ikplusq)

            # fk_nm =  (f_n[np.newaxis,bands] - f_m[bands,np.newaxis])
            # epsk_nm = 1/(ek_m[bands,np.newaxis] - ekplusq_n[np.newaxis,bands] - 1j * eta)
            fk_nm =  (f_n[bands,np.newaxis] - f_m[np.newaxis,bands])
            epsk_nm = 1/(ek_m[np.newaxis,bands] - ekplusq_n[bands,np.newaxis] - 1j * eta)
            nesting_nm += fk_nm * epsk_nm

            # dV = dV_av terms 
            for a in range(len(Z_a)):
                P_ni = np.conj(P_kani[ikplusq][a])
                P_mi = P_kani[ik][a]
                M_iiv = M_aiiv[a]
                M = np.einsum('ni,mj,ijv -> nmv',P_ni,P_mi,M_iiv)
                M_anmv[a,:,:,:] = M
                elph_av[a] += np.einsum('nmv -> v',M.conj()*M)
                elph_nesting_av[a] += np.einsum('nm,nmv -> v',fk_nm * epsk_nm, M.conj()*M)
            
            # dV = dV_av + dV_bw terms 
            for a in range(len(Z_a)):
                for b in range(len(Z_a)):
                    M = M_anmv[a,:,:,np.newaxis,:] * M_anmv[a,:,:,np.newaxis,:].conj() + M_anmv[b,:,:,:,np.newaxis] * M_anmv[b,:,:,:,np.newaxis].conj() + M_anmv[a,:,:,np.newaxis,:] * M_anmv[b,:,:,:,np.newaxis].conj() + M_anmv[b,:,:,:,np.newaxis] * M_anmv[a,:,:,np.newaxis,:].conj() 
                    # M = M_anmv[a,:,:,np.newaxis,:] * M_anmv[b,:,:,:,np.newaxis].conj() + M_anmv[b,:,:,:,np.newaxis] * M_anmv[a,:,:,np.newaxis,:].conj() 
                    elph2 += np.einsum('nmvw -> ',M)
                    elph2_nesting += np.einsum('nm,nmvw -> ',fk_nm * epsk_nm, M)

                    elph_mat[a*3:(a+1)*3,b*3:(b+1)*3] += np.einsum('nmvw -> vw',M) * 1/sqrt(M_a[a] * M_a[b])
                    elph_nesting_mat[a*3:(a+1)*3,b*3:(b+1)*3] += np.einsum('nm, nmvw -> vw',fk_nm * epsk_nm, M) * 1/sqrt(M_a[a] * M_a[b])
        elph_mat /= kd.nbzkpts
        elph_nesting_mat /= kd.nbzkpts
        eigs = np.sort(np.linalg.eigvals(elph_mat))
        eigs_nesting = np.sort(np.linalg.eigvals(elph_nesting_mat))

        nesting_q[q] = np.sum(nesting_nm) / kd.nbzkpts
        elph_qav[q] = elph_av / kd.nbzkpts
        elph_nesting_qav[q] = elph_nesting_av / kd.nbzkpts
        elph2_q[q] = elph2 / kd.nbzkpts
        elph2_nesting_q[q] = elph2_nesting / kd.nbzkpts
        eigs_q[q] = eigs
        eigs_nesting_q[q] = eigs_nesting
        pb.update(q / nq)
    
    pb.finish()

    if results_file is not None:
        data = {'eta': eta, 
                'bands': bands,
                'q_qc': q_qc, 
                'nesting_q': nesting_q,
                'elph_qav': elph_qav,
                'elph_nesting_qav': elph_nesting_qav,
                'elph2_q': elph2_q,
                'elph2_nesting_q': elph2_nesting_q,
                'eigs_q': eigs_q,
                'eigs_nesting_q': eigs_nesting_q}

        write_json('results-'+results_file+'.json', data)
    return

def plot_elph_nesting(atoms,results_file):
    data = read_json('results-'+results_file+'.json')
    q_qc = data['q_qc']
    nesting_q = data['nesting_q']
    elph_qav = data['elph_qav']
    elph_nesting_qav = data['elph_nesting_qav']
    elph2_q = data['elph2_q']
    elph2_nesting_q = data['elph2_nesting_q']
    eigs_q = data['eigs_q']
    eigs_nesting_q = data['eigs_nesting_q']

    nq, natoms, nv = elph_qav.shape
    print(nq, natoms, nv)

    symbols = atoms.get_chemical_symbols()
    cell_cv = atoms.cell
    a = np.linalg.norm(cell_cv[0])
    extent = (np.min(q_qc[:,0]),np.max(q_qc[:,0]),np.min(q_qc[:,1]),np.max(q_qc[:,1]))

    q_qv = kpoint_convert(cell_cv, skpts_kc=q_qc) * a / (2*pi)

    fig, ax = plt.subplots(natoms,3,figsize=(9,3*natoms))
    for a in range(natoms):
        for v,alpha in enumerate(['x','y','z']):
            Q = np.array(elph_qav[:,a,v], float).reshape(int(sqrt(nq)),int(sqrt(nq)))
            im = ax[a,v].imshow(Q, origin='lower', vmax=np.max(Q), vmin=np.min(Q), #interpolation='gaussian',
                        extent=extent,aspect='equal')
            fig.colorbar(im,ax=ax[a,v])
            # ax[a,v].set_xlim(-0.5,0.5)
            # ax[a,v].set_ylim(-0.5,0.5)
            ax[a,v].set_xlabel('$q_x$')
            ax[a,v].set_ylabel('$q_y$')
            ax[a,v].set_title('$E_{{ {0},{1} }}$'.format(symbols[a],alpha))
    fig.tight_layout()
    fig.savefig(results_file+'_elph.png')

    fig, ax = plt.subplots(natoms,3,figsize=(9,3*natoms))
    for a in range(natoms):
        for v,alpha in enumerate(['x','y','z']):
            Q = np.array(elph_nesting_qav[:,a,v], float).reshape(int(sqrt(nq)),int(sqrt(nq)))
            im = ax[a,v].imshow(Q, origin='lower', vmax=np.max(Q), vmin=np.min(Q), #interpolation='gaussian',
                        extent=extent,aspect='equal')
            fig.colorbar(im,ax=ax[a,v])
            # ax[a,v].set_xlim(-0.5,0.5)
            # ax[a,v].set_ylim(-0.5,0.5)
            ax[a,v].set_xlabel('$q_x$')
            ax[a,v].set_ylabel('$q_y$')
            ax[a,v].set_title('$D_{{ {0},{1} }}$'.format(symbols[a],alpha))
    fig.tight_layout()
    fig.savefig(results_file+'_elph_nesting.png')

    fig,ax = plt.subplots(3,2,figsize=(6,9))
    ax = ax.flatten()
    Q = np.array(nesting_q, float).reshape(int(sqrt(nq)),int(sqrt(nq)))
    im = ax[0].imshow(Q, origin='lower', vmax=np.max(Q), vmin=np.min(Q), #interpolation='gaussian',
                extent=extent,aspect='equal')
    fig.colorbar(im,ax=ax[0])
    # ax[0].set_xlim(-0.5,0.5)
    # ax[0].set_ylim(-0.5,0.5)
    ax[0].set_xlabel('$q_x$')
    ax[0].set_ylabel('$q_y$')
    ax[0].set_title('$N(q)$')

    Q = np.array(elph2_q, float).reshape(int(sqrt(nq)),int(sqrt(nq)))
    im = ax[2].imshow(Q, origin='lower', vmax=np.max(Q), vmin=np.min(Q), #interpolation='gaussian',
                extent=extent,aspect='equal')
    fig.colorbar(im,ax=ax[2])
    # ax[0].set_xlim(-0.5,0.5)
    # ax[0].set_ylim(-0.5,0.5)
    ax[2].set_xlabel('$q_x$')
    ax[2].set_ylabel('$q_y$')
    ax[2].set_title(r'$E_{\Sigma}(q)$')

    Q = np.array(elph2_nesting_q, float).reshape(int(sqrt(nq)),int(sqrt(nq)))
    im = ax[3].imshow(Q, origin='lower', vmax=np.max(Q), vmin=np.min(Q), #interpolation='gaussian',
                extent=extent,aspect='equal')
    fig.colorbar(im,ax=ax[3])
    # ax[0].set_xlim(-0.5,0.5)
    # ax[0].set_ylim(-0.5,0.5)
    ax[3].set_xlabel('$q_x$')
    ax[3].set_ylabel('$q_y$')
    ax[3].set_title(r'$D_{\Sigma}(q)$')

    Q = np.array(eigs_q[:,0], float).reshape(int(sqrt(nq)),int(sqrt(nq)))
    im = ax[4].imshow(Q, origin='lower', vmax=np.max(Q), vmin=np.min(Q), #interpolation='gaussian',
                extent=extent,aspect='equal')
    fig.colorbar(im,ax=ax[4])
    # ax[0].set_xlim(-0.5,0.5)
    # ax[0].set_ylim(-0.5,0.5)
    ax[4].set_xlabel('$q_x$')
    ax[4].set_ylabel('$q_y$')
    ax[4].set_title(r'$E_{\lambda^*}(q)$')

    Q = np.array(eigs_nesting_q[:,0], float).reshape(int(sqrt(nq)),int(sqrt(nq)))
    im = ax[5].imshow(Q, origin='lower', vmax=np.max(Q), vmin=np.min(Q), #interpolation='gaussian',
                extent=extent,aspect='equal')
    fig.colorbar(im,ax=ax[5])
    # ax[0].set_xlim(-0.5,0.5)
    # ax[0].set_ylim(-0.5,0.5)
    ax[5].set_xlabel('$q_x$')
    ax[5].set_ylabel('$q_y$')
    ax[5].set_title(r'$D_{\lambda^*}(q)$')

    special_points = {'G':[0.0,0.0,0.0], 'X':[0.0,0.5,0.0], 'M':[0.5,0.5,0.0], 'K':[1/3,1/3,0.0]}
    q_idx = []
    for q,q_c in special_points.items():
        q_idx.append(np.isclose(q_qc,q_c).all(1).argmax())
    ax[1].plot(eigs_nesting_q[q_idx,:],'.',color='C0')
    ax[1].set_xticks(np.arange(len(list(special_points.keys()))))
    ax[1].set_xticklabels(list(special_points.keys()))

    fig.tight_layout()
    fig.savefig(results_file+'_nesting.png')
    
    # nest = np.array(elph_q, float).reshape(int(sqrt(nq)),int(sqrt(nq)))
    # im = ax[1].imshow(nest, origin='lower', vmax=np.max(nest), vmin=np.min(nest), #interpolation='gaussian',
    #             extent=(-0.5,0.5,-0.5,0.5),aspect='equal')
    # fig.colorbar(im,ax=ax[1])
    # # ax[1].set_xlim(-0.5,0.5)
    # # ax[1].set_ylim(-0.5,0.5)
    # ax[1].set_xlabel('$q_x$')
    # ax[1].set_ylabel('$q_y$')
    # ax[1].set_title('$D(q)$')
    # fig.tight_layout()
    # fig.savefig(results_file+'_nesting.png')

    # fig,ax = plt.subplots()
    # for i in range(eigs_q.shape[1]):
    #     ax.plot(eigs_q[:,i],'.',color='C0')
    # fig.savefig(results_file+'_eigs.png')


if __name__ == '__main__':
    # Unstable materials:

    calc = GPAW('tree-c2db/AB2/GaF2/GaF2-1e195eaaf561/gs.gpw')
    # get_elph_nesting(calc,results_file='elph_GaF2')#,bands=5.0)#,q_qc=[[0.0,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0],[1/3,1/3,0.0]])
    plot_elph_nesting(calc.atoms,results_file='elph_GaF2')

    # calc = GPAW('tree-c2db/AB2/GeCl2/GeCl2-8b0f0207ae45/gs.gpw')
    # get_elph_nesting(calc,results_file='elph_GeCl2')#,bands=5.0)#,q_qc=[[0.0,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0],[1/3,1/3,0.0]])
    # plot_elph_nesting(calc.atoms,results_file='elph_GeCl2')

    # calc = GPAW('tree-c2db/AB2/TaI2/TaI2-b2b688fff665/gs.gpw')
    # get_elph_nesting(calc,results_file='elph_TaI2')#,bands=5.0)#,q_qc=[[0.0,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0],[1/3,1/3,0.0]])
    # plot_elph_nesting(calc.atoms,results_file='elph_TaI2')


    # # # Stable materials:

    
    
    # calc = GPAW('tree-c2db/AB2/MoS2/MoS2-b3b4685fb6e1/gs.gpw')
    # get_elph_nesting(calc,results_file='elph_MoS2')#,bands=5.0)#,q_qc=[[0.0,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0],[1/3,1/3,0.0]])
    # plot_elph_nesting(calc.atoms,results_file='elph_MoS2')

    # calc = GPAW('tree-c2db/AB2/AlI2/AlI2-ed00aecbc014/gs.gpw')
    # get_elph_nesting(calc,results_file='elph_AlI2')#,bands=5.0)#,q_qc=[[0.0,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0],[1/3,1/3,0.0]])
    # plot_elph_nesting(calc.atoms,results_file='elph_AlI2')

    # calc = GPAW('tree-c2db/AB2/CrO2/CrO2-aa2856f95afa/gs.gpw')
    # get_elph_nesting(calc,results_file='elph_CrO2')#,bands=5.0)#,q_qc=[[0.0,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0],[1/3,1/3,0.0]])
    # plot_elph_nesting(calc.atoms,results_file='elph_CrO2')

    
