from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full
from ase.units import Hartree
import cPickle as pickle
import numpy as np
from gpaw.mpi import world, rank


def get_bf_centers(atoms):
    calc = atoms.get_calculator()
    if not calc.initialized:
        calc.initialize(atoms)
    nao = calc.wfs.setups.nao
    pos_ac = atoms.get_positions()
    natoms = len(pos_ac)
    pos_ic = np.zeros((nao, 3), np.float)
    index = 0
    for a in range(natoms):
        n = calc.wfs.setups[a].niAO
        pos_c = pos_ac[a]
        pos_c.shape = (1, 3)
        pos_ic[index:index + n] = np.repeat(pos_c, n, axis=0)
        index += n
    return pos_ic


def get_bf_centers2(atoms, bfs_dict):
    """bfs_dict is a dictionary mapping atom symbols to a number of bfs."""
    pos_ic = []
    for pos, sym in zip(atoms.get_positions(), atoms.get_chemical_symbols()):
        pos_ic.extend(pos[None].repeat(bfs_dict[sym], 0))
    return np.array(pos_ic)


def get_realspace_hs(h_skmm, s_kmm, ibzk_kc, weight_k, R_c=(0, 0, 0),
                     usesymm=None):
    # usesymm=False only works if the k-point reduction is only along one
    # direction.
    # For more functionality, see: gpaw/transport/tools.py
    
    nspins, nk, nbf = h_skmm.shape[:-1]
    c_k = np.exp(2.j * np.pi * np.dot(ibzk_kc, R_c)) * weight_k
    c_k.shape = (nk, 1, 1)

    if usesymm is None:
        h_smm = np.sum((h_skmm * c_k), axis=1)
        if s_kmm is not None:
            s_mm = np.sum((s_kmm * c_k), axis=0)
    elif usesymm is False:
        h_smm = np.sum((h_skmm * c_k).real, axis=1)
        if s_kmm is not None:
            s_mm = np.sum((s_kmm * c_k).real, axis=0)
    else: #usesymm is True:
        raise NotImplementedError, 'Only None and False have been implemented'

    if s_kmm is None:
        return h_smm
    return h_smm, s_mm


def remove_pbc(atoms, h, s=None, d=0, centers_ic=None, cutoff=None):
    L = atoms.cell[d, d]
    nao = len(h)
    if centers_ic is None:
        centers_ic = get_bf_centers(atoms) # requires an attached LCAO calc
    ni = len(centers_ic)
    if nao != ni:
        assert nao == 2 * ni
        centers_ic = np.vstack((centers_ic, centers_ic))
        centers_ic[ni:, d] += L
        if cutoff is None:
            cutoff = L
    elif cutoff is None:
        cutoff = 0.5 * L
    pos_i = centers_ic[:, d]
    for i in range(nao):
        dpos_i = abs(pos_i - pos_i[i])
        mask_i = (dpos_i < cutoff).astype(int)
        h[i, :] *= mask_i
        h[:, i] *= mask_i
        if s != None:
            s[i, :] *= mask_i
            s[:, i] *= mask_i


def dump_hamiltonian(filename, atoms, direction=None):
    h_skmm, s_kmm = get_hamiltonian(atoms)
    if direction != None:
        d = 'xyz'.index(direction)
        for s in range(atoms.calc.nspins):
            for k in range(atoms.calc.nkpts):
                if s==0:
                    remove_pbc(atoms, h_skmm[s, k], s_kmm[k], d)
                else:
                    remove_pbc(atoms, h_skmm[s, k], None, d)

    
    if atoms.calc.master:
        fd = file(filename,'wb')
        pickle.dump((h_skmm, s_kmm), fd, 2)
        atoms_data = {'cell':atoms.cell, 'positions':atoms.positions,
                      'numbers':atoms.numbers, 'pbc':atoms.pbc}
        
        pickle.dump(atoms_data, fd, 2)
        calc_data ={'weight_k':atoms.calc.weight_k, 
                    'ibzk_kc':atoms.calc.ibzk_kc}
        
        pickle.dump(calc_data, fd, 2)
        fd.close()

    world.barrier()


def dump_hamiltonian_parallel(filename, atoms, direction=None):
    """
        Dump the lcao representation of H and S to file(s) beginning
        with filename. If direction is x, y or z, the periodic boundary
        conditions will be removed in the specified direction. 
        If the Fermi temperature is different from zero,  the
        energy zero-point is taken as the Fermi level.

        Note:
        H and S are parallized over spin and k-points and
        is for now dumped into a number of pickle files. This
        may be changed into a dump to a single file in the future.

    """
    if direction != None:
        d = 'xyz'.index(direction)

    calc = atoms.calc
    wfs = calc.wfs
    nao = wfs.setups.nao
    nq = len(wfs.kpt_u) // wfs.nspins
    H_qMM = np.empty((wfs.nspins, nq, nao, nao), wfs.dtype)
    calc_data = {'k_q':{},
                 'skpt_qc':np.empty((nq, 3)), 
                 'weight_q':np.empty(nq)}

    S_qMM = wfs.S_qMM
   
    for kpt in wfs.kpt_u:
        calc_data['skpt_qc'][kpt.q] = calc.wfs.ibzk_kc[kpt.k]
        calc_data['weight_q'][kpt.q] = calc.wfs.weight_k[kpt.k]
        calc_data['k_q'][kpt.q] = kpt.k
#        print 'Calc. H matrix on proc. %i: (rk, rd, q, k)=(%i, %i, %i, %i)' % (wfs.world.rank, wfs.kpt_comm.rank, wfs.gd.domain.comm.rank, kpt.q, kpt.k)
        wfs.eigensolver.calculate_hamiltonian_matrix(calc.hamiltonian,
                                                     wfs, 
                                                     kpt)

        H_qMM[kpt.s, kpt.q] = wfs.eigensolver.H_MM

        tri2full(H_qMM[kpt.s, kpt.q])
        if kpt.s==0:
            tri2full(S_qMM[kpt.q])
            if direction!=None:
                remove_pbc(atoms, H_qMM[kpt.s, kpt.q], S_qMM[kpt.q], d)
        else:
            if direction!=None:
                remove_pbc(atoms, H_qMM[kpt.s, kpt.q], None, d)
        if calc.occupations.kT>0:
            H_qMM[kpt.s, kpt.q] -= S_qMM[kpt.q] * \
                                   calc.occupations.get_fermi_level()    
    
    if wfs.gd.comm.rank == 0:
        fd = file(filename+'%i.pckl' % wfs.kpt_comm.rank, 'wb')
        H_qMM *= Hartree
        pickle.dump((H_qMM, S_qMM),fd , 2)
        calc_data
        pickle.dump(calc_data, fd, 2) 
        fd.close()


def get_lead_lcao_hamiltonian(calc, usesymm=False):
    S_qMM = calc.wfs.S_qMM.copy()
    H_sqMM = np.empty((calc.wfs.nspins,) + S_qMM.shape, calc.wfs.dtype)
    for kpt in calc.wfs.kpt_u:
        calc.wfs.eigensolver.calculate_hamiltonian_matrix(
            calc.hamiltonian, calc.wfs, kpt)
        H_sqMM[kpt.s, kpt.q] = calc.wfs.eigensolver.H_MM * Hartree
        tri2full(S_qMM[kpt.q])
        tri2full(H_sqMM[kpt.s, kpt.q])
    return lead_kspace2realspace(H_sqMM, S_qMM, calc.wfs.ibzk_kc,
                                 calc.wfs.weight_k, 'x', usesymm)


def get_hamiltonian(atoms):
    """Calculate the Hamiltonian and overlap matrix."""
    calc = atoms.calc
    Ef = calc.get_fermi_level()
    eigensolver = calc.wfs.eigensolver
    hamiltonian = calc.hamiltonian
    Vt_skmm = eigensolver.Vt_skmm
    print "Calculating effective potential matrix (%i)" % rank 
    hamiltonian.calculate_effective_potential_matrix(Vt_skmm)
    ibzk_kc = calc.ibzk_kc
    nkpts = len(ibzk_kc)
    nspins = calc.nspins
    weight_k = calc.weight_k
    nao = calc.nao
    h_skmm = np.zeros((nspins, nkpts, nao, nao), complex)
    s_kmm = np.zeros((nkpts, nao, nao), complex)
    for k in range(nkpts):
        s_kmm[k] = hamiltonian.S_kmm[k]
        tri2full(s_kmm[k])
        for s in range(nspins):
            h_skmm[s,k] = calc.eigensolver.get_hamiltonian_matrix(hamiltonian,
                                                                  k=k,
                                                                  s=s)
            tri2full(h_skmm[s, k])
            h_skmm[s,k] *= Hartree
            h_skmm[s,k] -= Ef * s_kmm[k]

    return h_skmm, s_kmm


def lead_kspace2realspace_fromfile(filename, direction='x', usesymm=None):
    """Convert a dumped hamiltonian representing a lead, to a realspace
    hamiltonian of double size representing two principal layers and the
    coupling between."""
    fd = file(filename, 'rb')
    h_skmm, s_kmm = pickle.load(fd)
    atom_data = pickle.load(fd)
    calc_data = pickle.load(fd)
    ibzk_kc = calc.data['ibzk_kc']
    weight_k = calc.data['weight_k']
    fd.close()
    return lead_kspace2realspace(h_skmm, s_kmm, ibzk_kc, weight_k,
                                 direction, usesymm)


def lead_kspace2realspace(h_skmm, s_kmm, ibzk_kc, weight_k,
                          direction='x', usesymm=None):
    """Convert a k-dependent (in transport dir) Hamiltonian representing
    a lead, to a realspace hamiltonian of double size representing two
    principal layers and the coupling between."""
    dir = 'xyz'.index(direction)
    nspin, nk, nbf = h_skmm.shape[:-1]
    h_smm = np.zeros((nspin, 2 * nbf, 2 * nbf), h_skmm.dtype)
    s_mm = np.zeros((2 * nbf, 2 * nbf), h_skmm.dtype)

    R_c = [0, 0, 0]
    h_sii, s_ii = get_realspace_hs(h_skmm, s_kmm, ibzk_kc, weight_k,
                                   R_c, usesymm)
    R_c[dir] = 1.
    h_sij, s_ij = get_realspace_hs(h_skmm, s_kmm, ibzk_kc, weight_k,
                                   R_c, usesymm)

    h_smm[:, :nbf, :nbf] = h_smm[:, nbf:, nbf:] = h_sii
    h_smm[:, :nbf, nbf:] = h_sij
    h_smm[:, nbf:, :nbf] = h_sij.swapaxes(1, 2).conj()

    s_mm[:nbf, :nbf] = s_mm[nbf:, nbf:] = s_ii
    s_mm[:nbf, nbf:] = s_ij
    s_mm[nbf:, :nbf] = s_ij.T.conj()

    return h_smm, s_mm
