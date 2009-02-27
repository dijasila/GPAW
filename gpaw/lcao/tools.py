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

def get_realspace_hs(h_skmm,s_kmm, ibzk_kc, weight_k, R_c=(0,0,0), 
                     usesymm=None):

    phase_k = np.dot(2 * np.pi * ibzk_kc, R_c)
    c_k = np.exp(1.0j * phase_k) * weight_k
    c_k.shape = (len(ibzk_kc),1,1)

    if usesymm==None:                     
        if h_skmm != None:
            nbf = h_skmm.shape[-1]
            nspins = len(h_skmm)
            h_smm = np.empty((nspins,nbf,nbf),complex)
            for s in range(nspins):
                h_smm[s] = np.sum((h_skmm[s] * c_k), axis=0)
        if s_kmm != None:
            nbf = s_kmm.shape[-1]
            s_mm = np.empty((nbf,nbf),complex)
            s_mm[:] = np.sum((s_kmm * c_k), axis=0)      
        if h_skmm != None and s_kmm != None:
            return h_smm, s_mm
        elif h_skmm == None:
            return s_mm
        elif s_kmm == None:
            return h_smm

    elif usesymm==False:        
        nbf = h_skmm.shape[-1]
        nspins = len(h_skmm)
        h_smm = np.empty((nspins, nbf, nbf))
        s_mm = np.empty((nbf,nbf))
        for s in range(nspins):
            h_smm[s] = np.sum((h_skmm[s] * c_k).real, axis=0)
   
        s_mm[:] = np.sum((s_kmm * c_k).real, axis=0)
        
        return h_smm, s_mm

    elif usesymm==True:
        raise 'Not implemented'
            

def get_kspace_hs(h_srmm, s_rmm, R_vector, kvector=(0,0,0)):
    phase_k = np.dot(2 * np.pi * R_vector, kvector)
    c_k = np.exp(-1.0j * phase_k)
    c_k.shape = (len(R_vector), 1, 1)
    
    if h_srmm != None:
        nbf = h_srmm.shape[-1]
        nspins = len(h_srmm)
        h_smm = np.empty((nspins, nbf, nbf), complex)
        for s in range(nspins):
            h_smm[s] = np.sum((h_srmm[s] * c_k), axis=0)
    elif s_rmm != None:
        nbf = s_rmm.shape[-1]
        s_mm = np.empty((nbf, nbf), complex)
        s_mm[:] = np.sum((s_rmm * c_k), axis=0)
    if h_srmm != None and s_rmm != None:    
        return h_smm, s_mm
    elif h_srmm == None:
        return s_mm
    elif s_rmm == None:
        return h_smm

def remove_pbc(atoms, h, s=None, d=0):
    calc = atoms.get_calculator()
    if not calc.initialized:
        calc.initialize(atoms)

    nao = calc.wfs.setups.nao
    
    cutoff = atoms.get_cell()[d,d] * 0.5 
    pos_i = get_bf_centers(atoms)[:,d]
    for i in range(nao):
        dpos_i = np.absolute(pos_i - pos_i[i])
        mask_i = (dpos_i < cutoff).astype(int)
        h[i,:] = h[i,:] * mask_i
        h[:,i] = h[:,i] * mask_i
        if s != None:
            s[i,:] = s[i,:] * mask_i
            s[:,i] = s[:,i] * mask_i



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
        may be changed into a dump to a single file in the feature.

    """
    if direction!=None:
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
    
    if wfs.gd.domain.comm.rank==0:
        fd = file(filename+'%i.pckl' % wfs.kpt_comm.rank, 'wb')
        H_qMM *= Hartree
        pickle.dump((H_qMM, S_qMM),fd , 2)
        calc_data
        pickle.dump(calc_data, fd, 2) 
        fd.close()

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

def lead_kspace2realspace(filename, direction='x'):
    """Convert a dumped hamiltonian representing a lead, to a realspace
    hamiltonian of double size representing two principal layers and the
    coupling between."""
    dir = 'xyz'.index(direction)
    fd = file(filename, 'rb')
    h_skmm, s_kmm = pickle.load(fd)
    atom_data = pickle.load(fd)
    calc_data = pickle.load(fd)
    fd.close()

    nbf = h_skmm.shape[-1]
    nspin = len(h_skmm)
    h_smm = np.zeros((nspin, 2 * nbf, 2 * nbf), h_skmm.dtype)
    s_mm = np.zeros((2 * nbf, 2 * nbf), h_skmm.dtype)

    R_c = [0, 0, 0]
    h_sii, s_ii = get_realspace_hs(h_skmm,
                                   s_kmm,
                                   calc_data['ibzk_kc'],
                                   calc_data['weight_k'],
                                   R_c)
    R_c[dir] = 1.
    h_sij, s_ij = get_realspace_hs(h_skmm,
                                   s_kmm,
                                   calc_data['ibzk_kc'],
                                   calc_data['weight_k'],
                                   R_c)

    h_smm[:, :nbf, :nbf] = h_smm[:, nbf:, nbf:] = h_sii
    h_smm[:, :nbf, nbf:] = h_sij
    h_smm[:, nbf:, :nbf] = h_sij.swapaxes(1, 2).conj()

    s_mm[:nbf, :nbf] = s_mm[nbf:, nbf:] = s_ii
    s_mm[:nbf, nbf:] = s_ij
    s_mm[nbf:, :nbf] = s_ij.T.conj()

    return h_smm, s_mm
