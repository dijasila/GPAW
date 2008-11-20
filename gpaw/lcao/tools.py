from gpaw.utilities import unpack
from ase import Hartree
import pickle
import numpy as npy
from gpaw.mpi import world, MASTER, rank

def tri2full(M,UL='L'):
    """UP='L' => fill upper triangle from lower triangle
       such that M=M^d"""
    nbf = len(M)
    if UL=='L':
        for i in range(nbf-1):
            M[i,i:] = M[i:,i].conjugate()
    elif UL=='U':
        for i in range(nbf-1):
            M[i:,i] = M[i,i:].conjugate()

def get_bf_centers(atoms):
    calc = atoms.get_calculator()
    if not calc.initialized:
        calc.initialize(atoms)
    nbf = calc.nao
    pos_ac = atoms.get_positions()
    #print pos_ac
    natoms = len(pos_ac)
    pos_ic = npy.zeros((nbf,3), npy.float)
    index = 0
    for a in xrange(natoms):
        nao = calc.nuclei[a].get_number_of_atomic_orbitals()
        pos_c = pos_ac[a]
        pos_c.shape = (1,3)
        pos_ic[index:index+nao] = npy.repeat(pos_c, nao, axis=0)
        index += nao
    return pos_ic

def get_realspace_hs(h_skmm,s_kmm, ibzk_kc, weight_k, R_c=(0,0,0)):
    nbf = h_skmm.shape[-1]
    nspins = len(h_skmm)
    h_smm = npy.empty((nspins,nbf,nbf))
    s_mm = npy.empty((nbf,nbf))
    phase_k = npy.dot(2 * npy.pi * ibzk_kc, R_c)
    c_k = npy.exp(1.0j * phase_k) * weight_k
    c_k.shape = (len(ibzk_kc),1,1)
    for s in range(nspins):
        h_smm[s] = npy.sum((h_skmm[s] * c_k).real, axis=0)
    
    s_mm[:] = npy.sum((s_kmm * c_k).real, axis=0)
    return h_smm, s_mm
##
def get_realspace_h(h_skmm, ibzk_kc, weight_k, R_c=(0,0,0)):
    nbf = h_skmm.shape[-1]
    nspins = len(h_skmm)
    h_smm = npy.empty((nspins,nbf,nbf))
    s_mm = npy.empty((nbf,nbf))
    phase_k = npy.dot(2 * npy.pi * ibzk_kc, R_c)
    c_k = npy.exp(1.0j * phase_k) * weight_k
    c_k.shape = (len(ibzk_kc),1,1)
    for s in range(nspins):
        h_smm[s] = npy.sum((h_skmm[s] * c_k).real, axis=0)
    
    return h_smm
#---------

def remove_pbc(atoms, h, s, d=0):
    calc = atoms.get_calculator()
    if not calc.initialized:
        calc.initialize(atoms)
    nbf = calc.nao
    
    cutoff = atoms.get_cell()[d,d] * 0.5 
    pos_i = get_bf_centers(atoms)[:,d]
    for i in xrange(nbf):
        dpos_i = npy.absolute(pos_i - pos_i[i])
        mask_i = (dpos_i < cutoff).astype(npy.int)
        h[i,:] = h[i,:] * mask_i
        h[:,i] = h[:,i] * mask_i
        s[i,:] = s[i,:] * mask_i
        s[:,i] = s[:,i] * mask_i


def dump_lcao_hamiltonian(calc, filename): # "parallel" dump of H
    if calc.master:
        # Dump calulation info on master
        fd = open(filename, 'wb')
        pickle.dump((calc.nao, calc.nspins, calc.ibzk_kc, calc.weight_k), 
                    fd, protocol=2) 
        fd.close()
    world.barrier()
        
    if calc.kpt_comm.rank == 0:
        Vt_skmm = calc.eigensolver.Vt_skmm
        ef = calc.get_fermi_level()
        hamiltonian = calc.hamiltonian
        eigensolver = calc.eigensolver
        hamiltonian.calculate_effective_potential_matrix(Vt_skmm) 
        if calc.domain.comm.rank == 0:
            fd = open(filename, 'ab')
    world.barrier()
    
    if calc.kpt_comm.rank == 0:
        for s in range(calc.nspins):
            for k in range(len(calc.ibzk_kc)):
                h_mm = eigensolver.get_hamiltonian_matrix(hamiltonian,k=0,s=0)
                s_mm = hamiltonian.S_kmm[k]#XXX writes twice when spin!!
                tri2full(h_mm)
                tri2full(s_mm)
                h_mm *= Hartree
                h_mm -= ef * s_mm
                hs_dmm = npy.array((h_mm,s_mm))
                if calc.domain.comm.rank == 0:
                    pickle.dump(hs_dmm, fd, 2)
                
        if calc.domain.comm.rank == 0:
            fd.close()

def load_lcao_hamiltonian(filename): # serial version only
    fd = open(filename, 'rb')
    nao, nspins, ibzk_kc, weight_k = pickle.load(fd)
    nkpt = len(ibzk_kc)
    HS_dskmm = npy.empty((2, nspins, nkpt, nao, nao))

    for s in range(nspins):
        for k in range(nkpt):
            HS_dskmm[:, s, k, :, :] = pickle.load(fd)
        
    fd.close()
    return nao, ibzk_kc, weight_k, HS_dskmm


