#!/usr/bin/env gpaw-python
# -*- coding: utf-8 -*-
from __future__ import print_function
from gpaw import GPAW
from gpaw.utilities.blas import gemm
import sys
import numpy as np
from ase.units import Hartree


def read_arguments():
    """Input Argument Parsing"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help="name of input GPAW file")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-wmin", "--omegamin",default=0., type=float,
                        help="energy range minimum (default: %(default)s eV)")
    parser.add_argument("-wmax", "--omegamax",
                        help="energy range maximum (default: %(default)s eV)",
                        default=5., type=float)
    parser.add_argument("-Dw", "--Domega",
                        help="energy increment (default: %(default)s eV)",
                        default=0.025, type=float)
    parser.add_argument("-eta",
                        help="electronic temperature '
                        '(default: %(default)s eV)", default=0.1, type=float)
    parser.add_argument("-c", "--cutocc",
                        help="cutoff for occupation difference |f_n - f_m| '
                        '(default: %(default)s)", default=1e-5, type=float)
    parser.add_argument("-HT", "--HilbertTransform",
                        help="use Hilbert transform", action="store_true")
    parser.add_argument("-s", "--singlet",
                        help="perform a singlet calculation assuming '
                        's=0 -> s=1 and s=1 -> s=0", action="store_true")
    parser.add_argument("-t", "--transitions",
                        help="output optical transitions", action="store_true")
    args = parser.parse_args()
    return (args.filename, args.omegamax, args.eta, args.Domega,
            args.omegamin, args.verbose, args.cutocc, args.singlet,
            args.HilbertTransform, args.transitions)


def write_absorption(epsilon_qvsw, omega_w, filename='out.gpw', cutocc=1e-5,
                     singlet=False, verbose=False, HilbertTransform=False,
                     spin=None):
    """Write Absorption File"""
    outfilename = filename.split('.gpw')[0]+'_absorption_cut'+str(cutocc)+'.dat'
    if singlet:
        outfilename = outfilename.split('.')[0]+'_singlet.dat'
    if HilbertTransform:
        outfilename = outfilename.split('.')[0]+'_HT.dat'
    if verbose:
        print("Writing", outfilename)
    if spin is None:
        epsilon_qvw = epsilon_qvsw.sum(1)
    else:
        epsilon_qvw = epsilon_qvsw[:,spin,:]
        outfilename = outfilename.split('.dat')[0]+'_S'+str(spin)+'.dat'
    f = open(outfilename, 'w')
    for dw in range(len(omega_w)):
        print(omega_w[dw], epsilon_qvw[0,dw], epsilon_qvw[1,dw],
              epsilon_qvw[2,dw], file=f)
    f.close()

def write_transitions(T_qvsnm, omega_snm, Deltaf_snm, filename='out.gpw',
                      cutocc=1e-5, singlet=False, verbose=False,
                      HilbertTransform=False, omegamin=0, omegamax=5.):
    """Write Transitions File"""
    axes = {0 : 'x', 1 : 'y', 2 : 'z'}
    outfilename = filename.split('.gpw')[0]+'_transitions_cut'+str(cutocc)
    if singlet:
        outfilename += '_singlet'
    if HilbertTransform:
        outfilename += '_HT'
    number_of_spins = T_qvsnm.shape[1]
    nbands = T_qvsnm.shape[-1]
    for spin in range(number_of_spins):
        for dq in range(3):
            outfilenamesq = outfilename+'_S'+str(spin)+'_'+axes[dq]+'.dat'
            if verbose:
                print("Writing", outfilenamesq)
            f = open(outfilenamesq, 'w')
            for n in range(0, nbands-1):
                for m in range(n+1, nbands):
                    if (omegamin < omega_snm[spin,n,m]
                        and omega_snm[spin,n,m] < omegamax
                        and abs(T_qvsnm[dq,spin,n,m]) > 1e-3):
                        if verbose and abs(T_qvsnm[dq,spin,n,m]) > 0.1:
                            print(n, "->", m, T_qvsnm[dq,spin,n,m],
                                  omega_snm[spin,n,m], "s =", spin, axes[dq],
                                  Deltaf_snm[spin,n,m])
                        print(omega_snm[spin,n,m], T_qvsnm[dq,spin,n,m],
                              "\""+str(n)+" -> "+str(m)+"\"",
                              Deltaf_snm[spin,n,m], file=f)
            f.close()

def get_dThetadR(calc):
    """Calculate grad_qvMM matrix dThetadR of LCAO orbitals"""
    spos_ac = calc.wfs.tci.atoms.get_scaled_positions() % 1.0
    ksl = calc.wfs.ksl
    nao = ksl.nao
    mynao = ksl.mynao
    nq = len(calc.wfs.kd.ibzk_qc)
    dtype = calc.wfs.dtype
    tci = calc.wfs.tci
    gd = calc.wfs.gd
    bfs = calc.wfs.basis_functions
    dThetadR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
    dTdR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
    dPdR_aqvMi = {}
    for a in calc.wfs.basis_functions.my_atom_indices:
        ni = calc.wfs.setups[a].ni
        dPdR_aqvMi[a] = np.empty((nq, 3, nao, ni), dtype)
    tci.calculate_derivative(spos_ac, dThetadR_qvMM, dTdR_qvMM, dPdR_aqvMi)
    gd.comm.sum(dThetadR_qvMM)
    return dThetadR_qvMM


def get_P_ani(calc, spin=0, k=0):
    """Obtain Projector Matrix"""
    kd = calc.wfs.kd
    ibzkpt = kd.bz2ibz_k[k]
    u = ibzkpt + kd.nibzkpts * spin
    kpt = calc.wfs.kpt_u[u]
    s = calc.wfs.kd.sym_k[k]
    P_ani = {}
    for a, id in enumerate(calc.wfs.setups.id_a):
        b = calc.wfs.kd.symmetry.a_sa[s, a]
        P_ani[a] = np.dot(kpt.P_ani[b], calc.wfs.setups[a].R_sii[s])
    return P_ani

def get_PAW_Omega(calc, spin1=0, spin2=0, k=0):
    """Calculate PAW Corrections to Omega_qvnm"""
    P_ani = get_P_ani(calc, spin1, k)
    # PAW corrections from other spin channel
    P_ami = get_P_ani(calc, spin2, k)
    nbands = calc.get_number_of_bands()
    PAW_Omega_qvnm = np.zeros((3, nbands, nbands))
    nbands = calc.get_number_of_bands()
    nocc = int((calc.get_number_of_electrons()
                + calc.input_parameters['charge']) // 2)
    setups = calc.wfs.setups
    for dq in range(3):
        for a, id in enumerate(calc.wfs.setups.id_a):
            #tmp = np.dot(setups[a].nabla_iiv[:,:,dq], P_ani[a].transpose())
            #gemm(1.0, P_ani[a], tmp, 1.0, PAW_Omega_qvnm[dq])
            # Take PAW corrections for n >= nocc from other spin channel
            # if different
            if not spin1 == spin2:
                P_ani[a][nocc:,:] = P_ami[a][nocc:,:]
            PAW_Omega_qvnm[dq,:,:] += \
                np.dot(P_ani[a], np.dot(setups[a].nabla_iiv[:,:,dq],
                                        P_ani[a].transpose()))
    return PAW_Omega_qvnm

def get_A_DeltaE(calc, dThetadR_qvMM, cutocc=1e-5, singlet=False,
                 verbose=True):
    """Calculate Matrix Elements A and Energy Differences DeltaE
    dThetaR_qvMM	Gradients of Basis Functions"""
    nbands = calc.get_number_of_bands()  # total number of bands
    # number of occupied bands in ground state:
    nocc = int((calc.get_number_of_electrons()
                + calc.input_parameters['charge'])//2)
    cell = calc.wfs.gd.cell_cv  # unit cell in Bohr^3
    # unit cell volume in Bohr^3:
    volume = np.abs(np.dot(cell[0], np.cross(cell[1],cell[2])))
    if verbose:
        axes = {0 : 'x', 1 : 'y', 2 : 'z'}
    prefactor = 4 * np.pi / volume  # prefactor for A_nm in chi0
    number_of_spins = calc.get_number_of_spins()  # number of spins
    # overlap matrix:
    A_qvsnm = np.zeros((3, number_of_spins, nbands, nbands))
    # energy difference matrix:
    DeltaE_snm = np.zeros((number_of_spins, nbands, nbands))
    # occupancy difference matrix:
    Deltaf_snm = np.zeros((number_of_spins, nbands, nbands))
    I_nm = np.identity(nbands)  # identity matrix
    for spin in range(number_of_spins):
        eigenvalues_n = calc.get_eigenvalues(spin=spin).copy()/Hartree
        occupations_n = calc.get_occupation_numbers(spin=spin).copy()
        C_nM = calc.wfs.kpt_u[spin].C_nM
        # Perform a pseudo-singlet calculation based on a triplet calculation
        # by swapping spin channels for n >= nocc
        if singlet:
            spin2 = np.mod(spin+1, number_of_spins) # other spin channel
            # Take eigenvalues for n >= nocc from other spin channel
            eigenvalues_m = calc.get_eigenvalues(spin=spin2).copy()/Hartree
            eigenvalues_n[nocc:] = eigenvalues_m[nocc:]
            # Take occupations for n >= nocc from other spin channel
            occupations_m = calc.get_occupation_numbers(spin=spin2).copy()
            occupations_n[nocc:] = occupations_m[nocc:]
            # Take LCAO coefficients for n >= nocc from other spin channel]
            C_mM = calc.wfs.kpt_u[spin2].C_nM
            C_nM[nocc:] = C_mM[nocc:]
        else:
            spin2 = spin
        if verbose:
            print("Calculating PAW Corrections")
        PAW_Omega_qvnm = get_PAW_Omega(calc, spin1=spin, spin2=spin2)
        DeltaE_snm[spin] = np.outer(eigenvalues_n, np.ones(nbands))
        - np.outer(np.ones(nbands), eigenvalues_n)
        Deltaf_snm[spin] =  -np.outer(occupations_n, np.ones(nbands)) \
            + np.outer(np.ones(nbands), occupations_n)
        if cutocc:
            if verbose:
                print("Applying Cutoff |f_n - f_m| >", cutocc)
            Deltaf_snm[spin,:,:] = (abs(Deltaf_snm[spin,:,:])
                                    > cutocc) * Deltaf_snm[spin,:,:]
        gradC_Mm = np.zeros(C_nM.shape)
        for dq in range(3):
            Omega_nm = PAW_Omega_qvnm[dq]
            if verbose:
                print('Starting spin', spin, 'along', axes[dq], 'axis')
            gemm(1.0, dThetadR_qvMM[0,dq], C_nM, 0.0, gradC_Mm)
            gemm(1.0, C_nM, gradC_Mm, 1.0, Omega_nm, 't')
            Omega_nm /=  DeltaE_snm[spin] + I_nm
            A_qvsnm[dq,spin] = prefactor * Deltaf_snm[spin] * Omega_nm**2
    return A_qvsnm, DeltaE_snm, Deltaf_snm

def get_T(A_qvsnm, DeltaE_snm, eta=0.1):
    """Calculate Optical Transition Intensities at DeltaE"""
    T_qvsnm = np.zeros(A_qvsnm.shape)
    eta /= Hartree
    #A_qvsnm *= eta
    omega_snm = abs(DeltaE_snm)
    for dq in range(3):
        T_qvsnm[dq] = eta * A_qvsnm[dq] * \
            ( 1 / ((omega_snm - DeltaE_snm)**2 + eta**2) - 1
              / ((omega_snm + DeltaE_snm)**2 + eta**2))
    return T_qvsnm, omega_snm * Hartree

def get_epsilon(A_qvsnm, DeltaE_snm, eta=0.1, omegamax=5.0, Domega=0.025,
                omegamin=0.0, HilbertTransform=False):
    """Calculate Optical Absorption Imaginary Epsilon
    A_qvsnm      Matrix Elements
    DeltaE_snm     Energy Differences"""
    eta /= Hartree
    #A_qvsnm *= eta
    omega_w = np.arange(omegamin/Hartree, omegamax/Hartree, Domega/Hartree)
    epsilon_qvsw = np.zeros((3, A_qvsnm.shape[1], len(omega_w)))
    iu = np.triu_indices(A_qvsnm.shape[-1],1)
    for dq in range(3):
        for dw in range(len(omega_w)):
            for spin in range(A_qvsnm.shape[1]):
                if HilbertTransform:
                    epsilon_qvsw[dq, spin, dw] += eta * (A_qvsnm[dq,spin][iu] \
                        * (1 / ((omega_w[dw] - DeltaE_snm[spin][iu])**2 \
                        + eta**2) - 1 / ((omega_w[dw] + \
                        DeltaE_snm[spin][iu])**2 + eta**2))).sum()
                else:
                    epsilon_qvsw[dq, spin, dw] = eta * \
                        (A_qvsnm[dq, spin] / ((omega_w[dw] \
                        - DeltaE_snm[spin])**2 + eta**2)).sum()
    return epsilon_qvsw, omega_w * Hartree

if __name__ == '__main__':
    # Read Arguments
    (filename, omegamax, eta, Domega, omegamin, verbose, cutocc, singlet,
     HilbertTransform, transitions) = read_arguments()
    if verbose:
        print("Calculating Absorption Function from", omegamin,
              "to", omegamax, "eV in increments of ", Domega, "eV")
        print("Electronic Temperature", eta, "eV")
        print("|f_n - f_m| >", cutocc)
        print("Opening", filename)
    calc = GPAW(filename, txt=None)
    atoms = calc.get_atoms()
    if verbose:
        print("Initializing Positions")
    calc.initialize_positions(atoms)
    if verbose:
        print("Calculating Basis Function Gradients")
    dThetadR_qvMM = get_dThetadR(calc)
    if verbose:
        print("Calculating Matrix Elements")
    A_qvsnm, DeltaE_snm, Deltaf_snm = get_A_DeltaE(calc, dThetadR_qvMM,
                                                   cutocc, singlet, verbose)
    if transitions:
        if verbose:
            print("Calculating Optical Transitions")
        T_qvsnm, omega_snm = get_T(A_qvsnm, DeltaE_snm, eta)
        write_transitions(T_qvsnm, omega_snm, Deltaf_snm, filename, cutocc,
                          singlet, verbose, HilbertTransform, omegamin,
                          omegamax)
    if verbose:
        print("Calculating Optical Absorption")
        if HilbertTransform:
            print("Using Hilbert Transform")
    epsilon_qvsw, omega_w = get_epsilon(A_qvsnm, DeltaE_snm, eta, omegamax,
                                        Domega, omegamin, HilbertTransform)
    write_absorption(epsilon_qvsw, omega_w, filename, cutocc, singlet,
                     verbose, HilbertTransform)
    if calc.get_number_of_spins() == 2:
        write_absorption(epsilon_qvsw, omega_w, filename, cutocc, singlet,
                         verbose, HilbertTransform, spin=0)
        write_absorption(epsilon_qvsw, omega_w, filename, cutocc, singlet,
                         verbose, HilbertTransform, spin=1)
