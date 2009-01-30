# Copyright (C) 2008  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module is used in delta self-consistent field (dSCF) calculations

dSCF is a simple 'ad hoc' method to estimating excitation energies within
DFT. The only difference to ordinary DFT is that one or more electrons(s)
are forced to occupy one or more predefined orbitals. The only restriction
on these orbitals is that they must be linear combinations of available
Kohn-Sham orbitals.

"""

import copy
import numpy as np
from gpaw.occupations import OccupationNumbers, FermiDirac
import gpaw.mpi as mpi

def dscf_calculation(calc, orbitals, atoms=None):
    """Helper function to prepare a calculator for a dSCF calculation

    Parameters
    ==========
    orbitals: list of lists
        Orbitals which one wants to occupy. The format is
        orbitals = [[1.0,orb1,0],[1.0,orb2,1],...], where 1.0 is the no.
        of electrons, orb1 and orb2 are the orbitals (see MolecularOrbitals
        below for an example of an orbital class). 0 and 1 represents the
        spin (up and down). This number is ignored in a spin-paired
        calculation.

    Example
    =======
    
    >>> atoms.set_calculator(calc)
    >>> e_gs = atoms.get_potential_energy() #ground state energy
    >>> sigma_star=MolecularOrbitals(calc, molecule=[0,1],
    >>>                              w=[[1.,0.,0.,0.],[-1.,0.,0.,0.]])
    >>> dscf_calculation(calc, [[1.0,sigma_star,1]], atoms)
    >>> e_exc=atoms.get_potential_energy() #excitation energy

    """

    # if the calculator has not been initialized it does not have an
    # occupation object
    if not hasattr(calc, 'occupation'):
        calc.initialize(atoms)
    occ = calc.occupations
    if occ.kT == 0:
        occ.kT = 1e-6
    if isinstance(occ, OccupationsDSCF):
        calc.occupations.orbitals = orbitals
    else:
        new_occ = OccupationsDSCF(occ.ne, occ.nspins, occ.kT, orbitals, calc)
        new_occ.set_communicator(occ.kpt_comm)
        calc.occupations = new_occ
        calc.converged = False

class OccupationsDSCF(FermiDirac):
    """Occupation class.

    Corresponds to the ordinary FermiDirac class in occupation.py. Only
    difference is that it forces some electrons in the supplied orbitals
    in stead of placing all the electrons by a Fermi-Dirac distribution.
    """

    def __init__(self, ne, nspins, kT, orbitals, paw):
        FermiDirac.__init__(self, ne, nspins, kT)
        
        self.orbitals = orbitals
        self.paw=paw

        self.cnoe = 0.
        for orb in orbitals:
            self.cnoe += orb[0]
        self.ne -= self.cnoe

    def calculate_band_energy(self, kpt_u):
        # Sum up all eigenvalues weighted with occupation numbers:
        Eband = 0.0
        for kpt in kpt_u:
            Eband += np.dot(kpt.f_n, kpt.eps_n)
            if hasattr(kpt, 'ft_omn'):
                for i in range(len(kpt.ft_omn)):
                    Eband += np.dot(np.diagonal(kpt.ft_omn[i]).real,
                                    kpt.eps_n)
        self.Eband = self.kpt_comm.sum(Eband)

    def calculate(self, kpts):
        OccupationNumbers.calculate(self, kpts)

        if self.epsF is None:
             #Fermi level not set.  Make a good guess:
             self.guess_fermi_level(kpts)
        # Now find the correct Fermi level for the non-controlled electrons
        self.find_fermi_level(kpts)

        # Get the expansion coefficients for the dscf-orbital(s)
        # and create the density matrices, kpt.ft_mn
        ft_okm = []
        for orb in self.orbitals:
            ft_okm.append(orb[1].get_ft_km(self.epsF))
            
        for u, kpt in enumerate(self.paw.wfs.kpt_u):
            kpt.ft_omn = np.zeros((len(self.orbitals),
                                    len(kpt.f_n), len(kpt.f_n)), np.complex)
            for o in range(len(self.orbitals)):
                ft_m = ft_okm[o][u]
                for n1 in range(len(kpt.f_n)):
                     for n2 in range(len(kpt.f_n)):
                         kpt.ft_omn[o,n1,n2] = (self.orbitals[o][0] *
                                                ft_m[n1] *
                                                np.conjugate(ft_m[n2]))

                if self.nspins == 2 and self.orbitals[o][2] == kpt.s:
                    kpt.ft_omn[o] *= kpt.weight
                elif self.nspins == 2 and self.orbitals[o][2] < 2:
                    kpt.ft_omn[o] *= 0.0
                else:
                    kpt.ft_omn[o] *= 0.5 * kpt.weight

            #print np.diagonal(kpt.ft_omn[0]).real
        
        S = 0.0
        for kpt in kpts:
            if self.fixmom:
                x = np.clip((kpt.eps_n - self.epsF[kpt.s]) / self.kT,
                             -100.0, 100.0)
            else:
                x = np.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
            y = np.exp(x)
            z = y + 1.0
            y *= x
            y /= z
            y -= np.log(z)
            S -= kpt.weight * np.sum(y)

        self.S = self.kpt_comm.sum(S) * self.kT
        self.calculate_band_energy(kpts)

        for orb in self.orbitals:
            if orb[2] == 0:
                self.magmom += orb[0]
            elif orb[2] == 1:
                self.magmom -= orb[0]
        
class MolecularOrbital:
    """Class defining the orbitals that should be filled in a dSCF calculation.
    
    An orbital is defined through a linear combination of the atomic
    projector functions. In each self-consistent cycle the method get_ft_km
    is called. This method take the Kohn-Sham orbittals forfilling the
    criteria given by Estart, Eend and no_of_states and return the best
    possible expansion of the orbital in this basis.

    Parameters
    ----------
    paw: gpaw calculator instance
        The calculator used in the dSCF calculation.
    molecule: list of integers
        The atoms, which are a part of the molecule.
    Estart: float
        Kohn-Sham orbitals with an energy above Efermi+Estart are used
        in the linear expansion.
    Eend: float
        Kohn-Sham orbitals with an energy below Efermi+Eend are used
        in the linear expansion.
    no_of_states: int
        The maximum number of Kohn-Sham orbitals used in the linear expansion.
    w: list
        The weights of the atomic projector functions.
        Format::

          [[weight of 1. projector function of the 1. atom,
            weight of 2. projector function of the 1. atom, ...],
           [weight of 1. projector function of the 2. atom,
            weight of 2. projector function of the 2. atom, ...],
           ...]
    """

    def __init__(self, paw, molecule=[0,1], Estart=0.0, Eend=1.e6,
                 no_of_states=None, w=[[1.,0.,0.,0.],[-1.,0.,0.,0.]]):
        
        self.paw = paw
        self.mol = molecule
        self.w = w
        self.Estart = Estart
        self.Eend = Eend
        self.nos = no_of_states

    def get_ft_km(self, epsF):

        # get P_uni from the relevent nuclei
        P_auni = [[kpt.P_ani[a] for kpt in self.paw.wfs.kpt_u]
                  for a in self.mol]

        if self.paw.wfs.nspins == 1:
            epsF = [epsF]
        elif not self.paw.input_parameters.fixmom:
            epsF = [epsF, epsF]
            
        if self.nos == None:
            self.nos = len(self.paw.wfs.kpt_u[0].f_n)

        ft_km = []
        for u, kpt in enumerate(self.paw.wfs.kpt_u):
            Porb_n = np.zeros(len(kpt.f_n), np.complex)
            for atom in range(len(self.mol)):
                for pw_no in range(len(self.w[atom])):
                    Porb_n += (self.w[atom][pw_no] *
                               np.swapaxes(P_auni[atom][u], 0, 1)[pw_no])
            Porb_n = np.swapaxes(Porb_n, 0, 1)
            Pabs_n = abs(Porb_n)**2
            argsort = np.argsort(Pabs_n)

            ft_m = np.zeros(len(kpt.f_n), np.complex)
            nosf = 0
            for m in argsort[::-1]:
                if (kpt.eps_n[m] > epsF[kpt.s] + self.Estart and
                    kpt.eps_n[m] < epsF[kpt.s] + self.Eend):
                    ft_m[m] = Porb_n[m]
                    nosf += 1
                if nosf == self.nos:
                    break

            ft_m /= np.sqrt(sum(abs(ft_m)**2))
            ft_km.append(ft_m)
        return ft_km
                    
class AEOrbital:
    """Class defining the orbitals that should be filled in a dSCF calculation.
    
    An orbital is defined through a linear combination of KS orbitals
    which is determined by this class as follows: For each kpoint we
    calculate the quantity ``ft_m = <m|a>`` where ``|m>`` is the
    all-electron KS states in the calculation and ``|a>`` is the
    all-electron resonant state to be kept occupied.  We can then
    write ``|a> = Sum(ft_m|m>)`` and in each self-consistent cycle the
    method get_ft_km is called. This method take the Kohn-Sham
    orbitals fulfilling the criteria given by Estart, Eend and
    no_of_states and return the best possible expansion of the orbital
    in this basis.

    Parameters
    ----------
    paw: gpaw calculator instance
        The calculator used in the dSCF calculation.
    molecule: list of integers
        The atoms, which are a part of the molecule.
    Estart: float
        Kohn-Sham orbitals with an energy above Efermi+Estart are used
        in the linear expansion.
    Eend: float
        Kohn-Sham orbitals with an energy below Efermi+Eend are used
        in the linear expansion.
    no_of_states: int
        The maximum number of Kohn-Sham orbitals used in the linear expansion.
    wf_u: list of wavefunction arrays
        Wavefunction to be occupied on the kpts on this processor.
    P_aui: list of two-dimensional arrays.
        [Calulator.nuclei[a].P_uni[:,n,:] for a in molecule]
        Projector overlaps with the wavefunction to be occupied for each
        kpoint. These are used when correcting to all-electron wavefunction
        overlaps.
    """

    def __init__(self, paw, wf_u, P_aui, Estart=0.0, Eend=1.e6,
                 molecule=[0,1], no_of_states=None):
    
        self.paw = paw
        self.wf_u = wf_u
        self.P_aui = P_aui
        self.Estart = Estart
        self.Eend = Eend
        self.mol = molecule
        self.nos = no_of_states

    def get_ft_km(self, epsF):
        kpt_u = self.paw.wfs.kpt_u
        if self.paw.wfs.nspins == 1:
            epsF = [epsF]
        elif not self.paw.input_parameters.fixmom:
            epsF = [epsF, epsF]
        if self.nos == None:
            self.nos = len(kpt_u[0].f_n)

        if len(self.wf_u) == len(kpt_u):
            wf_u = self.wf_u
            P_aui = self.P_aui
        else:
            raise RuntimeError('List of wavefunctions has wrong size')

        if kpt_u[0].psit_nG is None:
            return np.zeros((len(kpt_u), self.paw.nbands), float)
              
        ft_un = []
        
        for u, kpt in enumerate(kpt_u):

            # Inner product of pseudowavefunctions
            wf = np.reshape(wf_u[u], -1)
            Wf_n = kpt.psit_nG
            Wf_n = np.reshape(Wf_n, (len(kpt.f_n), -1))
            p_n = np.dot(Wf_n.conj(), wf) * self.paw.gd.dv
            
            # Correction to obtain inner product of AE wavefunctions
            P_ani = [kpt.P_ani[a] for a in self.mol]
            for P_ni, a, b in zip(P_ani, self.mol, range(len(self.mol))):
                for n in range(self.paw.wfs.nbands):
                    for i in range(len(P_ni[0])):
                        for j in range(len(P_ni[0])):
                            p_n[n] += (P_ni[n][i].conj() *
                                       self.paw.wfs.setups[a].O_ii[i][j] *
                                       P_aui[b][u][j])

##             self.paw.gd.comm.sum(p_n)

            #print abs(p_n)**2
            print 'Kpt:', kpt.k, ' Spin:', kpt.s, \
                  ' Sum_n|<orb|nks>|^2:', sum(abs(p_n)**2)
            
            if self.paw.wfs.dtype == float:
                ft_n = np.zeros(len(kpt.f_n), np.float)
            else:
                ft_n = np.zeros(len(kpt.f_n), np.complex)

            argsort = np.argsort(abs(p_n)**2)
            nosf = 0
            for m in argsort[::-1]:
                if (kpt.eps_n[m] > epsF[kpt.s] + self.Estart and
                    kpt.eps_n[m] < epsF[kpt.s] + self.Eend):
                    ft_n[m] = p_n[m].conj()
                    nosf += 1
                if nosf == self.nos:
                    break

            ft_n /= np.sqrt(sum(abs(ft_n)**2))
            
            ft_un.append(ft_n)
            
        return ft_un
