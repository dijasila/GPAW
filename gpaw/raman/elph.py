# coding: utf-8

#General
import numpy as np
import sys
from math import pi
from os.path import isfile, join
from pathlib import Path

# GPAW/ASE
from gpaw import GPAW, PW, FermiDirac
from gpaw.elph.electronphonon import ElectronPhononCoupling
from ase.parallel import rank, size, world, MPI4PY
from ase.io import read, write
from ase.parallel import rank, size
from ase.phonons import Phonons

def elph(atoms, params_fd, sc = (1,1,1)):
    """
        Finds the forces and effective potential at different atomic positions used to calculate the change in the effective potential.

        Input
        ----------
        params_fd : Calculation parameters used for the phonon calculation
        sc (tuple): Supercell, default is (1,1,1) used for gamma phonons

        Output
        ----------
        elph.*.pckl files with the effective potential
        phonons.*.pckl files with the atomic forces.

    """
    par = MPI4PY()

    calc_fd = GPAW(**params_fd)

    #Calculate the forces and effective potential at different nuclear positions
    elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell = sc, calculate_forces = True)
    elph.run()

    par.comm.Barrier()

    elph = ElectronPhononCoupling(atoms, supercell = sc, calc = calc_fd)
    elph.set_lcao_calculator(calc_fd)
    elph.calculate_supercell_matrix(dump=1)

def get_elph_elements(atoms, params_fd, sc = (1,1,1), basename = None):
    """
        Evaluates the dipole transition matrix elements

        Input
        ----------
        params_fd : Calculation parameters used for the phonon calculation
        sc (tuple): Supercell, default is (1,1,1) used for gamma phonons
        basename  : If you want give a specific name (gqklnn_{}.pckl)

        Output
        ----------
        gqklnn.pckl, the electron-phonon matrix elements
    """

    par = MPI4PY()
    par.comm.Barrier()
    if basename is None:
        calc_gs = GPAW('gs.gpw')
    else:
        calc_gs = GPAW('gs_{}.gpw'.format(basename))

    calc_fd = GPAW(**params_fd)
    calc_gs.initialize_positions(atoms)
    kpts = calc_gs.get_ibz_k_points()
    nk = len(kpts)
    gamma_kpt = [[0,0,0]]
    nbands = calc_gs.wfs.bd.nbands
    qpts = gamma_kpt


    #Phonon calculation, We'll read the forces from the elph.run function
    #This only looks at gamma point phonons
    ph = Phonons(atoms = atoms, name="phonons", supercell = sc)
    ph.read()
    frequencies, modes = ph.band_structure(qpts, modes=True)

    if rank == 0:
        print("Phonon frequencies are loaded.")

    #Find el-ph matrix in the LCAO basis
    elph = ElectronPhononCoupling(atoms, calc=None, supercell = sc)

    elph.set_lcao_calculator(calc_fd)
    elph.load_supercell_matrix(basis = "dzp")
    if rank == 0:
        print("Supercell matrix is loaded")

    #Find the bloch expansion coefficients
    kpt_comm = calc_gs.wfs.kd.comm
    c_kn = np.zeros((nk,nbands, nbands), dtype = complex)

    for k in range(len(kpts)):
        c_k = calc_gs.wfs.collect_array("C_nM",k,0)
        if rank == 0:
            c_kn[k] = c_k

    par.comm.Bcast(c_kn, root = 0)

    #And we finally find the electron-phonon coupling matrix elements!
    g_qklnn = elph.bloch_matrix(c_kn = c_kn, kpts = kpts, qpts = qpts, u_ql = modes)
    if rank == 0:
        print("Saving the elctron-phonon coupling matrix")
        if basename is None:
            np.save("gqklnn.npy", np.array(g_qklnn))
        else:
            np.save("gqklnn_{}.npy".format(basename), np.array(g_qklnn))

if __name__ == '__main__':
    from parameters import *
    par = MPI4PY()

    #Start by doing a groundstate calculation, if that has not been done earlier
    if not Path('gs.gpw').is_file():
        calc_gs = GPAW(**params_gs)
        atoms.calc = calc_gs
        atoms.get_potential_energy()
        atoms.calc.write("gs.gpw", mode="all")
    else:
        calc_gs = GPAW('gs.gpw')

    #The real space grid of the two calculators should match.
    params_fd['gpts'] = calc_gs.wfs.gd.N_c * [sc[0], sc[1], sc[2]]

    #The change in the forces and the effective potential are calculated
    par.comm.Barrier()
    elph(atoms, params_fd, sc = sc)

    #And the electron-phonon coupling terms are evaluated
    par.comm.Barrier()
    get_elph_elements(atoms, params_fd, sc = sc)
