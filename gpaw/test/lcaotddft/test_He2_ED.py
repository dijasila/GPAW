from gpaw import GPAW
from ase.units import _amu, _me, Bohr, AUT, Hartree
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from ase.io import Trajectory
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from gpaw.lcaotddft import LCAOTDDFT
from ase.build import graphene, add_adsorbate
from gpaw.test import equal


def test_ehrenfest_LCAO():
    kwargs = dict(mode='lcao', basis='sz(dzp)', symmetry='off',
                  convergence={'density': 1e-12}, xc='PBE')
    atoms = graphene(formula='He2', a=2.460, size=(1, 1, 1))
    add_adsorbate(atoms, 'He', 2.0, [0.0, 1.3])
    atoms.center(vacuum=2, axis=2)
    atoms.center()
    atoms.pbc = (True)
    kpts_no_gamma = [[0, 0, 0], [0, 0, 0.25], [0, 0, -0.25], [0, 0, 0.5]]
    kpts_gamma = {'size': (2, 2, 1), 'gamma': True}

    def single(atoms, kpts_gamma=kpts_gamma):
        calc = GPAW(**kwargs,
                    gpts=(16, 16, 16),
                    txt='1_gpaw.kpts.txt',
                    kpts=kpts_gamma)
        atoms.calc = calc
        E = atoms.get_potential_energy()/len(atoms)
        calc.write('1_gs.gpw', mode='all')
        return E

    def double(atoms):
        atoms = atoms * (2, 2, 1)
        calc = GPAW(**kwargs,
                    gpts=(32, 32, 16),
                    txt='2_gpaw.kpts.txt',
                    kpts=[[0, 0, 0]])
        atoms.calc = calc
        E = atoms.get_potential_energy() / len(atoms)
        calc.write('2_gs.gpw', mode='all')
        return E

    e1 = single(atoms)
    e2 = double(atoms)
    print(e1)
    print(e2)
    # print('error', e1 - e2)

    def ehrenfest_run(inp_gs):
        np.set_printoptions(precision=15, suppress=1, linewidth=180)
        traj_file = inp_gs + '.traj'
        Ekin = 80
        timestep = 32.0  # *0.2 #* np.sqrt(10/Ekin)
        amu_to_aumass = _amu/_me
        tdcalc = LCAOTDDFT(inp_gs, propagator='edsicn',
                           #PLCAO_flag=False,
                           Ehrenfest_flag=True,
                           txt=inp_gs + '_out_td.txt',  S_flag=True)
                           #Ehrenfest_force_flag=False)
        tdcalc.tddft_init()
        f = open(inp_gs + '_td.log', 'w')
        proj_idx = 1
        pos = tdcalc.atoms.get_positions()
        # v = np.zeros((proj_idx+1,3))
        v = np.zeros((len(pos[:, 0]), 3))
        Mproj = tdcalc.atoms.get_masses()[proj_idx]
        # Ekin *= Mproj
        Ekin = Ekin / Hartree
        Mproj *= amu_to_aumass
        for m in np.arange(0, len(v[:, 0]))[2::3]:
            v[m, 2] = -np.sqrt((2*Ekin)/Mproj) * Bohr / AUT
        tdcalc.atoms.set_velocities(v)
        evv = EhrenfestVelocityVerlet(tdcalc)
        traj = Trajectory(traj_file, 'w', tdcalc.get_atoms())
        ndiv = 1
        # NUMBER OF STEPS
        niters = 5

        for i in range(niters):
            print("STEP MD i", i)
            Ekin_p = 0.5*evv.M[proj_idx]*(evv.velocities[proj_idx, 0]**2
                                          + evv.velocities[proj_idx, 1]**2
                                          + evv.velocities[proj_idx, 2]**2)
            if i % ndiv == 0:
                # nLow, n, nLowm, nm, eigv, nLow_kM = calculate_norm(tdcalc,
                #                                                   tdcalc.wfs,
                #                                                   inp_gs)
                # Etot = evv.get_energy()
                F_av = evv.Forces * Hartree / Bohr
                epot = tdcalc.get_td_energy() * Hartree
                ekin = tdcalc.atoms.get_kinetic_energy()
                T = i * timestep
                ep = Ekin_p * Hartree
                f.write('%6.2f %18.10f %18.10f %18.10f %18.10f \n' % (T,
                        ep, ekin, epot, ekin+epot))
                f.flush()
                epot = tdcalc.get_td_energy() * Hartree
                spa = tdcalc.get_atoms()
                spc = SinglePointCalculator(epot)
                spa.set_calculator(spc)
                traj.write(spa)
            evv.propagate(timestep)
        traj.close()
        return F_av, ekin+epot

    inp_gs = ['1_gs.gpw', '2_gs.gpw']

    F0 = []
    E0 = []
    # compare one gamma with supercell gammas
    for i in inp_gs:
        print('==========', i, '==========')
        print('===========================')
        F, E = ehrenfest_run(i)
        F0.append(F[0])
        E0.append(E)
        print('===========================')
    # print('F error',np.abs(F[0]-F[1]).max())
    print('F error', np.abs(F0[0]-F0[1]).max())
    print('EEEEE', E0[:])
    print('FFFFF', F0[:])
    assert np.abs(F0[0]-F0[1]).max() < 1.0e-11

    equal(F0[0][0], -0.00025, 1e-5)
    equal(F0[1][0], -0.00025, 1e-5)
    equal(E0[0], 82.68792240168231, 1e-12)
    equal(E0[1], 330.75168960672966, 1e-12)
    # check energy and forces for non gamma points
    single(atoms, kpts_gamma=kpts_no_gamma)

    F_no_gamma, etot_no_gamma = ehrenfest_run('1_gs.gpw')
    np.set_printoptions(precision=15, suppress=1, linewidth=180)
    print('F_no_gamma etot_no_gamma', F_no_gamma[0, :], etot_no_gamma)

    equal(etot_no_gamma, 82.7482268662977, 1e-12)
    equal(F_no_gamma[0, 0], -0.00375, 1e-5)

