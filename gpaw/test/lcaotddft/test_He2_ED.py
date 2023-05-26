from gpaw import GPAW
from ase import Atoms
from ase.units import _amu, _me, Bohr, AUT, Hartree
from gpaw.tddft import TDDFT
from ase.parallel import paropen
from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from gpaw.mpi import world
import numpy as np
from ase.units import Hartree, Bohr, AUT
from ase.io import Trajectory
from ase.parallel import parprint
import numpy as np
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from gpaw.lcaotddft import LCAOTDDFT
from ase.io import read , write
from scipy.linalg import eigh
from scipy.linalg import norm
from scipy.linalg import schur, eigvals, inv
from get_norm2 import calculate_norm

kwargs = dict(mode='lcao', basis='dzp', symmetry='off',convergence={'density': 1e-12}, xc='PBE')

d_bond=0.8

atoms = Atoms('He2', positions=[(0, 0, 0) , (0, 0.5, 1.0 )] , pbc=True ,cell=[1, 1, 1])
atoms.center(vacuum=1.0)
atoms.positions[:,2]-=0.940
def single(atoms):
    calc = GPAW(**kwargs,
                gpts=(16,16,16),
                txt='1_gpaw.kpts.txt',
                kpts= [[0, 0, 0],[0,0,0.25],[0,0,-0.25], [0, 0, 0.5]])
    atoms.calc = calc
    E = atoms.get_potential_energy() / len(atoms)
    #randomize_C(atoms)
    calc.write('1_gs.gpw', mode='all')

    return E

def double(atoms):
    atoms = atoms * (1, 1, 4)
    calc = GPAW(**kwargs,
                gpts=(16,16,64),
                txt='2_gpaw.kpts.txt',
                kpts=[[0, 0, 0]])

    atoms.calc = calc
    E = atoms.get_potential_energy() / len(atoms)
    #randomize_C(atoms)
    #for kpt in atoms.calc.wfs.kpt_u:
    #    print(kpt.C_nM.shape )
    #    Ar=np.random.rand(len(kpt.C_nM[0,:]),len(kpt.C_nM[:,0]))
    #    Ai=np.random.rand(len(kpt.C_nM[0,:]),len(kpt.C_nM[:,0]))
    #    Q, R = np.linalg.qr(Ar)
    #    kpt.C_nM[:,:]= Q @ kpt.C_nM

    calc.write('2_gs.gpw', mode='all')
    return E

def randomize_C(atoms):
    for kpt in atoms.calc.wfs.kpt_u:
        print(kpt.C_nM.shape )
        Ar=np.random.rand(len(kpt.C_nM[0,:]),len(kpt.C_nM[:,0]))
        Ai=np.random.rand(len(kpt.C_nM[0,:]),len(kpt.C_nM[:,0]))
        Q, R = np.linalg.qr(Ar + 1j*Ai)
        kpt.C_nM[:,:]= Q @ kpt.C_nM


e1 = single(atoms)
print(e1)

e2 = double(atoms)
print(e2)
print('error', e1 - e2)

############################
############################
############################

name = 'He2'

#f = paropen(strbody + '_td.log', 'w')

def ehrenfest_run(inp_gs):
    traj_file = inp_gs + '.traj'
    Ekin = 80
    timestep = 32.0 #*0.2 #* np.sqrt(10/Ekin)
    ekin_str = '_ek' + str(int(Ekin))
    amu_to_aumass = _amu/_me

    tdcalc = LCAOTDDFT(inp_gs, propagator='edsicn',PLCAO_flag=False,
                    Ehrenfest_flag = True,
                    txt=inp_gs + '_out_td.txt',  S_flag=True, Ehrenfest_force_flag=False)
    tdcalc.tddft_init()
    f = open(inp_gs + '_td.log', 'w')
    proj_idx = 1
    pos=tdcalc.atoms.get_positions()
    #v = np.zeros((proj_idx+1,3))
    v = np.zeros((len(pos[:,0]),3))
    delta_stop = 2.0 / Bohr
    Mproj = tdcalc.atoms.get_masses()[proj_idx]
    #Ekin *= Mproj
    Ekin = Ekin / Hartree

    Mproj *= amu_to_aumass
    for l in np.arange(0,len(v[:,0]))[0::2]:
        print('lll',l)
        v[l,2] = -np.sqrt((2*Ekin)/Mproj) * Bohr / AUT

    tdcalc.atoms.set_velocities(v)
    evv = EhrenfestVelocityVerlet(tdcalc)
    traj = Trajectory(traj_file, 'w', tdcalc.get_atoms())
    ndiv = 1
    # NUMBER OF STEPS
    niters =  15 #*5*3
    #f.write('T (as) Epot (eV) Ekin (eV) Etot (eV) \n ')
    
    for i in range(niters):
        print("STEM MD i",i)
        Ekin_p = 0.5*evv.M[proj_idx]*(evv.velocities[proj_idx,0]**2 \
                                      + evv.velocities[proj_idx,1]**2 \
                                          + evv.velocities[proj_idx,2]**2)
        if i % ndiv ==0:
            nLow, n, nLowm , nm, eigv, nLow_kM  = calculate_norm(tdcalc,tdcalc.wfs,inp_gs)
            Etot = evv.get_energy()
            F_av = evv.Forces * Hartree / Bohr
            epot = tdcalc.get_td_energy() * Hartree
            ekin = tdcalc.atoms.get_kinetic_energy()
            T = i * timestep
            ep = Ekin_p * Hartree
            f.write('%6.2f %18.10f %18.10f %18.10f %18.10f \n'  \
                  % (T, ep, ekin, epot, ekin+epot))
            f.flush()
            epot = tdcalc.get_td_energy() * Hartree
            spa = tdcalc.get_atoms()
            spc = SinglePointCalculator(epot)
            spa.set_calculator(spc)
            traj.write(spa)
        evv.propagate(timestep)
    traj.close()
    return F_av

inp_gs=['1_gs.gpw','2_gs.gpw']

F=[]
for i in inp_gs:
    print('==========', i, '==========')
    print('===========================')
    F.append(ehrenfest_run(i)[0])
    print('===========================')
print('F error',np.abs(F[0]-F[1]).max())

