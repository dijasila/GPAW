from gpaw.analyse.observers import Observer
from gpaw import GPAW
from ase.units import Hartree
from gpaw.mpi import world
import numpy as np

# Time dependent density functional perturbation theory

def transform_local_operator(gpw_file=None, tdop_file=None, fqop_file=None, omega=None, eta=0.1):
    assert world.size == 1
    assert gpw_file is not None
    assert tdop_file is not None
    assert fqop_file is not None
    assert omega is not None
    omega /= Hartree
    eta /= Hartree

    tdf = open(tdop_file+'.sG','r')
    tdaf = open(tdop_file+'.asp','r')
    calc = GPAW(gpw_file)

    Fq_G = calc.hamiltonian.gd.zeros(dtype=complex)
    NG = np.prod(Fq_G.shape)

    # Reference values, taken from t=0 value
    ref_dH_asp = {}
    Fq_dH_asp = {}
    for a, dH_sp in calc.hamiltonian.dH_asp.iteritems():
        ref_dH_asp[a] = np.zeros_like(dH_sp, dtype=complex)
        Fq_dH_asp[a] = np.zeros_like(dH_sp, dtype=complex)

    ref_sG = None

    iteration = 0

    while True:
        # Read pseudo potential and spherical corrections
        Td_G = np.fromfile(tdf, dtype=np.float64, count=NG)
        if np.prod(Td_G.shape) < NG:
            break

        Td_G = Td_G.reshape(Fq_G.shape)
        if ref_sG is None:
            ref_sG = Td_G.copy()

        header = tdaf.readline().split()
        print header
        if len(header) == 0:
            break
        assert len(header) == 2
        time = float(header[0])
        natoms = int(header[1])
        Td_dH_asp = {}
        for n in range(natoms):
            data = tdaf.readline().split()
            a = int(data[0])
            Td_dH_asp[a] = np.zeros_like(calc.hamiltonian.dH_asp[a])
            Td_dH_asp[a][:] = map(float, data[1:])

        if ref_dH_asp is None:
            ref_dH_asp = Td_dH_asp[a].deepcopy()

        if iteration == 0:
            print time
            assert time == 0.0
        c = np.exp(1j * time * omega - eta * time)
        print "Iteration", iteration, c
        # Add to fourier transform
        Fq_G += c * (Td_G-ref_sG)
        for a, dH_sp in ref_dH_asp.iteritems():
            Fq_dH_asp[a] += c * (Td_dH_asp[a] - ref_dH_asp[a])

        iteration += 1

    tdf.close() 
    tdaf.close()
    
    fqf = open(fqop_file+'.sG','w')
    Fq_G.tofile(fqf)
    fqf.close()

    # Make sure we know how to read it
    fqf = open(fqop_file+'.sG','r')
    Fq2_G = np.fromfile(fqf, dtype=np.complex128, count=NG).reshape(Fq_G.shape)
    fqf.close()
    assert np.abs(Fq2_G-Fq_G).sum() <1e-10# XXX

    # Output the Fourier transformed Hamiltonian
    fqaf = open(fqop_file+'.asp','w')
    print >>fqaf, "%.10f %.10f %d" % (omega, eta, len(Fq_dH_asp))
    for a, dH_sp in Fq_dH_asp.iteritems():
        print >>fqaf, a,
        for H in dH_sp.ravel():
            print >>fqaf, H.real, H.imag,
        print >>fqaf


class TDDFPT(GPAW):
    def __init__(self, gpw_filename, Vop_filename, **kwargs):
        GPAW.__init__(self, gpw_filename, **kwargs)

        # Read the complex smooth potential
        tdf = open(Vop_filename+'.sG','r')
        NG = np.prod(self.hamiltonian.vt_sG.shape)
        self.cvt_sG = np.fromfile(tdf, dtype=np.complex128, count=NG).reshape(self.hamiltonian.vt_sG.shape)

        # Read the complex PAW corrections
        tdaf = open(Vop_filename+'.asp','r')
        header = tdaf.readline().split()
        assert len(header) == 3
        self.omega = float(header[0])
        self.eta = float(header[1])
        NA = int(header[2])

        print "TDDFPT"
        print "Omega: %.4f eV" % (self.omega * Hartree)
        print "Eta: %.4f eV" % (self.omega * Hartree)

        self.cH_asp = {}
        for a, dH_sp in self.hamiltonian.dH_asp.iteritems():
            data = tdaf.readline().split()
            a = int(data[0])
            self.cH_asp[a] = np.zeros_like(dH_sp, dtype=complex)
            self.cH_asp[a][:] = np.array(map(float, data[1::2]))
            self.cH_asp[a][:] += 1j*np.array(map(float, data[2::2]))

    def calculate(self):
        self.initialize(self.atoms)
        self.set_positions(self.atoms)

        # Real part 
        self.hamiltonian.vt_sG[:] = self.cvt_sG.real
        for a, dH_sp in self.hamiltonian.dH_asp.iteritems():
            self.hamiltonian.dH_asp[a][:] = self.cH_asp[a].real
        
        print self.hamiltonian.vt_sG, np.sum(self.hamiltonian.vt_sG)
        print self.hamiltonian.dH_asp[0]
        print self.hamiltonian.dH_asp[1]
        Hp_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(\
                     self.hamiltonian, self.wfs, self.wfs.kpt_u[0], root=-1)
        print "Hp_MM", Hp_MM

        # Imag part 
        self.hamiltonian.vt_sG[:] = self.cvt_sG.imag
        for a, dH_sp in self.hamiltonian.dH_asp.iteritems():
            self.hamiltonian.dH_asp[a][:] = self.cH_asp[a].imag
        Hm_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(\
                     self.hamiltonian, self.wfs, self.wfs.kpt_u[0], root=-1)
        print "Hm_MM", Hm_MM

        S_MM = self.wfs.S_qMM[0]
        C0_nM = self.wfs.kpt_u[0].C_nM.copy()
        
        d_m = np.dot(np.dot(S_MM, Hm_MM), S_MM.T)
        Hm_nn = np.dot(np.dot(C0_nM, d_m), C0_nM.T.conjugate())
        print "Hm_nn", Hm_nn

        d_p = np.dot(np.dot(S_MM, Hp_MM), S_MM.T)
        Hp_nn = np.dot(np.dot(C0_nM, d_p), C0_nM.T.conjugate())
        print "Hp_nn", Hp_nn

class HamiltonianCollector(Observer):

    def __init__(self, filename, lcao):
        Observer.__init__(self)        
        self.lcao = lcao               
        self.filename = filename+'.sG'
        self.H_asp_filename = filename+'.asp'
        self.first_iteration = True

    def update(self):
        hamiltonian = self.lcao.hamiltonian

        if self.first_iteration:
            self.first_iteration = False
            if hamiltonian.world.rank == 0:
                # Create an empty file
                f = open(self.filename, 'w')
                f.close()
                f = open(self.H_asp_filename, 'w')
                f.close()

        vt_sG = hamiltonian.gd.collect(hamiltonian.vt_sG, broadcast=False)

        if hamiltonian.world.rank == 0:
            f = open(self.filename,'a+')
            vt_sG.tofile(f)
            f.close()

        dH_asp = hamiltonian.dH_asp.deepcopy()
        serial_partition = dH_asp.partition.as_serial()
        dH_asp.redistribute(serial_partition)

        if serial_partition.comm.rank == 0 and self.lcao.wfs.bd.comm.rank == 0:
            f = open(self.H_asp_filename,'a+')
            print >>f, self.lcao.time, len(dH_asp)
            for a, dH_sp in dH_asp.iteritems():
                print >>f, a,
                for dH in dH_sp.ravel():
                    print >>f, dH,
                print >>f
            f.close()



class DensityCollector(Observer):

    def __init__(self, filename, lcao):
        Observer.__init__(self)        
        self.lcao = lcao               
        self.filename = filename+'.sG'
        self.D_asp_filename = filename+'.asp'
        self.first_iteration = True

    def update(self):
        hamiltonian = self.lcao.hamiltonian
        density = self.lcao.density
        iter = self.niter

        if self.first_iteration:
            self.first_iteration = False
            if hamiltonian.world.rank == 0:
                # Create an empty file
                f = open(self.filename, 'w')
                f.close()
                f = open(self.D_asp_filename, 'w')
                f.close()

        nt_sG = density.gd.collect(density.nt_sG, broadcast=False)

        if hamiltonian.world.rank == 0:
            f = open(self.filename,'a+')
            nt_sG.tofile(f)
            print(nt_sG.sum())
            f.close()

        D_asp = density.D_asp.deepcopy()
        serial_partition = D_asp.partition.as_serial()
        D_asp.redistribute(serial_partition)

        if serial_partition.comm.rank == 0 and self.lcao.wfs.bd.comm.rank == 0:
            f = open(self.D_asp_filename,'a+')
            print >>f, self.lcao.time, len(D_asp)
            for a, D_sp in D_asp.iteritems():
                print >>f, a,
                for D in D_sp.ravel():
                    print >>f, D,
                print >>f
            f.close()


"""

# Simplest example of use of LCAO-TDDFPT code

from ase import Atoms
from gpaw import GPAW
from ase.optimize import BFGS
from gpaw.tddft import photoabsorption_spectrum
from gpaw import PoissonSolver
from gpaw.lcaotddft.tddfpt import HamiltonianCollector, transform_hamiltonian

# Sodium dimer
atoms = Atoms('Na2', positions=[[0.0,0.0,0.0],[3.0,0.0,0.0]])
atoms.center(vacuum=5.0)

from gpaw.lcaotddft import LCAOTDDFT

# Increase accuragy of density for ground state
convergence = {'density':1e-7}

# Increase accuracy of Poisson Solver and apply multipole corrections up to l=2
poissonsolver = PoissonSolver(eps=1e-20, remove_moment=1+3+5)

if 0:
    # Calculate all bands
    td_calc = LCAOTDDFT(setups={'Na':'1'}, basis='1.dzp', xc='oldLDA', h=0.3,
                        convergence=convergence, poissonsolver=poissonsolver)

    atoms.set_calculator(td_calc)
    atoms.get_potential_energy()
    td_calc.write('Na_gs.gpw',mode='all')

if 0:
    td_calc = LCAOTDDFT('Na_gs.gpw', poissonsolver=poissonsolver)
    td_calc.attach(HamiltonianCollector('Na.TdHam', td_calc))
    td_calc.kick([1e-5, 0.0, 0.0])
    td_calc.propagate(10, 500, 'Na2.dm')



[juhanim@glenn lcaotddft]$ cat test.
cat: test.: No such file or directory
[juhanim@glenn lcaotddft]$ cat test.py 
# Simplest example of use of LCAO-TDDFPT code

from ase import Atoms
from gpaw import GPAW
from ase.optimize import BFGS
from gpaw.tddft import photoabsorption_spectrum
from gpaw import PoissonSolver
from gpaw.lcaotddft.tddfpt import HamiltonianCollector, transform_hamiltonian

# Sodium dimer
atoms = Atoms('Na2', positions=[[0.0,0.0,0.0],[3.0,0.0,0.0]])
atoms.center(vacuum=5.0)

from gpaw.lcaotddft import LCAOTDDFT

# Increase accuragy of density for ground state
convergence = {'density':1e-7}

# Increase accuracy of Poisson Solver and apply multipole corrections up to l=2
poissonsolver = PoissonSolver(eps=1e-20, remove_moment=1+3+5)

if 0:
    # Calculate all bands
    td_calc = LCAOTDDFT(setups={'Na':'1'}, basis='1.dzp', xc='oldLDA', h=0.3, 
                        convergence=convergence, poissonsolver=poissonsolver)

    atoms.set_calculator(td_calc)
    atoms.get_potential_energy()
    td_calc.write('Na_gs.gpw',mode='all')

if 0:
    td_calc = LCAOTDDFT('Na_gs.gpw', poissonsolver=poissonsolver)
    td_calc.attach(HamiltonianCollector('Na.TdHam', td_calc))
    td_calc.kick([1e-5, 0.0, 0.0])
    td_calc.propagate(10, 500, 'Na2.dm')

transform_local_operator(gpw_file='Na_gs.gpw', TdHam_file='Na.TdHam', FqHam_file='Na.FqHam', omega=2.5, eta=0.4)


"""
