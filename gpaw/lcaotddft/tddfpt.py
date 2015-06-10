from gpaw.analyse.observers import Observer
from gpaw import GPAW
from ase.units import Hartree
from gpaw.mpi import world
import numpy as np

# Time dependent density functional perturbation theory

def transform_hamiltonian(gpw_file=None, TdHam_file=None, FqHam_file=None, omega=None, eta=0.1):
    assert world.size == 1
    assert gpw_file is not None
    assert TdHam_file is not None
    assert FqHam_file is not None
    assert omega is not None
    omega /= Hartree
    eta /= Hartree

    tdf = open(TdHam_file,'r')
    tdaf = open(TdHam_file+'.H_asp','r')
    calc = GPAW(gpw_file)
    print "Hasp", calc.hamiltonian.dH_asp

    Fq_G = calc.hamiltonian.gd.zeros(dtype=complex)
    NG = np.prod(Fq_G.shape)

    Fq_dH_asp = {}
    for a, dH_sp in calc.hamiltonian.dH_asp.iteritems():
        Fq_dH_asp[a] = np.zeros_like(dH_sp, dtype=complex)
    
    iteration = 0
    while True:
        # Read pseudo potential and spherical corrections
        Td_G = np.fromfile(tdf, dtype=np.float64, count=NG)

        header = tdaf.readline().split()
        if len(header) == 0:
            break
        assert len(header) == 2
        time = float(header[0])
        natoms = int(header[1])
        Td_dH_asp = {}
        for n in range(natoms):
            data = tdaf.readline().split()
            a = int(data[0])
            Td_dH_asp[a] = np.zeros_like(Fq_dH_asp[a])
            Td_dH_asp[a][:] = map(float, data[1:])
        if iteration == 0:
            print time
            assert time == 0.0
        c = np.exp(1j * time * omega - eta * time)
        print "Iteration", iteration, c
        # Add to fourier transform
        Fq_G += c * Td_G
        for a, dH_sp in Fq_dH_asp:
            Fq_dH_asp[a] += c * (Td_dH_asp[a] - calc.hamiltonian.dH_asp[a])

        iteration += 1

    tdf.close() 
    tdaf.close()
    
    fqf = open(FqHam_file,'w')
    Fq_G.tofile(fqf)
    fqf.close()

    # Output the Fourier transformed Hamiltonian
    fqaf = open(FqHam_file+'.H_asp','w')
    print >>fqaf, "%.10f %.10f %d" % (omega, eta, len(Fq_dH_asp))
    for a, dH_sp in Fq_dH_asp:
        print >>fqaf, a,
        for H in dH_sp.ravel():
            print >>fqaf, H,
        print >>faqf

class TDDFPT(GPAW):
    def __init__(self, gpw_filename, FqHam_filename, **kwargs):
        GPAW.__init__(self, gpw_filename, **kwargs)
    
    def calculate(self):
        pass


class HamiltonianCollector(Observer):

    def __init__(self, filename, lcao):
        Observer.__init__(self)        
        self.lcao = lcao               
        self.filename = filename
        self.H_asp_filename = filename+'.H_asp'
        self.first_iteration = True

    def update(self):
        hamiltonian = self.lcao.hamiltonian
        iter = self.niter

        if self.first_iteration:
            self.first_iteration = False
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


