from ase import Atoms
from gpaw import GPAW, PW
from gpaw.response.df import DielectricFunction
from gpaw.test import findpeak, equal

def get_hydrogen_chain_dielectric_function(NH, NK):
    a = Atoms('H', cell=[2, 2, 0.5], pbc=True)
    a = a.repeat((1, 1, NH))
    a.center()
    a.calc = GPAW(mode=PW(200), kpts={'size': (1, 1, NK), 'gamma': True}, 
                  parallel={'band': 1})
    a.get_potential_energy()
    a.calc.diagonalize_full_hamiltonian(nbands=NH)
    a.calc.write('H_chain.gpw', 'all')
    
    DF = DielectricFunction('H_chain.gpw', ecut=1)
    eps_NLF, eps_LF = DF.get_dielectric_function(direction='z')
    omega_w = DF.get_frequencies()
    return omega_w, eps_LF

omega1_w, eps1_w = get_hydrogen_chain_dielectric_function(10, 8)
omega2_w, eps2_w = get_hydrogen_chain_dielectric_function(20, 4)
omega3_w, eps3_w = get_hydrogen_chain_dielectric_function(40, 2)

eels1_w = -(1. / eps1_w).imag
eels2_w = -(1. / eps2_w).imag
eels3_w = -(1. / eps3_w).imag

opeak1, peak1 = findpeak(omega1_w, eels1_w)
opeak2, peak2 = findpeak(omega2_w, eels2_w)
opeak3, peak3 = findpeak(omega3_w, eels3_w)

equal(opeak1, opeak2, tolerance=1e-3)
equal(opeak2, opeak3, tolerance=1e-3)

from gpaw.mpi import world
from matplotlib import pyplot as plt
if not world.rank:
    plt.plot(omega1_w, -(1. / eels1_w).imag, label='1')
    plt.plot(omega2_w, -(1. / eels2_w).imag, label='2')
    plt.plot(omega3_w, -(1. / eels3_w).imag, label='3')
    plt.xlim(xmin=0.0, xmax=10.0)
    plt.legend()
    plt.show()
