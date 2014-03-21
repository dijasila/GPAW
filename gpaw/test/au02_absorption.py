import numpy as np
from os import system
from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.mpi import rank
from gpaw.response.df2 import DielectricFunction
from gpaw.test import equal, findpeak

GS = 1
ABS = 1
if GS:
    cluster = Atoms([Atom('Au', (0, 0, 0)),
                     Atom('Au', (0, 0, 2.564))])
    cluster.set_cell((6, 6, 6), scale_atoms=False)
    cluster.center()
    calc=GPAW(mode='pw',
              dtype=complex,
              xc='RPBE',
              basis='dzp')
    
    cluster.set_calculator(calc)
    cluster.get_potential_energy()
    calc.write('Au2.gpw','all')

if ABS:
    df = DielectricFunction('Au2.gpw',
                            frequencies=np.linspace(0,14,141),
                            eta=0.1,
                            nbands=18,
                            ecut=10)

    b0, b = df.get_dielectric_function(filename=None,#'au2_df.csv',
                                       direction='z')             
    a0, a = df.get_polarizability(filename=None,#'au2_pol.csv',
                                 direction='z')             
    a0_ws, a_ws = df.get_polarizability(filename=None,#'au2_pol_ws.csv',
                                        wigner_seitz_truncation=True,
                                        direction='z')

    w0_ = 5.60491055
    I0_ = 244.693028
    w_ = 5.696528390
    I_ = 208.4
    
    w, I = findpeak(np.linspace(0, 14., 141), b0.imag)
    equal(w, w0_, 0.05)
    equal(6**3 * I / (4 * np.pi), I0_, 0.5)
    w, I = findpeak(np.linspace(0, 14., 141), a0.imag)
    equal(w, w0_, 0.05)
    equal(I, I0_, 0.5)
    w, I = findpeak(np.linspace(0, 14., 141), a0_ws.imag)
    equal(w, w0_, 0.05)
    equal(I, I0_, 0.5)
    w, I = findpeak(np.linspace(0, 14., 141), b.imag)
    equal(w, w_, 0.05)
    equal(6**3 * I / (4 * np.pi), I_, 0.5)
    w, I = findpeak(np.linspace(0, 14., 141), a.imag)
    equal(w, w_, 0.05)
    equal(I, I_, 0.5)
    # The Wigner-Seitz truncation does not give exactly the same for small cell
    w, I = findpeak(np.linspace(0, 14., 141), a_ws.imag)
    equal(w, w_, 0.2)
    equal(I, I_, 8.0)

if GS:
    if rank == 0:
        system('rm Au2.gpw')
