import pytest
from ase import Atom, Atoms
import numpy as np

from gpaw import GPAW, FermiDirac
from gpaw.test import gen
from gpaw.xas import XAS, RecursionMethod
import gpaw.mpi as mpi


@pytest.mark.later
def test_corehole_si(in_tmp_dir, add_cwd_to_setup_paths):
    # Generate setup for oxygen with half a core-hole:
    gen('Si', name='hch1s', corehole=(1, 0, 0.5), gpernode=30, write_xml=True)
    a = 2.6
    si = Atoms('Si', cell=(a, a, a), pbc=True)

    import numpy as np
    calc = GPAW(mode='fd',
                nbands=None,
                h=0.25,
                occupations=FermiDirac(width=0.05),
                setups='hch1s',
                convergence={'maximum iterations': 1})
    si.calc = calc
    _ = si.get_potential_energy()
    calc.write('si.gpw')

    # restart from file
    calc = GPAW('si.gpw')

    if mpi.size == 1:
        xas = XAS(calc)
        x, y = xas.get_spectra()
    else:
        x = np.linspace(0, 10, 50)

    k = 2
    calc = calc.new(kpts=(k, k, k))
    calc.initialize(si)
    calc.set_positions(si)
    assert calc.wfs.dtype == complex

    r = RecursionMethod(calc)
    r.run(40)
    if mpi.size == 1:
        z = r.get_spectra(x)

    if 0:
        import pylab as p
        p.plot(x, y[0])
        p.plot(x, sum(y))
        p.plot(x, z[0])
        p.show()

    # 2p corehole
    s = gen('Si', name='hch2p', corehole=(2, 1, 0.5), gpernode=30)
    calc = GPAW(mode='fd',
                nbands=None,
                h=0.25,
                occupations=FermiDirac(width=0.05),
                setups={0: s})
    si.calc = calc

    def stopcalc():
        calc.scf.converged = True

    calc.attach(stopcalc, 1)
    _ = si.get_potential_energy()


@pytest.mark.later
def test_si_nonortho(in_tmp_dir, add_cwd_to_setup_paths):
    # Generate setup for oxygen with half a core-hole:
    #gen('Si', name='hch1s', corehole=(1, 0, 0.5))
    gen('Si', name='hch1s', corehole=(1, 0, 0.5), gpernode=30, write_xml=True)
    
    a = 5.43095

    si_nonortho = Atoms([Atom('Si', (0, 0, 0)),
                         Atom('Si', (a / 4, a / 4, a / 4))],
                        cell=[(a / 2, a / 2, 0),
                              (a / 2, 0, a / 2),
                              (0, a / 2, a / 2)],
                        pbc=True)

    # calculation with full symmetry
    calc = GPAW(mode='fd',
                nbands=-10,
                h=0.25,
                kpts=(2, 2, 2),
                occupations=FermiDirac(width=0.05),
                setups={0: 'hch1s'})

    si_nonortho.calc = calc
    _ = si_nonortho.get_potential_energy()
    calc.write('si_nonortho_xas_sym.gpw')

    # calculation without any symmetry
    calc = GPAW(mode='fd',
                nbands=-10,
                h=0.25,
                kpts=(2, 2, 2),
                occupations=FermiDirac(width=0.05),
                setups={0: 'hch1s'},
                symmetry='off')

    si_nonortho.calc = calc
    _ = si_nonortho.get_potential_energy()
    calc.write('si_nonortho_xas_nosym.gpw')

    # restart from file
    calc1 = GPAW('si_nonortho_xas_sym.gpw')
    calc2 = GPAW('si_nonortho_xas_nosym.gpw')
    if mpi.size == 1:
        xas1 = XAS(calc1)
        x, y1 = xas1.get_spectra()
        xas2 = XAS(calc2)
        x2, y2 = xas2.get_spectra(E_in=x)

        assert (np.sum(abs(y1 - y2)[0, :500]**2) < 5e-9)
        assert (np.sum(abs(y1 - y2)[1, :500]**2) < 5e-9)
        assert (np.sum(abs(y1 - y2)[2, :500]**2) < 5e-9)
