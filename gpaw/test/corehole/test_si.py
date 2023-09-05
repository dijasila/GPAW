import pytest

from gpaw import GPAW
from gpaw.test import gen
from gpaw.xas import XAS, RecursionMethod


@pytest.mark.later
def test_corehole_si(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    import numpy as np
    # Generate setup for oxygen with half a core-hole:
    gen('Si', name='hch1s', corehole=(1, 0, 0.5), gpernode=30, write_xml=True)

    # restart from file
    calc = GPAW(gpw_files['si_corehole_pw'])
    si = calc.atoms

    import gpaw.mpi as mpi
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
    calc = GPAW(gpw_files['si_corehole_pw'],
                setups={0: s})
    si.calc = calc

    def stopcalc():
        calc.scf.converged = True

    calc.attach(stopcalc, 1)
    _ = si.get_potential_energy()
