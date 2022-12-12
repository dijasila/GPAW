import pytest
from ase import Atoms

from gpaw import GPAW, FermiDirac
from gpaw.mpi import serial_comm
from gpaw.xc.fxc import FXCCorrelation
from gpaw.xc.rpa import RPACorrelation


@pytest.mark.rpa
@pytest.mark.response
def test_ralda_ralda_energy_He(in_tmp_dir, scalapack):
    a = 3.0
    atoms = Atoms('He', cell=[a, a, a], pbc=True)
    calc = GPAW(mode=dict(name='pw', ecut=200),
                kpts=dict(size=(2, 2, 2), gamma=True),
                nbands=2,
                occupations=FermiDirac(0.001),
                # FXCCorrelation needs a serial-comm GPAW object:
                communicator=serial_comm)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.diagonalize_full_hamiltonian(nbands=20)

    ecuts = [20, 30]
    rpa = RPACorrelation(calc, nfrequencies=8)
    E_rpa1 = rpa.calculate(ecut=ecuts)[-1]

    def fxc(xc, nfrequencies=8, **kwargs):
        return FXCCorrelation(calc, xc=xc, **kwargs).calculate(ecut=ecuts)[-1]

    energies = [
        fxc('RPA', nlambda=16),
        fxc('rALDA', unit_cells=[1, 1, 2]),
        fxc('rAPBE', unit_cells=[1, 1, 2]),
        fxc('rALDA', av_scheme='wavevector'),
        fxc('rAPBE', av_scheme='wavevector'),
        fxc('JGMs', av_scheme='wavevector', Eg=3.1, nlambda=2),
        fxc('CP_dyn', av_scheme='wavevector', nfrequencies=2, nlambda=2)]

    assert E_rpa1 == pytest.approx(energies[0], abs=0.01)

    refs = [-0.1054,
            -0.0560,
            -0.0523,
            -0.0241,
            -0.0288,
            -0.0263,
            -0.0275]

    for val, ref in zip(energies, refs):
        assert val == pytest.approx(ref, abs=0.001)
