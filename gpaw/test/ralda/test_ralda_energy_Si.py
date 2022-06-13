import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack

from gpaw import GPAW, FermiDirac
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation


@pytest.mark.slow
@pytest.mark.response
def test_ralda_ralda_energy_Si(in_tmp_dir, scalapack):
    a0 = 5.43
    cell = bulk('Si', 'fcc', a=a0).get_cell()
    Si = Atoms('Si2', cell=cell, pbc=True,
               scaled_positions=((0, 0, 0), (0.25, 0.25, 0.25)))

    kpts = monkhorst_pack((2, 2, 2))
    kpts += np.array([1 / 4, 1 / 4, 1 / 4])

    calc = GPAW(mode=dict(name='pw', ecut=250),
                kpts=kpts,
                occupations=FermiDirac(0.001),
                communicator=serial_comm)
    Si.calc = calc
    Si.get_potential_energy()
    calc.diagonalize_full_hamiltonian(nbands=50)

    ecuts = [20, 30]
    rpa = RPACorrelation(calc, nfrequencies=8)
    E_rpa1 = rpa.calculate(ecut=ecuts)

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

    equal(E_rpa1[-1], energies[0], 0.01)

    refs = [
        -9.5303,
        -8.9431,
        -8.8272,
        -8.7941,
        -8.6809,
        -8.8596,
        -4.6787]
    tols = [0.002] * 5 + [0.001] * 2

    for val, ref, tol in zip(enerigies, refs, tols):
        assert val == pytest.approx(ref, abs=tol)


if __name__ == '__main__':
    test_ralda_ralda_energy_Si(1, 2)
