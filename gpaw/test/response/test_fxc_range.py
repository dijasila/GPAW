import pytest
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw import PW
from gpaw.xc.fxc import FXCCorrelation
from ase.units import Hartree

@pytest.mark.response
def test_xc_short_range(in_tmp_dir):
        bulk_si = bulk('Si', a=5.42935602)
        calc = GPAW(mode=PW(400.0),
                    xc='LDA',
                    occupations=FermiDirac(width=0.01),
                    kpts={'size': (4, 4, 4), 'gamma': True},
                    parallel={'domain': 1},
                    txt='si.gs.txt')
        bulk_si.calc = calc

        E_lda = bulk_si.get_potential_energy()
        calc.diagonalize_full_hamiltonian()
        calc.write('si.lda_wfcs.gpw', mode='all')

        # test high r_c and low r_c to ensure results in the 2 limits for the aprx are equal
        rc, ec, nbnd =  2.0, 2.25, 100

        fxc = FXCCorrelation('si.lda_wfcs.gpw',
                                xc='range_RPA',
                                txt='si_range.' + str(rc) + '.txt',
                                range_rc=rc)
        E_i = fxc.calculate(ecut=[ec * Hartree], nbands=nbnd)

        assert E_i[0] == pytest.approx(-12.250, 0.05)
