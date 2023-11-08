import pytest
import numpy as np
from ase.build import molecule
from ase.parallel import paropen
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.qed import RRemission
from gpaw.tddft.spectrum import read_td_file_data
from . import check_txt_data


@pytest.mark.rttddft
def test_rremission(in_tmp_dir):
    atoms = molecule('Na2')
    atoms.center(vacuum=4.0)
    calc = GPAW(mode='lcao', h=0.4, basis='dzp',
                setups={'Na': '1'},
                symmetry={'point_group': False},
                convergence={'density': 1e-12})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')
    td_calc = LCAOTDDFT('gs.gpw', rremission=RRemission(0.5, [0, 0, 1]),
                        propagator={'name': 'scpc', 'tolerance': 1e-0})
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5])
    td_calc.propagate(40, 20)

    with paropen('dm_ref.dat', 'w') as fd:
        fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
# Start; Time = 0.00000000
          0.00000000      -3.48400569e-16     1.460564021477e-15     3.225647182566e-15    -3.982847629258e-14
# Kick = [    0.000000000000e+00,     0.000000000000e+00,     1.000000000000e-05]; Time = 0.00000000
          0.00000000      -6.08581937e-16     5.454276145067e-15     4.927570818403e-15    -3.914902027642e-14
          1.65365493      -6.85797052e-16     2.783530505795e-15     2.217350475848e-15     3.419488131280e-05
          3.30730987       3.56327242e-16    -6.259560092043e-16    -1.779745193738e-15     6.328738771521e-05
          4.96096480      -6.85797052e-16     9.405130019378e-15     8.259235775502e-15     8.608533242884e-05
          6.61461974      -7.33357087e-16     5.064040506896e-15     2.186898561887e-15     1.032870345833e-04
          8.26827467       1.35872492e-16     1.944411098862e-15     2.763229229821e-16     1.156375187581e-04
          9.92192960      -1.98446577e-16     9.981460687312e-16     9.778447927570e-16     1.238315688509e-04
         11.57558454       2.77899812e-17     1.239505683091e-15     2.971881232889e-15     1.285064383632e-04
         13.22923947      -1.22798145e-15     1.429999322649e-14     1.309319515469e-14     1.302401811969e-04
         14.88289440       2.06093485e-17     2.206071989196e-15    -1.160556276525e-15     1.295514557099e-04
         16.53654934      -4.47903623e-16     2.035766840746e-15     2.089903576677e-15     1.269006285844e-04
         18.19020427       9.21359108e-17    -3.221135787906e-15    -2.068474452037e-15     1.226921048004e-04
         19.84385921      -8.09546398e-16     3.814384185818e-15     5.569316708921e-15     1.172777277082e-04
         21.49751414       1.86323431e-16    -3.160231959983e-15    -5.973086531074e-15     1.109610266703e-04
         23.15116907      -9.47750264e-16     5.668567391461e-15     7.347934053993e-15     1.040020450314e-04
         24.80482401      -6.65840489e-16     6.207679053443e-15     9.655512423060e-15     9.662244521506e-05
         26.45847894      -5.09918178e-16     3.870776619080e-15     3.298957345807e-15     8.901056071782e-05
         28.11213387      -1.07448377e-15     8.076524291734e-15     5.227578563355e-15     8.132605402745e-05
         29.76578881      -2.28194913e-16     6.296779097996e-15     4.008374156238e-15     7.370387298427e-05
         31.41944374      -1.13976225e-15     7.537412629752e-15     3.489563770231e-15     6.625729128738e-05
         33.07309868       6.30403600e-16    -5.636987628835e-15    -2.798192538443e-15     5.907997327753e-05
'''.strip())  # noqa: E501

    check_txt_data('dm.dat', 'dm_ref.dat', atol=1e-8)

    """
    Restart check for rremission. Does restarting change the outcome?
    """

    td_calc = LCAOTDDFT('gs.gpw', rremission=RRemission(0.5, [0, 0, 1]),
                        propagator={'name': 'scpc', 'tolerance': 1e-0})
    DipoleMomentWriter(td_calc, 'dm_rrsplit.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5])
    td_calc.propagate(40, 10)
    td_calc.write('td_rrsplit0.gpw', mode='all')

    td_calc_restart = LCAOTDDFT('td_rrsplit0.gpw')
    DipoleMomentWriter(td_calc_restart, 'dm_rrsplit.dat')
    td_calc_restart.propagate(40, 10)

    dipole_full = read_td_file_data('dm.dat')[1][-10:]
    dipole_restart = read_td_file_data('dm_rrsplit.dat',)[1][-10:]
    assert np.allclose(dipole_full, dipole_restart)
