import pytest
from gpaw.basis_data import Basis
from gpaw.mpi import world
from gpaw.lcao.generate_ngto_augmented import do_nao_ngto_basis

# Generate file with GTO parameters
pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


def test_lcao_generate_ngto(in_tmp_dir):
    with open('gbs.txt', 'w') as f:
        def w(s):
            f.write('%s\n' % s)
        w('****')
        w('H     0')
        w('S   1   1.00')
        w('      0.1000000              1.0000000')
        w('P   1   1.00')
        w('      0.0500000              1.0000000')
        w('D   1   1.00')
        w('      0.5000000              1.0000000')
        w('****')

    # Run the generator script in order to keep the syntax up-to-date
    do_nao_ngto_basis('H', 'LDA', 'sz', 'gbs.txt', 'NAO+NGTO')

    # Check that the generated basis contains the correct number of functions
    basis = Basis('H', 'NAO+NGTO', readxml=False)
    basis.read_xml('H.NAO+NGTO.sz.basis')
    bf_j = basis.bf_j

    assert len(bf_j) == 1 + 3

    with open('gbs.txt', 'w') as f:
        def w(s):
            f.write('%s\n' % s)
        w('****')
        w('C     0')
        w('S   1   1.00')
        w('      1.596000D-01           1.000000D+00')
        w('S   1   1.00')
        w('      0.0469000              1.0000000')
        w('P   4   1.00')
        w('      9.439000D+00           3.810900D-02')
        w('      2.002000D+00           2.094800D-01')
        w('      5.456000D-01           5.085570D-01')
        w('      1.517000D-01           4.688420D-01')
        w('P   1   1.00')
        w('      1.517000D-01           1.000000D+00')
        w('P   1   1.00')
        w('      0.0404100              1.0000000')
        w('D   1   1.00')
        w('      5.500000D-01           1.0000000')
        w('D   1   1.00')
        w('      0.1510000              1.0000000')
        w('****')

    do_nao_ngto_basis('C', 'LDA', 'dzp', 'gbs.txt', 'DZP+NGTO')

    basis = Basis('C', 'DZP+NGTO', readxml=False)
    basis.read_xml('C.DZP+NGTO.dzp.basis')
    bf_j = basis.bf_j

    assert len(bf_j) == 5 + 7
