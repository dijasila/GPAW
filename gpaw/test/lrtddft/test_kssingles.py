from ase.build import molecule

from gpaw import GPAW
from gpaw.lrtddft.kssingle import KSSingles


def test_old_io():
    """Test reading of old style output files"""
    with open('veryold.dat', 'w') as f:
        f.write("""# KSSingles
2
0 1   0 0   0.129018 1   -4.7624e-02 -3.2340e-01 -4.6638e-01
0 1   1 0   0.129018 1   -4.7624e-02 -3.2340e-01 -4.6638e-01""")

    kss = KSSingles.read('veryold.dat')
    assert len(kss) == 2

    with open('old.dat', 'w') as f:
        f.write("""# KSSingles
2 float64
0.001
0 1  0 0  0.392407 2     8.82768e-14 2.91720e-14 7.98158e-01   1.00594e-13 2.77592e-14 7.85104e-01   1
.45841e-17 1.05676e-17 1.08535e-19
0 2  0 0  0.402312 2     2.24404e-14 8.49182e-14 -5.24372e-14   2.30876e-14 8.26330e-14 -4.99038e-14""")

    kss = KSSingles.read('veryold.dat')
    assert len(kss) == 2
    assert kss.restrict['eps'] == 0.001


def test_io():
    ch4 = molecule('CH4')
    ch4.center(vacuum=2)
    calc = GPAW(h=0.25, nbands=8, txt=None)
    calc.calculate(ch4)

    # full KSSingles
    kssfull = KSSingles(calc, restrict={'eps': 0.9})
    kssfull.write('kssfull.dat')

    # read full
    kss1 = KSSingles.read('kssfull.dat')
    assert len(kss1) == 16
    
    # restricted KSSingles
    istart, jend = 1, 4
    kss = KSSingles(calc,
                    restrict={'eps': 0.9, 'istart': istart, 'jend': jend})
    kss.write('kss_1_4.dat')

    kss2 = KSSingles.read('kss_1_4.dat')
    assert len(kss2) == len(kss)
    assert kss2.restrict['istart'] == istart
    assert kss2.restrict['jend'] == jend

    # restrict when reading
    kss3 = KSSingles.read('kssfull.dat',
                          restrict={'istart': istart, 'jend': jend})
    assert len(kss3) == len(kss)
    assert kss3.restrict['istart'] == istart
    assert kss3.restrict['jend'] == jend


if __name__ == '__main__':
    # test_old_io()
    test_io()
