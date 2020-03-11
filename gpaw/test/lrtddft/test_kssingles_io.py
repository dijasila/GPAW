from ase.build import molecule

from gpaw import GPAW
from gpaw.lrtddft.kssingle import KSSingles


def test_old_io():
    """Test reading of old style output files"""
    with open('veryold.dat', 'w') as f:
        f.write("""# KSSingles
2
0 1   0 0   0.129018 1   -4.7624e-02 -3.2340e-01 -4.6638e-01
0 1   1 0   0.129018 1   -4.7624e-02 -3.2340e-01 -4.6638e-01
""")

    kss = KSSingles.read('veryold.dat')
    assert len(kss) == 2

    with open('old.dat', 'w') as f:
        f.write("""# KSSingles
2 float64
0.024
0 1  0 0  0.392407 2  8.82 2.91 7.98  1.0 2.7 7.8   1.4 1.2 1.08
0 2  0 0  0.402312 2  2.24 8.49 -5.24   2.3 8.26 -4.99038e-14
""")

    kss = KSSingles.read('old.dat')
    assert len(kss) == 2
    assert kss.restrict['eps'] == 0.024


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
    test_old_io()
    # test_io()
