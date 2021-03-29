from ase import Atoms

from gpaw import GPAW, FermiDirac
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft2 import LrTDDFT2


def test_lrtddft2():
    """Test equivalence"""
    atoms = Atoms('He')
    atoms.cell = [4, 4, 5]
    atoms.center()

    atoms.calc = GPAW(occupations=FermiDirac(width=0.1),
                      nbands=2)
    atoms.get_potential_energy()

    lr = LrTDDFT(atoms.calc)
    print(lr.kss[0])
    lr2 = LrTDDFT2('O_lr', atoms.calc, fxc='LDA')
    #print(dir(lr2.ks_singles.kss_list))
    print(lr2.ks_singles.kss_list)

    assert 0
