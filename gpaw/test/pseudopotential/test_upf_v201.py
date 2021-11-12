from ase.build import molecule
from gpaw.upf import UPFSetupData
from gpaw import GPAW, PW

# test the performance of a UPF file, v2.0.1 (produces by Quantum Espresso 6.8)
# where the radial grid is non-linear and starts from non-zero.


def test_upf_v201(in_tmp_dir):
    system = molecule('H2', cell=(5, 5, 5))
    system.center()
    system.calc = GPAW(txt='-',
                       nbands=4,
                       setups={'H': UPFSetupData('H.ccecp.UPF')},
                       mode=PW(3000),  # large cutoff needed for ccecp
                       xc='LDA',
                       )
    E = system.get_potential_energy()
    # compare to reference energy from Quantum Espresso
    #   if not, comp_charge or something else probably broken
    E_qe = -31.031016
    assert abs(E - E_qe) < 0.05, 'Energy %f differs from ref: %f' % (E, E_qe)
