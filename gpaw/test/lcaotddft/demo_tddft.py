from gpaw import LCAO
from gpaw.calculator import GPAW as old_GPAW
from gpaw.new.ase_interface import GPAW as new_GPAW
from gpaw.new.rttddft import RTTDDFT
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

from ase.build import molecule

atoms = molecule('H2')
atoms.center(vacuum=5)
atoms.pbc = False

run_old_gs = False
run_new_gs = True
run_old_td = False
run_new_td = True

if run_old_gs:
    calc = old_GPAW(mode=LCAO(), basis='sz(dzp)', xc='LDA', txt='old.out')
    atoms.calc = calc
    E = atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')
    old_C_nM = calc.wfs.kpt_u[0].C_nM
    old_calc = calc


if run_new_gs:
    calc = new_GPAW(mode='lcao', basis='sz(dzp)', xc='LDA', txt='new.out',
                    force_complex_dtype=True)
    atoms.calc = calc
    E = atoms.get_potential_energy()
    new_C_nM = calc.calculation.state.ibzwfs.wfs_qs[0][0].C_nM
    new_calc = calc

if run_old_td:
    old_tddft = LCAOTDDFT('gs.gpw')
    dm = DipoleMomentWriter(old_tddft, 'dm.out')
    old_tddft.absorption_kick([0, 0, 1e-3])
    old_tddft.propagate(10, 20)

if run_new_td:
    new_tddft = RTTDDFT.from_dft_calculation(new_calc)

    dt = 10
    for result in new_tddft.ipropagate(dt, 10):
        print(result)
