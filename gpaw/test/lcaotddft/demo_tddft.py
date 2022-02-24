from gpaw import LCAO
from gpaw.calculator import GPAW as old_GPAW
from gpaw.new.ase_interface import GPAW as new_GPAW
from gpaw.new.rttddft import RTTDDFT
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.tddft.units import as_to_au, autime_to_asetime

from ase.build import molecule

atoms = molecule('H2')
atoms.center(vacuum=5)
atoms.pbc = False

run_old_gs = False
run_new_gs = True
run_old_td = False
run_new_td = True

if run_old_gs:
    calc = old_GPAW(mode=LCAO(), basis='sz(dzp)', xc='LDA', txt='old.out',
                    convergence={'density': 1e-12})
    atoms.calc = calc
    E = atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')
    old_C_nM = calc.wfs.kpt_u[0].C_nM
    old_calc = calc


if run_new_gs:
    calc = new_GPAW(mode='lcao', basis='sz(dzp)', xc='LDA', txt='new.out',
                    force_complex_dtype=True,
                    convergence={'density': 1e-12})
    atoms.calc = calc
    E = atoms.get_potential_energy()
    new_C_nM = calc.calculation.state.ibzwfs.wfs_qs[0][0].C_nM
    new_calc = calc

if run_old_td:
    old_tddft = LCAOTDDFT('gs.gpw', propagator='ecn')
    dm = DipoleMomentWriter(old_tddft, 'dm.out')
    old_tddft.propagate(10, 10)
    old_C_nM = old_tddft.wfs.kpt_u[0].C_nM
    old_f_n = old_calc.get_occupation_numbers()
    old_rho_MM = old_C_nM.T.conj() @ (old_f_n[:, None] * old_C_nM)

if run_new_td:
    new_tddft = RTTDDFT.from_dft_calculation(new_calc)

    dt = 10 * as_to_au * autime_to_asetime
    for result in new_tddft.ipropagate(dt, 10):
        print(result)
    wfs = new_calc.calculation.state.ibzwfs.wfs_qs[0][0]
    new_f_n = wfs.weight * wfs.spin_degeneracy * wfs.myocc_n
    new_C_nM = wfs.C_nM.data
    new_rho_MM = (new_C_nM.T.conj() * new_f_n) @ new_C_nM
