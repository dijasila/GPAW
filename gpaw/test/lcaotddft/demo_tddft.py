import numpy as np

from ase.build import molecule

from gpaw import LCAO
from gpaw.calculator import GPAW as old_GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.new.ase_interface import GPAW as new_GPAW
from gpaw.new.rttddft import RTTDDFT
from gpaw.tddft.units import as_to_au, autime_to_asetime


def main():
    atoms = molecule('H2')
    atoms.center(vacuum=5)
    atoms.pbc = False

    run_old_gs = False
    run_new_gs = False
    run_old_td = False
    run_new_td = True

    def assert_equal(a, b):
        from gpaw.core.matrix import Matrix
        from gpaw.core.atom_arrays import AtomArrays

        def extract(o):
            if isinstance(o, Matrix):
                return o.data
            elif isinstance(o, AtomArrays):
                return o.data
            else:
                return o

        a = extract(a)
        b = extract(b)

        assert np.allclose(a, b), f'{str(a)} != {str(b)}'

    if run_old_gs:
        old_calc = old_GPAW(mode=LCAO(), basis='sz(dzp)', xc='LDA',
                            txt='old.out', convergence={'density': 1e-12})
        atoms.calc = old_calc
        atoms.get_potential_energy()
        old_calc.write('old_gs.gpw', mode='all')

    if run_new_gs:
        new_calc = new_GPAW(mode='lcao', basis='sz(dzp)', xc='LDA',
                            txt='new.out', force_complex_dtype=True,
                            convergence={'density': 1e-12})
        atoms.calc = new_calc
        atoms.get_potential_energy()
        new_calc.write('new_gs.gpw', mode='all')

        new_restart_calc = new_GPAW('new_gs.gpw')

        assert_equal(
            new_calc.calculation.state.ibzwfs.wfs_qs[0][0].P_ain,
            new_restart_calc.calculation.state.ibzwfs.wfs_qs[0][0].P_ain)

        assert_equal(
            new_calc.calculation.state.ibzwfs.wfs_qs[0][0].C_nM,
            new_restart_calc.calculation.state.ibzwfs.wfs_qs[0][0].C_nM)

    if run_old_td:
        old_tddft = LCAOTDDFT('gs.gpw', propagator='ecn')
        DipoleMomentWriter(old_tddft, 'dm.out')
        old_tddft.propagate(10, 10)
        old_C_nM = old_tddft.wfs.kpt_u[0].C_nM
        old_f_n = old_calc.get_occupation_numbers()
        old_rho_MM = old_C_nM.T.conj() @ (old_f_n[:, None] * old_C_nM)
        print(old_rho_MM)

    if run_new_td:
        #  new_tddft = RTTDDFT.from_dft_calculation(new_calc)
        new_tddft = RTTDDFT.from_dft_file('new_gs.gpw')

        dt = 10 * as_to_au * autime_to_asetime
        for result in new_tddft.ipropagate(dt, 10):
            print(result)
        wfs = new_tddft.state.ibzwfs.wfs_qs[0][0]
        new_f_n = wfs.weight * wfs.spin_degeneracy * wfs.myocc_n
        new_C_nM = wfs.C_nM.data
        new_rho_MM = (new_C_nM.T.conj() * new_f_n) @ new_C_nM
        print(new_rho_MM)


if __name__ == '__main__':
    main()
