import pytest
from ase import Atom, Atoms

from gpaw import GPAW, restart, Davidson, Mixer
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from gpaw import setup_paths
from gpaw.mpi import world
from gpaw.test import equal

# This test calculates the derivative discontinuity of Ne-atom
# first on 3D without restart. Then does restart and recalculates.


@pytest.mark.gllb
@pytest.mark.libxc
def test_gllb_ne_disc(in_tmp_dir):
    atom = 'Ne'
    setup_paths.insert(0, '.')

    for xcname in ['GLLB', 'GLLBSC']:
        if world.rank == 0:
            g = Generator(atom, xcname=xcname, scalarrel=False, nofiles=True)
            g.run(**parameters[atom])
            eps = g.e_j[-1]
        world.barrier()

        a = 10
        Ne = Atoms([Atom(atom, (0, 0, 0))],
                   cell=(a, a, a), pbc=False)
        Ne.center()
        calc = GPAW(eigensolver=Davidson(4), nbands=10, h=0.18, xc=xcname,
                    basis='dzp',
                    mixer=Mixer(0.6))
        Ne.calc = calc
        Ne.get_potential_energy()
        homo, lumo = calc.get_homo_lumo()
        response = calc.hamiltonian.xc.response
        dxc_pot = response.calculate_discontinuity_potential(homo, lumo)
        KS, dxc = response.calculate_discontinuity(dxc_pot)
        if xcname == 'GLLB':
            equal(KS + dxc, 24.89, 1.5e-1)
        else:
            equal(KS + dxc, 27.70, 6.0e-2)
        eps3d = calc.wfs.kpt_u[0].eps_n[3]
        calc.write('Ne_temp.gpw', mode='all')

        atoms, calc = restart('Ne_temp.gpw')
        homo, lumo = calc.get_homo_lumo()
        response = calc.hamiltonian.xc.response
        dxc_pot = response.calculate_discontinuity_potential(homo, lumo)
        KS2, dxc2 = response.calculate_discontinuity(dxc_pot)
        equal(KS, KS2, 1e-5)
        equal(dxc2, dxc, 1e-5)

        # Hardness of Ne 24.71eV by GLLB+Dxc, experimental I-A = I = 21.56eV
        #
        # Not sure where 24.71 comes from, but with better grid and better
        # stencil, result becomes 24.89.  --askhl

        if world.rank == 0:
            equal(eps, eps3d, 1e-3)
        if xcname == 'GLLB':
            equal(24.89, KS2 + dxc2, 1.2e-1)
