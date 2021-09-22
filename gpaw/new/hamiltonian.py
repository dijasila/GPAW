import numpy as np
from gpaw.utilities import pack, unpack
from gpaw.typing import Array1D, Array3D
from gpaw.setup import Setup
from gpaw.ase_interface import XCFunctional


class Hamiltonian:
    def __init__(self, layout, base, poisson_solver):
        setups = base.setups
        grid2 = base.grid2

        self.interpolate = layout.transformer(grid2)
        self.restrict = base.grid2.transformer(layout)

        fracpos = base.positions
        self.compensation_charges = setups.create_compensation_charges(
            grid2, fracpos)
        self.local_potentials = setups.create_local_potentials(layout, fracpos)
        self.poisson_solver = poisson_solver
        self.xc = base.xc
        self.v0 = grid2.zeros()
        self.local_potentials.add_to(self.v0)

    def calculate_potential(self,
                            density):
        density1 = density.density
        density2 = self.interpolate(density1)
        vxc = density2.new()
        e_xc = self.xc.calculate(density2, vxc)

        vext = density2.grid.empty()
        charge = vext.new(data=density2.data[:density.ndensities].sum(axis=0))
        e_zero = self.v0.integrate(charge)
        coefs = density.calculate_compensation_charge_coefficients()
        self.compensation_charges.add_to(charge, coefs)
        self.poisson_solver.solve(vext.data, charge.data)
        e_coulomb = 0.5 * vext.integrate(charge)

        potential2 = vxc
        potential2.data += vext.data + self.v0.data

        potential1 = self.restrict(potential2)
        e_kinetic = 0.0
        for s, (p, d) in enumerate(zip(potential1, density1)):
            e_kinetic -= p.integrate(d)
            if s < density.ndensities:
                e_kinetic += p.integrate(density.core_density)

        vnonloc, corrections = calculate_non_local_potential(
            density, self.xc,
            self.compensation_charges, vext)

        e_external = 0.0

        de_kinetic, de_coulomb, de_zero, de_xc, de_external = corrections
        energies = {'kinetic': e_kinetic + de_kinetic,
                    'coulomb': e_coulomb + de_coulomb,
                    'zero': e_zero + de_zero,
                    'xc': e_xc + de_xc,
                    'external': e_external + de_external}

        return potential1, vnonloc, energies


def calculate_non_local_potential(density, xc,
                                  compensation_charges, vext):
    coefs = compensation_charges.integrate(vext)
    vnonloc = density.density_matrices.new()
    energy_corrections = np.zeros(5)
    for a, D in density.density_matrices.items():
        Q = coefs[a]
        setup = density.setups[a]
        dH, energies = calculate_non_local_potential1(setup, xc, D, Q)
        vnonloc[a][:] = dH
        energy_corrections += energies

    return vnonloc, energy_corrections


def calculate_non_local_potential1(setup: Setup,
                                   xc: XCFunctional,
                                   D: Array3D,
                                   Q: Array1D) -> tuple[Array3D, Array1D]:
    ndensities = 2 if D.shape[2] == 2 else 1
    d = pack(D.T)
    d1 = d[:ndensities].sum(0)

    h1 = (setup.K_p + setup.M_p +
          setup.MB_p + 2.0 * setup.M_pp @ d1 +
          setup.Delta_pL @ Q)
    e_kinetic = setup.K_p @ d1 + setup.Kc
    e_zero = setup.MB + setup.MB_p @ d1
    e_coulomb = setup.M + d1 @ (setup.M_p + setup.M_pp @ d1)

    h = np.zeros_like(d)
    h[:ndensities] = h1
    e_xc = xc.calculate_paw_correction(setup, d, h)
    e_kinetic = -(d * h).sum().real

    e_external = 0.0

    H = unpack(h).T

    return H, np.array([e_kinetic, e_coulomb, e_zero, e_xc, e_external])
