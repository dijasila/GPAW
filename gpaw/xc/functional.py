import numpy as np
from gpaw.xc.kernel import XCKernel


class XCFunctional:
    orbital_dependent = False

    def __init__(self, name: str, type: str):
        self.name = name
        self.gd = None
        self.ekin = 0.0
        self.type = type
        self.kernel: XCKernel

    def todict(self):
        """Get dictionary representation of XC functional.

        This representation works for libxc kernels; other classes should
        likely override this function and should probably not rely on
        this implementation."""
        return {'type': self.kernel.type,
                'kernel': self.kernel.name}

    def tostring(self):
        """Get string representation of XC functional.

        This will give the name for libxc functionals but other data for
        hybrids."""
        return self.name

    def get_setup_name(self):
        return self.name

    def initialize(self, density, hamiltonian, wfs):
        pass

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        """Calculate energy and potential.

        gd: GridDescriptor
            Descriptor for 3-d grid.
        n_sg: rank-4 ndarray
            Spin densities.
        v_sg: rank-4 ndarray
            Array for potential.  The XC potential is added to the values
            already there.
        e_g: rank-3 ndarray
            Energy density.  Values must be written directly, not added.

        The total XC energy is returned."""

        if gd is not self.gd:
            self.set_grid_descriptor(gd)
        if e_g is None:
            e_g = gd.empty()
        if v_sg is None:
            v_sg = np.zeros_like(n_sg)
        self.calculate_impl(gd, n_sg, v_sg, e_g)
        return gd.integrate(e_g)

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        raise NotImplementedError

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        xcc = setup.xc_correction
        if xcc is None:
            return 0.0

        from gpaw.xc.noncollinear import NonCollinearLDAKernel
        collinear = not isinstance(self.kernel, NonCollinearLDAKernel)

        expander = self.get_radial_expander(setup, D_sp, xp=self.xp)
        rcalc = self.get_radial_calculator(xp=self.xp)
        E = 0
        for sign, ae in [(1.0, True), (-1.0, False)]:
            expansion = expander.expand(ae=ae, addcoredensity=addcoredensity)
            E += expansion.integrate(rcalc(expansion), sign=sign, dEdD_sp=dEdD_sp)

        if addcoredensity:
            E -= xcc.e_xc0
        return E 

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        nspins = len(n_sg)
        class RadialDensity:
            def __init__(self, rgd, n_sg):
                self.rgd = rgd
                self.n_sg = n_sg
                self.xc_correction = self
                self.B_pqL = np.eye(nspins).reshape((nspins, nspins, 1))
                self.n_qg = n_sg * (4 * np.pi)**0.5
                self.nc_g = 0

        expander = self.get_radial_expander(RadialDensity(rgd, n_sg), np.eye(nspins))
        expansion = expander.expand()
        potential = self.get_radial_calculator()(expansion)
        v_sg[:] = potential.dedn_sng[:,0,:]
        e_g = potential.e_ng[0]
        return expansion.integrate(potential)

    def get_radial_expander(self, setup, D_sp, xp=np):
        raise NotImplementedError

    def get_radial_calculator(self, xp=np):
        raise NotImplementedError

    def set_positions(self, spos_ac, atom_partition=None):
        pass

    def get_description(self):
        """Get long description of functional as a string, or None."""
        return None

    def summary(self, fd):
        """Write summary of last calculation to file."""
        pass

    def write(self, writer, natoms=None):
        pass

    def read(self, reader):
        pass

    def estimate_memory(self, mem):
        pass

    # Orbital dependent stuff:
    def apply_orbital_dependent_hamiltonian(self, kpt, psit_nG,
                                            Htpsit_nG, dH_asp=None):
        pass

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        # In what sense?  Some documentation here maybe?
        pass

    def add_correction(self, kpt, psit_xG, R_xG, P_axi, c_axi, n_x=None,
                       calculate_change=False):
        # Which kind of correction is this?  Maybe some kind of documentation
        # could be written?  What is required of an implementation?
        pass

    def rotate(self, kpt, U_nn):
        pass

    def get_kinetic_energy_correction(self):
        return self.ekin

    def add_forces(self, F_av):
        pass

    def stress_tensor_contribution(self, n_sg, skip_sum=False):
        raise NotImplementedError('Calculation of stress tensor is not ' +
                                  f'implemented for {self.name}')
