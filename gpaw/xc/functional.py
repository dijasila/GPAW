import numpy as np
from gpaw.xc.kernel import XCKernel
from gpaw.utilities import pack2, unpack, unpack2, pack

class XCFunctional:
    orbital_dependent = False

    def __init__(self, name, type):
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

    def calculate_paw_corrections(self, setups, D_asii, dEdD_asii, addcoredensity=True):
        by_species = {}
        by_atom = []
        id_by_atom = []

        for atom_id, setup in enumerate(setups):
            species_id = setups.id_a[atom_id]
            if species_id not in by_species:
                by_species[species_id] = setup
            by_atom.append(by_species[species_id])
            id_by_atom.append(species_id)

        E = 0
        ekin = 0
        for sid, setup in by_species.items():
            D_asp = self.xp.array([ self.xp.array([ self.xp.asarray(pack(self.xp.asnumpy(D_ii).real)) for D_ii in D_asii[a]]) for a, s in enumerate(id_by_atom) if s == sid])
            dEdD_asp = self.xp.zeros_like(D_asp)
            expander = self.get_radial_expander(setup, D_asp, xp=self.xp)
            rcalc = self.get_radial_calculator(xp=self.xp)
            E = 0
            for sign, ae in [(1.0, True), (-1.0, False)]:
                expansion = expander.expand(ae=ae, addcoredensity=addcoredensity)
                Ein = expansion.integrate(rcalc(expansion), sign=sign, dEdD_asp=dEdD_asp)
                E += Ein

            print(dEdD_asp.shape,'123')
            a2 = 0
            for a, s in enumerate(id_by_atom):
                if s != sid:
                    continue
                dEdD_asii[a] += self.xp.asarray(unpack(self.xp.asnumpy(dEdD_asp[a2])))
                a2 += 1

            if addcoredensity:
                E -= setup.xc_correction.e_xc0 * len(dEdD_asp)
            ekin -= (dEdD_asp * D_asp).sum().real
        return float(E), float(ekin)

        for setup, D_sii, dEdD_sii in zip(setups, D_asii, dEdD_asii):
            if self.xp is not np:
                D_sp = self.xp.array([ self.xp.asarray(pack(self.xp.asnumpy(D_ii).real)) for D_ii in D_sii])
            else:
                D_sp = self.xp.array([ pack(D_ii.real) for D_ii in D_sii])
            dEdD_sp = self.xp.zeros_like(D_sp)
            print('D_sp in', D_sp)
            #E += self.calculate_paw_correction(setup, D_sp, dEdD_sp)
            
            if self.xp is not np:
                H_sii = self.xp.asarray(unpack(self.xp.asnumpy(dEdD_sp))) 
                dEdD_sii += self.xp.asarray(unpack(self.xp.asnumpy(dEdD_sp)))
                ekin -= (H_sii * self.xp.asarray(D_sii)).sum().real
            else:
                ekin -= (unpack(dEdD_sp) * D_sii).sum().real
                dEdD_sii[:] += unpack(dEdD_sp)

        return float(E), float(ekin)

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
            Ein = expansion.integrate(rcalc(expansion), sign=sign, dEdD_sp=dEdD_sp)
            print(sign*Ein, sign)
            E += Ein

        if addcoredensity:
            E -= xcc.e_xc0
        print('final dEdD', dEdD_sp)
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
