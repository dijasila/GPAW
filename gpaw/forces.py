class ForceCalculator:
    """``F_ac``    Forces.
    """
    def reset(self):
        self.F_ac = None

    def calculate_forces(self):
        """Return the atomic forces."""

        if self.F_ac is not None:
            return

        self.F_ac = np.empty((self.natoms, 3))

        self.density.update(self.wfs, self.symmetry)
        self.update_kinetic()
        self.hamiltonian.update(self.density)
        
        nt_g = self.density.nt_g
        vt_sG = self.hamiltonian.vt_sG
        vHt_g = self.hamiltonian.vHt_g

        if self.nspins == 2:
            vt_G = 0.5 * (vt_sG[0] + vt_sG[1])
        else:
            vt_G = vt_sG[0]

        for nucleus in self.my_nuclei:
            nucleus.F_c[:] = 0.0

        # Calculate force-contribution from k-points:
        for kpt in self.wfs.kpt_u:
            for nucleus in self.pt_nuclei:
                # XXX
                if self.eigensolver.lcao:
                    nucleus.calculate_force_kpoint_lcao(kpt, self.hamiltonian)
                else:
                    nucleus.calculate_force_kpoint(kpt)
        for nucleus in self.my_nuclei:
            self.kpt_comm.sum(nucleus.F_c)
            self.band_comm.sum(nucleus.F_c)

        for nucleus in self.nuclei:
            nucleus.calculate_force(vHt_g, nt_g, vt_G)

        # Global master collects forces from nuclei into self.F_ac:
        if self.master:
            for a, nucleus in enumerate(self.nuclei):
                if nucleus.in_this_domain:
                    self.F_ac[a] = nucleus.F_c
                else:
                    self.domain.comm.receive(self.F_ac[a], nucleus.rank, 7)
        else:
            if self.kpt_comm.rank == 0 and self.band_comm.rank == 0:
                for nucleus in self.my_nuclei:
                    self.domain.comm.send(nucleus.F_c, MASTER, 7)

        # Broadcast the forces to all processors
        self.world.broadcast(self.F_ac, MASTER)

        # Add non-local contributions
        for kpt in self.wfs.kpt_u:
            self.F_ac += self.xcfunc.get_non_local_force(kpt)
    
        # Add contributions from external fields
        external = self.input_parameters['external']
        if hasattr(external, 'get_ion_energy_and_forces'):
            E_ext, F_ext = external.get_ion_energy_and_forces(self.atoms)
            self.F_ac += F_ext

        if self.symmetry is not None:
            # Symmetrize forces:
            F_ac = np.zeros((self.natoms, 3))
            for map_a, symmetry in zip(self.symmetry.maps,
                                       self.symmetry.symmetries):
                swap, mirror = symmetry
                for a1, a2 in enumerate(map_a):
                    F_ac[a2] += np.take(self.F_ac[a1] * mirror, swap)
            self.F_ac[:] = F_ac / len(self.symmetry.symmetries)
        
        self.print_forces()

