    def calculate_magnetic_moments(self):
        """Calculate the local magnetic moments within augmentation spheres.
        Local magnetic moments are scaled to sum up to the total magnetic
        moment"""

        if self.nspins == 2:
            self.density.calculate_local_magnetic_moments()
            for a, nucleus in enumerate(self.nuclei):
                self.magmom_a[a] = nucleus.mom
            # scale the moments to sum up tp the total magnetic moment
            M = self.magmom_a.sum()
            if abs(M) > 1e-4:
                scale = self.occupation.magmom / M
                self.magmom_a *= scale

