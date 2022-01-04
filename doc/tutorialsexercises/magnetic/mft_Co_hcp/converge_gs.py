"""Converge ground state of Co(hcp) and save with all plane-wave components

A number of unoccupied bands are converged for the sake of subsequent
response calculations.

Note : k must be a multiple of 6 in order to do computations at all the
high-symmetry points of the hcp lattice.

Note : Co(hcp) has 2 atoms in the unit cell, each with 9 valence electrons
which fully or partially occupy 6 bands per atom. Hence there are 12 occupied
bands, so nbands_gs >= nbands_response > 12 is required"""