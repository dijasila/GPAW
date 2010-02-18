# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as np

from gpaw import debug


class Symmetry:
    def __init__(self, id_a, cell_cv, pbc_c=np.ones(3, bool), tolerance=1e-11):
        """Symmetry object.

        Two atoms can only be identical if they have the same atomic
        numbers, setup types and magnetic moments.  If it is an LCAO
        type of calculation, they must have the same atomic basis
        set also."""

        self.id_a = id_a
        self.cell_cv = np.array(cell_cv, float)
        assert self.cell_cv.shape == (3, 3)
        self.pbc_c = np.array(pbc_c, bool)
        self.tol = tolerance

        self.op_scc = np.identity(3, int).reshape((1, 3, 3))
        
    def analyze(self, spos_ac):
        """Determine list of symmetry operations.

        First determine all symmetry operations of the cell.
        Then call prune_symmetries(spos_ac) to remove those symmetries that
        are not satisfied by the atoms.
        """

        self.op_scc = [] # Symmetry operations as matrices in 123 basis
        
        # Metric tensor
        metric_cc = np.dot(self.cell_cv, self.cell_cv.T)

        # Generate all possible 3x3 symmetry matrices using base-3 integers
        power = (6561, 2187, 729, 243, 81, 27, 9, 3, 1)

        # operation is a 3x3 matrix, with possible elements -1, 0, 1, thus
        # there are 3**9 = 19683 possible matrices
        for base3id in xrange(19683):
            op_cc = np.empty((3, 3), dtype=int)
            m = base3id
            for ip, p in enumerate(power):
                d, m = divmod(m, p)
                op_cc[ip // 3, ip % 3] = 1 - d

            # The metric of the cell should be conserved after applying
            # the operation
            opmetric_cc = np.dot(np.dot(op_cc, metric_cc), op_cc.T)
                                       
            if np.abs(metric_cc - opmetric_cc).sum() > self.tol:
                continue

            # Operation must not swap axes that are not both periodic:
            pbc_cc = np.logical_and.outer(self.pbc_c, self.pbc_c)
            if op_cc[~(pbc_cc | np.identity(3, bool))].any():
                continue

            # Operation must not invert axes that are not periodic:
            pbc_cc = np.logical_and.outer(self.pbc_c, self.pbc_c)
            if not (op_cc[np.diag(~self.pbc_c)] == 1).all():
                continue

            # operation is a valid symmetry of the unit cell
            self.op_scc.append(op_cc)

        self.op_scc = np.array(self.op_scc)
        
        # Check if symmetry operations are also valid when taking account
        # of atomic positions
        self.prune_symmetries(spos_ac)
        
    def prune_symmetries(self, spos_ac):
        """prune_symmetries(atoms)

        Remove symmetries that are not satisfied by the atoms."""

        # Build lists of (atom number, scaled position) tuples.  One
        # list for each combination of atomic number, setup type,
        # magnetic moment and basis set:
        species = {}
        for a, id in enumerate(self.id_a):
            spos_c = spos_ac[a]
            if id in species:
                species[id].append((a, spos_c))
            else:
                species[id] = [(a, spos_c)]

        opok = []
        maps = []
        # Reduce point group using operation matrices
        for op_cc in self.op_scc:
            map = np.zeros(len(spos_ac), int)
            for specie in species.values():
                for a1, spos1_c in specie:
                    spos1_c = np.dot(spos1_c, op_cc)
                    ok = False
                    for a2, spos2_c in specie:
                        sdiff = spos1_c - spos2_c
                        sdiff -= np.floor(sdiff + 0.5)
                        if np.dot(sdiff, sdiff) < self.tol:
                            ok = True
                            map[a1] = a2
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                opok.append(op_cc)
                maps.append(map)

        if debug:
            for op_cc, map_a in zip(opok, maps):
                for a1, id1 in enumerate(self.id_a):
                    a2 = map_a[a1]
                    assert id1 == self.id_a[a2]
                    spos1_c = spos_ac[a1]
                    spos2_c = spos_ac[a2]
                    sdiff = np.dot(spos1_c, op_cc) - spos2_c
                    sdiff -= np.floor(sdiff + 0.5)
                    assert np.dot(sdiff, sdiff) < self.tol

        self.maps = maps
        self.op_scc = np.array(opok)

    def check(self, spos_ac):
        """Check(positions) -> boolean

        Check if positions satisfy symmetry operations."""

        nsymold = len(self.op_scc)
        self.prune_symmetries(spos_ac)
        if len(self.op_scc) < nsymold:
            raise RuntimeError('Broken symmetry!')

    def reduce(self, bzk_kc):
        op_scc = self.op_scc
        inv_cc = -np.identity(3, int)
        have_inversion_symmetry = False
        for op_cc in op_scc:
            if (op_cc == inv_cc).all():
                have_inversion_symmetry = True
                break

        # Add inversion symmetry if it's not there:
        if not have_inversion_symmetry:
            op_scc = np.concatenate((op_scc, -op_scc))

        nbzkpts = len(bzk_kc)
        ibzk0_kc = np.empty((nbzkpts, 3))
        ibzk_kc = ibzk0_kc[:0]
        weight_k = np.ones(nbzkpts)
        nibzkpts = 0
        kbz = nbzkpts
        kibz_k = np.empty(nbzkpts, int)
        for k_c in bzk_kc[::-1]:
            kbz -= 1
            found = False
            for op_cc in op_scc:
                if len(ibzk_kc) == 0:
                    break
                b_k = ((np.dot(ibzk_kc, op_cc.T) - k_c)**2).sum(1) < self.tol
                if b_k.any():
                    found = True
                    kibz = np.where(b_k)[0][0]
                    weight_k[kibz] += 1.0
                    kibz_k[kbz] = kibz 
                    break
            if not found:
                kibz_k[kbz] = nibzkpts 
                nibzkpts += 1
                ibzk_kc = ibzk0_kc[:nibzkpts]
                ibzk_kc[-1] = k_c

        # Reverse order (looks more natural):
        self.kibz_k = nibzkpts - 1 - kibz_k
        return ibzk_kc[::-1].copy(), weight_k[:nibzkpts][::-1] / nbzkpts

    def symmetrize(self, a, gd):
        gd.symmetrize(a, self.op_scc)

    def symmetrize_forces(self, F0_av):
        F_ac = np.zeros_like(F0_av)
        for map_a, op_cc in zip(self.maps, self.op_scc):
            op_vv = np.dot(np.linalg.inv(self.cell_cv),
                           np.dot(op_cc, self.cell_cv))
            for a1, a2 in enumerate(map_a):
                F_ac[a2] += np.dot(F0_av[a1], op_vv)
        return F_ac / len(self.op_scc)
        
    def print_symmetries(self, text):
        n = len(self.op_scc)
        text('Symmetries present: %s' % n)
