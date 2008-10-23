# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as npy

from gpaw import debug


class Symmetry:
    def __init__(self, Z_a, type_a, magmom_a, basis_a,
                 domain, tolerance=1e-11, magmom_decimals=3):
        """Symmetry object.

        Two atoms can only be identical if they have the same atomic
        numbers, setup types and magnetic moments.  If it is an LCAO
        type of calculation, they must have the same atomic basis
        set also."""

        # Round off:
        magmom_a = magmom_a.round(decimals=magmom_decimals)
        
        self.ZTMB_a = zip(Z_a, type_a, magmom_a, basis_a)
        self.cell_c = domain.cell_c
        self.pbc_c = domain.pbc_c
        self.scale_position = domain.scale_position  # XXX ref to domain!
        self.tol = tolerance

    def analyze(self, pos_ac):
        """Analyse(atoms)

        Find a list of symmetry operations."""

                    
        # There are six orderings of the axes:
        allpossibleswaps = [(0, 1, 2), (0, 2, 1),
                            (1, 0, 2), (1, 2, 0),
                            (2, 0, 1), (2, 1, 0)]
        # Only swap axes of equal length:
        cellsyms = [[abs(npy.vdot(self.cell_c[c1],self.cell_c[c1])-npy.vdot(self.cell_c[c2],self.cell_c[c2]))<self.tol and
                     self.pbc_c[c1] and self.pbc_c[c2]
                     for c1 in range(3)]
                    for c2 in range(3)]
        swaps = []
        for swap in allpossibleswaps:
            ok = True
            for c1, c2 in enumerate(swap):
                if c1 == c2 or cellsyms[c1][c2]:
                    continue
                else:
                    ok = False
                    break
            if ok:
                swaps.append(swap)

        mirrors = [[1.0], [1.0], [1.0]]
        for c in range(3):
            if self.pbc_c[c]:
                mirrors[c].append(-1.0)
        mirrors = [npy.array((m0, m1, m2))
                   for m0 in mirrors[0]
                   for m1 in mirrors[1]
                   for m2 in mirrors[2]]

        self.symmetries = [] #symmetry operations as pairs of swaps and mirrors
        self.operations = [] #symmetry operations as matrices
        cell_cdt=npy.dot(npy.transpose(self.cell_c),self.cell_c) #metric tensor

        #make (orthogonal) operation matrix out of every swap/operation pair
        for swap in swaps:
            for mirror in mirrors:
                operation=[[1,0,0],[0,1,0],[0,0,1]]

                for i1 in range(3):
                    operation[i1]=npy.take(operation[i1]*mirror,swap)

                #generalized criterion of a matrix being a symmetry operation 
                cell_cdo  =npy.dot(self.cell_c,operation)
                cell_cdodt=npy.dot(npy.transpose(cell_cdo),cell_cdo)

                if not npy.sometrue(cell_cdt-cell_cdodt):
                    self.operations.append(operation)
                    
        self.prune_symmetries(pos_ac)

    def prune_symmetries(self, pos_ac):
        """prune_symmetries(atoms)

        Remove symmetries that are not satisfied."""

        # Build lists of (atom number, scaled position) tuples.  One
        # list for each combination of atomic number, setup type,
        # magnetic moment and basis set:
        species = {}
        for a, ZTMB in enumerate(self.ZTMB_a):
            spos_c = self.scale_position(pos_ac[a])
            if species.has_key(ZTMB):
                species[ZTMB].append((a, spos_c))
            else:
                species[ZTMB] = [(a, spos_c)]

        opok = []
        maps = []
        #reduce point group using operation matrices
        for ioperation, operation in enumerate(self.operations):
            map = npy.zeros(len(pos_ac), int)
            for specie in species.values():
                for a1, spos1_c in specie:
                    spos1_c = npy.dot(operation,spos1_c)
                    ok = False
                    for a2, spos2_c in specie:
                        sdiff = spos1_c - spos2_c
                        sdiff -= npy.floor(sdiff + 0.5)
                        if npy.dot(sdiff, sdiff) < self.tol:
                            ok = True
                            map[a1] = a2
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                opok.append(operation)
                maps.append(map)

        if debug:
            for symmetry, map in zip(symmok, maps):
                swap, mirror = symmetry
                for a1, ZTMB1 in enumerate(self.ZTMB_a):
                    a2 = map[a1]
                    assert ZTMB1 == self.ZTMB_a[a2]
                    spos1_c = self.scale_position(pos_ac[a1])
                    spos2_c = self.scale_position(pos_ac[a2])
                    sdiff = npy.take(spos1_c * mirror, swap) - spos2_c
                    sdiff -= npy.floor(sdiff + 0.5)
                    assert npy.dot(sdiff, sdiff) < self.tol

        self.maps = maps
        self.operations = opok
                
    def check(self, pos_ac):
        """Check(positions) -> boolean

        Check if positions satisfy symmetry operations."""

        nsymold = len(self.operations)
        self.prune_symmetries(pos_ac)
        if len(self.operations) < nsymold:
            raise RuntimeError('Broken symmetry!')

    def reduce(self, bzk_kc):
        # Add inversion symmetry if it's not there:
        have_inversion_symmetry = False
        identity=npy.identity(3).ravel()
        for operation in self.operations:
            if sum(abs(npy.array(operation).ravel()+identity))<self.tol:
                have_inversion_symmetry = True
                break
        nsym = len(self.operations)
        if not have_inversion_symmetry:
            for operation in self.operations[:nsym]:
                self.operations.append(npy.negative(operation))

        nbzkpts = len(bzk_kc)
        ibzk0_kc = npy.empty((nbzkpts, 3))
        ibzk_kc = ibzk0_kc[:0]
        weight_k = npy.ones(nbzkpts)
        nibzkpts = 0
        for k_c in bzk_kc[::-1]:
            found = False
            for operation in self.operations:
                if len(ibzk_kc)==0:
                    break
                opit=npy.transpose(npy.linalg.inv(operation))
                d_kc = [npy.dot(opit,ibzk_kc[i1]) for i1 in range(len(ibzk_kc))] - k_c
                d_kc *= d_kc
                d_k = d_kc.sum(1) < self.tol
                if d_k.any():
                    found = True
                    weight_k[:nibzkpts] += d_k
                    break
            if not found:
                nibzkpts += 1
                ibzk_kc = ibzk0_kc[:nibzkpts]
                ibzk_kc[-1] = k_c

        del self.operations[nsym:]
        self.symmetries=self.convert_operations(self.operations)

        return ibzk_kc[::-1].copy(), weight_k[:nibzkpts][::-1] / nbzkpts

    def convert_operations(self,operations):
        #create pairs of mirrors and swaps for (orthogonal) matrices
        symmetries=[]
        for operation in self.operations:
            if not npy.sometrue(npy.dot(operation,npy.transpose(operation))-npy.identity(3)):
                swap_c,mirror_c=self.break_operation(operation)
                symmetries.append((swap_c,mirror_c))
            else:
                symmetries.append(operation)
        return symmetries

    def break_operation(self,operation):
        #break an (orthogonal) matrix to swaps and mirrors
        swap=[0,0,0]; mirror=npy.array([0.,0.,0.])
        for i1 in range(3):
            for i2 in range(3):
                if abs(operation[i1][i2])>0:
                    swap[i1]=i2
                    mirror[i2]=operation[i1][i2]
        return (tuple(swap),mirror)
                                                                                                                                        
    def symmetrize(self, a, gd):
        b = a.copy()
        a[:] = 0.0
        for swap, mirror in self.symmetries:
            d = b
            for c, m in enumerate(mirror):
                if m == -1:
                    d = gd.mirror(d, c)
            a += gd.swap_axes(d, swap)
        a /= len(self.symmetries)

    def print_symmetries(self, text):
        n = len(self.operations)
        if n == 48:
            text('symmetries: all')
            return
        line1 = []
        line2 = []
        for swap, mirror in self.symmetries:
            line1.extend(['_  '[int(s) + 1] for s in mirror] + [' '])
            line2.extend(['XYZ'[c] for c in swap] + [' '])
        line1 = ''.join(line1)
        line2 = ''.join(line2)
        n1 = 0
        n2 = 64
        text('symmetries:')
        while n1 < 4 * n:
            text('%s\n%s\n' % (line1[n1:n2], line2[n1:n2]), end='')
            n1 = n2
            n2 += 64
