import warnings

import numpy as np
from gpaw.mpi import MPIComm
from gpaw.new import zips
from gpaw.new.brillouin import IBZ, BZPoints
from gpaw.rotation import rotation
from gpaw.symmetry import Symmetry as OldSymmetry


class GridSymmetrizationPlan:
    def __init__(self, xp, gd, op_scc, translation_sc):
        self.xp = xp
        offset_c = xp.asarray((1 - gd.pbc_c).astype(np.int32))
        if hasattr(gd, 'N_c'):
            size_c = gd.N_c
        else:
            size_c = gd.size_c
        size_c = xp.asarray(size_c)
        print('xp:', xp, size_c)

        if hasattr(gd, 'myshape'):
            realsize_c = gd.myshape
        else:
            realsize_c = gd.end_c - gd.beg_c
        translation_sc = xp.asarray(translation_sc) 
        I_cR = xp.indices(realsize_c, dtype=np.int32).reshape((3, -1)) + offset_c[:, None]
        t_sc = xp.asarray((translation_sc * size_c).round().astype(np.int32))
        I_csR = xp.zeros((3, t_sc.shape[0], I_cR.shape[1]), dtype=np.int32)
        def mat33():
            for s in range(t_sc.shape[0]):
                for c1 in range(3):
                    for c2 in range(3):
                        op = op_scc[s, c1, c2]
                        if op == 1:
                            I_csR[c2, s, :] += I_cR[c1, :]
                        elif op == -1:
                            I_csR[c2, s, :] -= I_cR[c1, :]
                        else:
                            assert op == 0
        mat33()
        I_csR -= t_sc.T[:, :, None]
        #I_Rsc = np.einsum('sCc,CR->Rsc', op_scc, I_cR) - t_sc
        I_csR = I_csR  % size_c[:, None, None] - offset_c[:, None, None]
        R_Rs = xp.ravel_multi_index(I_csR.transpose([0, 2, 1]), realsize_c)
        #self.R_Zs_old = xp.asarray(np.unique(np.sort(R_Rs, axis=1), axis=0))
        _, R_Z = xp.unique(xp.min(R_Rs, axis=1), return_index=True)
        self.R_Zs =R_Rs[R_Z, :]
        #print(self.R_Zs_old.shape)
        #print(self.R_Zs.shape)
        #if not np.allclose(self.R_Zs_old, self.R_Zs):
        #    for i, (a, b) in enumerate(zip(self.R_Zs_old, self.R_Zs)):
        #        print(i,a,b)
        #    asd
        self.factor = 1.0 / len(op_scc)

    def apply(self, n_sR):
        for n_R in n_sR:
            n_R = n_R.ravel()
            n_Z = self.xp.sum(n_R[self.R_Zs], axis=1) * self.factor
            n_R[:] = 0.0
            # On purpose using += which doesn't add same indices twice!
            n_R[self.R_Zs] += n_Z[:, None]

    def reduce_grid(self, arrays):
        return tuple([self.reduce_array(arr) for arr in arrays])

    def extend_grid(self, arrays_out, arrays_in):
        for array_out, array_in in zip(arrays_out, arrays_in):
            self.extend_array(array_out, array_in)

    def extend_array(self, array_out, array_in):
        if len(array_in.shape) == 1:
            array_out[:] = 0
            array_out.ravel()[self.R_Zs] += array_in[:, None]
        else:
            print(array_out.shape, array_in.shape,'out in saheps')
            for arr_out, arr_in in zip(array_out, array_in):
                self.extend_array(arr_out, arr_in)

    def reduce_array(self, array):
        if len(array.shape) == 3:
            array = array.ravel()
            array_Z = self.xp.sum(array[self.R_Zs], axis=1) * self.factor
            print('Reduced shape', array_Z.shape)
            return array_Z
        else:
            array_sZ = self.xp.zeros( tuple(array.shape[:-3]) + (self.R_Zs.shape[0],))
            assert len(array.shape) == 4
            for s, arr_R in enumerate(array):
                array_sZ[s] = self.reduce_array(arr_R)
            print('Redued shape4', array_sZ.shape)
            return array_sZ



class SymmetrizationPlan:
    def __init__(self, xp, rotations, a_sa, l_aj, layout):
        ns = a_sa.shape[0]  # Number of symmetries
        na = a_sa.shape[1]  # Number of atoms

        if xp is np:
            import scipy
            sparse = scipy.sparse
        else:
            from gpaw.gpu import cupyx
            sparse = cupyx.scipy.sparse

        # Find orbits, i.e. point group action,
        # which also equals to set of all cosets.
        # In practical terms, these are just atoms which map
        # to each other via symmetry operations.
        # Mathematically {{as: s∈ S}: a∈ A}, where a is an atom.
        cosets = {frozenset(a_sa[:, a]) for a in range(na)}

        S_aZZ = {}
        work = []
        for coset in map(list, cosets):
            nA = len(coset)  # Number of atoms in this orbit
            a = coset[0]  # Representative atom for coset

            # The atomic density matrices transform as
            # ρ'_ii = R_sii ρ_ii R^T_sii
            # Which equals to vec(ρ'_ii) = (R^s_ii ⊗  R^s_ii) vec(ρ_ii)
            # Here we to the Kronecker product for each of the
            # symmetry transformations.
            R_sii = xp.asarray(rotations(l_aj[a], xp))
            i2 = R_sii.shape[1]**2
            R_sPP = xp.einsum('sab,scd->sacbd', R_sii, R_sii)
            R_sPP = R_sPP.reshape((ns, i2, i2)) / ns

            S_ZZ = xp.zeros((nA * i2,) * 2)

            # For each orbit, the symetrization operation is represented by
            # a full matrix operating on a subset of indices to the full array.
            for loca1, a1 in enumerate(coset):
                Z1 = loca1 * i2
                Z2 = Z1 + i2
                for s, a2 in enumerate(a_sa[:, a1]):
                    loca2 = coset.index(a2)
                    Z3 = loca2 * i2
                    Z4 = Z3 + i2
                    S_ZZ[Z1:Z2, Z3:Z4] += R_sPP[s]
            # Utilize sparse matrices if sizes get out of hand
            # Limit is hard coded to 100MB per orbit
            if S_ZZ.nbytes > 100 * 1024**2:
                S_ZZ = sparse.csr_matrix(S_ZZ)
            S_aZZ[a] = S_ZZ
            indices = []
            for loca1, a1 in enumerate(coset):
                a1_, start, end = layout.myindices[a1]
                # When parallelization is done, this needs to be rewritten
                assert a1_ == a1
                for X in range(i2):
                    indices.append(start + X)
            work.append((a, xp.array(indices)))

        self.work = work
        self.S_aZZ = S_aZZ
        self.xp = xp


    def apply(self, source, target):
        total = 0
        for a, ind in self.work:
            for spin in range(len(source)):
                total += len(ind)
                target[spin, ind] = self.S_aZZ[a] @ source[spin, ind]
        #assert total == source.shape[1]


def create_symmetries_object(atoms, ids=None, magmoms=None, parameters=None):
    ids = ids or [()] * len(atoms)
    if magmoms is None:
        pass
    elif magmoms.ndim == 1:
        ids = [id + (m,) for id, m in zips(ids, magmoms)]
    else:
        ids = [id + tuple(m) for id, m in zips(ids, magmoms)]
    symmetry = OldSymmetry(ids,
                           atoms.cell.complete(),
                           atoms.pbc,
                           **(parameters or {}))
    symmetry.analyze(atoms.get_scaled_positions())
    return Symmetries(symmetry)


def mat(rot_cc) -> str:
    """Convert 3x3 matrix to str.

    >>> mat([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    '[[-1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]'

    """
    return '[[' + '], ['.join(', '.join(f'{r:2}'
                                        for r in rot_c)
                              for rot_c in rot_cc) + ']]'


class Symmetries:
    """Wrapper for old symmetry object.  Exposes what we need now."""
    def __init__(self, symmetry):
        self.symmetry = symmetry
        self.rotation_scc = symmetry.op_scc
        self.translation_sc = symmetry.ft_sc
        self.a_sa = symmetry.a_sa
        cell_cv = symmetry.cell_cv
        self.rotation_svv = np.einsum('vc, scd, dw -> svw',
                                      np.linalg.inv(cell_cv),
                                      self.rotation_scc,
                                      cell_cv)
        self.rotation_lsmm = [
            np.array([rotation(l, r_vv) for r_vv in self.rotation_svv])
            for l in range(4)]
        self._rotations = {}

        self._gdcache = {}

    def reduce_grid(self, gd, arrays):
        import cupy
        if gd not in self._gdcache:
            self._gdcache[gd] = GridSymmetrizationPlan(cupy, gd, self.rotation_scc, self.translation_sc)
        return self._gdcache[gd].reduce_grid(arrays)

    def extend_grid(self, gd, arrays_out, arrays_in):
        import cupy
        if gd not in self._gdcache:
            self._gdcache[gd] = GridSymmetrizationPlan(cupy, gd, self.rotation_scc, self.translation_sc)
        return self._gdcache[gd].extend_grid(arrays_out, arrays_in)

    def __len__(self):
        return len(self.rotation_scc)

    def __str__(self):
        lines = ['symmetry:',
                 f'  number of symmetries: {len(self)}']
        if self.symmetry.symmorphic:
            lines.append('  rotations: [')
            for rot_cc in self.rotation_scc:
                lines.append(f'    {mat(rot_cc)},')
        else:
            nt = self.translation_sc.any(1).sum()
            lines.append(f'  number of symmetries with translation: {nt}')
            lines.append('  rotations and translations: [')
            for rot_cc, t_c in zips(self.rotation_scc, self.translation_sc):
                a, b, c = t_c
                lines.append(f'    [{mat(rot_cc)}, '
                             f'[{a:6.3f}, {b:6.3f}, {c:6.3f}]],')
        lines[-1] = lines[-1][:-1] + ']\n'
        return '\n'.join(lines)

    def reduce(self,
               bz: BZPoints,
               comm: MPIComm = None,
               strict: bool = True) -> IBZ:
        """Find irreducible set of k-points."""
        if not (self.symmetry.time_reversal or
                self.symmetry.point_group):
            N = len(bz)
            return IBZ(self,
                       bz,
                       ibz2bz=np.arange(N),
                       bz2ibz=np.arange(N),
                       weights=np.ones(N) / N)

        (_, weight_k, sym_k, time_reversal_k, bz2ibz_K, ibz2bz_k,
         bz2bz_Ks) = self.symmetry.reduce(bz.kpt_Kc, comm)

        if -1 in bz2bz_Ks:
            msg = 'Note: your k-points are not as symmetric as your crystal!'
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)

        return IBZ(self, bz, ibz2bz_k, bz2ibz_K, weight_k)

    def check_positions(self, fracpos_ac):
        self.symmetry.check(fracpos_ac)

    def symmetrize_forces(self, F_av):
        return self.symmetry.symmetrize_forces(F_av)

    def rotations(self, l_j, xp=np):
        ells = tuple(l_j)
        rotation_sii = self._rotations.get(ells)
        if rotation_sii is None:
            ni = sum(2 * l + 1 for l in l_j)
            rotation_sii = np.zeros((len(self), ni, ni))
            i1 = 0
            for l in l_j:
                i2 = i1 + 2 * l + 1
                rotation_sii[:, i1:i2, i1:i2] = self.rotation_lsmm[l]
                i1 = i2
            rotation_sii = xp.asarray(rotation_sii)
            self._rotations[ells] = rotation_sii
        return rotation_sii
