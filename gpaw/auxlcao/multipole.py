from ase.neighborlist import PrimitiveNeighborList
import numpy as np

from gpaw.auxlcao.generatedcode import generated_W_LL,\
                                       generated_W_LL_screening


class LazySparseTensor:
    def __init__(self, name, indextypes):
        self.name = name
        self.indextypes = indextypes

    def get(self, index):
        if self.cache_i.haskey(index):
            return self.cache_i[index]
        else:
            value = self.evaluate(index)
            self.cache[index] = value
            return value

    def lazy_evaluate(self, index):
        raise NotImplementedError

    def get_index_list(self):
        raise NotImplementedError

class LazyDensityMatrix(LazySparseTensor):
    def __init__(self, rho_MM):
        LazySparseTensor.__init__('rho', 'MM')
        self.rho_MM = rho_MM

    def lazy_evaluate(self, index):
        M1 = slice( Mslices[index[0]], Mslices[index[0]+1] )
        M2 = slice( Mslices[index[0]], Mslices[index[0]+1] )
        return self.rho_MM[M1, M2]

    def get_index_list(self):
        a1a2_p = []
        for a in atoms:
            for a2 in atoms:
                a1a2_p.append((a1,a2))

class LazyRIProjection(LazySparseTensor):
    def __init__(self, calculator):
        LazySparseTensor.__init__('P', 'AMM')
        
    def lazy_evaluate(self, index):
        a1, a2, a3 = index
        return self.P_AMM(a1,a2,a3)

    def get_index_list(self):
        a1a2a3_p = []
        for a1, a2 in overlapping_pairs:
            a1a2a3_p.append((a1, a1, a2))
            if a1 != a2:
                a1a2a3_p.append((a2, a1, a2))
        return a1a2a3_p

def plan_meinsum(T1, index1, T2, index2, output):
    contraction_indices = list(set(index1)&set(index2))
    T1_i = [ index1.find(idx) for idx in contraction_indices ]
    T2_i = [ index2.find(idx) for idx in contraction_indices ]

    outref = []
    for idx in output:
        t1_idx = T1_i.find(idx)
        if t1_idx>=0:
            outref.append( (1, t1_idx) )
        else:
            t2_idx = T2_i.find(idx)
            outref.append( (2, t2_idx) )

    def get_out(indices1, indices2):
        s = []
        for id, index in outref:
            if id == 1:
                s.append(indices1[index])
            else: 
                s.append(indices2[index])
        return tuple(s)

    for indices1 in T1.get_indices():
        for indices2 in T1.get_indices():
            if indices1[T1_i] == indices2[T2_i]:
                out_indices = get_out(indices1, indices2)
                plan = ( out_indices, index1, index2, indices1, indices2, output )

    def dot(T1, T2):
        T3 = LazySparseTensor()
        ctr_str = '%s,%s->%s' % (index1, index2, output)

        #for out_indices, index1, index2, indices1, indices2, output in plan:
        #    np.einsum(ctr_str, T1.get(indices1), T2.get(indices2))

    return dot

class DensityMatrix:
    def __init__(self):
        rho_Ra1a2_MM = {}

        for a1, a2 in pairs:
            for offsets in offsets:
                rho_Ra1a2_MM [ (ox,oy,oz, a1, a2) ] = rho_MM

class MultipoleMatrixElements:
    def __init__(self):
        self.W_qLL = {}
        self.W_xLL = {} # Cached storage, indexed by offset tuple

    def update(cell_cv, spos_ac, pbc_c, bzq_qc, dtype, lcomp, omega=None):
        self.W_qLL = {}
        self.W_xLL = {}
        self.omega = omega
        if omega != 0.0:
            self.cutoff = 2.5 / self.omega
        else:
            self.cutoff = np.max(np.sum(cell_cv**2, axis=1))
        self.S = (self.lcomp+1)**2
        self.Na = len(self.spos_ac)
        W_LL = np.zeros((Na*S, Na*S))

        # Use ASE neighbour list to enumerate the supercells which need to be enumerated
        nl = PrimitiveNeighborList([ cutoff ], skin=0, 
                                   self_interaction=True,
                                   use_scaled_positions=True)

        nl.update(pbc=pbc_c, cell=cell_cv, coordinates=np.array([[0.5, 0.5, 0.5]]))
        a_a, self.disp_xc = nl.get_neighbors(0)

        # Calculate displacement vectors
        R_av = np.dot(spos_ac, cell_cv)
        dx = R_av[:, None, 0] - R_av[None, :, 0]
        dy = R_av[:, None, 1] - R_av[None, :, 1]
        dz = R_av[:, None, 2] - R_av[None, :, 2]

        for x, disp_c in enumerate(self.disp_xc):
            zero_disp = np.all(offset_c == 0)
            disp_v = np.dot(offset_c, cell_cv)

            # Diagonals will be done separately, just avoid division by zero here
            if zero_disp:
                dx1 = dx + 10*np.eye(Na)
                dy1 = dy + 10*np.eye(Na)
                dz1 = dz + 10*np.eye(Na)
            else:
                dx1 = dx + disp_v[0]
                dy1 = dy + disp_v[1]
                dz1 = dz + disp_v[2]

            d2 = dx1**2 + dy1**2 + dz1**2
            d = d2**0.5

            W_LL[:] = 0.0
            generated_W_LL_screening(W_LL, d, dx1, dy1, dz1, omega)
            W_LL *= 4*np.pi
            if zero_disp:
                get_W_LL_diagonals_from_setups(W_LL, lcomp, setups)

            self.W_xLL[offset_c] = W_LL.copy()


        if dtype == float:
            phase_q = np.array([1.0])
        else:
            phase_q = np.exp(2j*np.pi*np.dot(bzq_qc, offset_c))
            
        W_qLL += phase_q[:, None, None] * W_LL[None, :, :]

def get_W_LL_diagonals_from_setups(W_LL, lmax, setups):
    S = (lmax+1)**2
    for a, setup in enumerate(setups):
        W_LL[a*S:(a+1)*S:,a*S:(a+1)*S] = setup.W_LL[:S, :S]

def calculate_W_qLL(setups, cell_cv, spos_ac, pbc_c, bzq_qc, dtype, lcomp, coeff = 4*np.pi, omega=None):
    assert lcomp == 2

    Na = len(spos_ac)
    nq = len(bzq_qc)
    S = (lcomp+1)**2
    W_qLL = np.zeros( (nq, Na*S, Na*S), dtype=dtype )
    W_LL = np.zeros( (Na*S, Na*S), dtype=dtype )

    # Calculate displacement vectors
    R_av = np.dot(spos_ac, cell_cv)
    dx = R_av[:, None, 0] - R_av[None, :, 0]
    dy = R_av[:, None, 1] - R_av[None, :, 1]
    dz = R_av[:, None, 2] - R_av[None, :, 2]

    # Use ASE neighbour list to enumerate the supercells which need to be enumerated
    if omega != 0.0:
        cutoff = 2.5 / omega
    else:
        cutoff = np.max(np.sum(cell_cv**2, axis=1))
    nl = PrimitiveNeighborList([ cutoff ], skin=0, 
                               self_interaction=True,
                               use_scaled_positions=True)

    nl.update(pbc=pbc_c, cell=cell_cv, coordinates=np.array([[0.0, 0.0, 0.0]]))
    a_a, offset_ac = nl.get_neighbors(0)

    for offset_c in offset_ac:
        zero_disp = np.all(offset_c == 0)
        disp_v = np.dot(offset_c, cell_cv)

        # Diagonals will be done separately, just avoid division by zero here
        if zero_disp:
            dx1 = dx + 10*np.eye(Na)
            dy1 = dy + 10*np.eye(Na)
            dz1 = dz + 10*np.eye(Na)
        else:
            dx1 = dx + disp_v[0]
            dy1 = dy + disp_v[1]
            dz1 = dz + disp_v[2]

        d2 = dx1**2 + dy1**2 + dz1**2
        d = d2**0.5

        W_LL[:] = 0.0
        generated_W_LL_screening(W_LL, d, dx1, dy1, dz1, omega)
        W_LL *= coeff
        if zero_disp:
            get_W_LL_diagonals_from_setups(W_LL, lcomp, setups)

        if dtype == float:
            phase_q = np.array([1.0])
        else:
            phase_q = np.exp(2j*np.pi*np.dot(bzq_qc, offset_c))
            
        W_qLL += phase_q[:, None, None] * W_LL[None, :, :]

    return W_qLL


