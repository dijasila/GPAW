import numpy as np
from gpaw.auxlcao.paw import paw_exx_correction

"""
for (a1, a2, R) in fock.blocks_i:    


class Contraction:
    def __init__(self, contraction_str, *args):
        for index, block in source:
            

Contraction('kl,Aik,AB,Bjl->ij', RhoSource, PSource, WSource, PSource,  
 SparseDestination([ ((a1,0), (a2, R2)) ] ))

"""

class SparseTensor:
    def __init__(self, name, indextypes):
        self.name = name
        self.indextypes = indextypes
        self.zero()
        self.meinsum = meinsum

    def zero(self):
        self.block_i = defaultdict(float)

    def to_full2d(self, M1_a, M2_a):
        T_MM = np.zeros( (M1_a[-1], M2_a[-1]) )
        for index, block_xx in self.block_i.items():
            a1, a2 = index
            T_MM[ M1_a[a1]:M1_a[a1+1], M2_a[a2]:M2_a[a2+1] ] += block_xx
        return T_MM

    def to_full3d(self, M1_a, M2_a, M3_a):
        T_MMM = np.zeros( (M1_a[-1], M2_a[-1], M3_a[-1]) )
        for index, block_xx in self.block_i.items():
            a1, a2,a3 = index
            T_MMM[ M1_a[a1]:M1_a[a1+1], M2_a[a2]:M2_a[a2+1], M3_a[a3]:M3_a[a3+1] ] += block_xx
        return T_MMM

    def to_full3d_R(self, M1_a, M2_a, M3_a):
        T_MMM = np.zeros( (M1_a[-1], M2_a[-1], M3_a[-1]) )
        for index, block_xx in self.block_i.items():
            print(index)
            (a1, R1), (a2, R2),(a3, R3) = index
            T_MMM[ M1_a[a1]:M1_a[a1+1], M2_a[a2]:M2_a[a2+1], M3_a[a3]:M3_a[a3+1] ] += block_xx
        return T_MMM

    def __iadd__(self, index_and_block):
        if isinstance(index_and_block, SparseTensor):
            for index, block_xx in index_and_block.block_i.items():
                self.block_i[index] += block_xx.copy()
        else:
            index, block_xx = index_and_block
            print('added', index, block_xx, self.name)
            self.block_i[index] += block_xx.copy() #xxx

        return self

    def get(self, index):
        if index in self.block_i:
            return self.block_i[index]

    def get_name(self):
        return '%s_%s' % (self.name, self.indextypes)

    def show(self):
        s = self.get_name() + ' {\n'
        for i, block in self.block_i.items():
            s += '   ' + repr(i) + ' : ' + repr(block.shape) + ' ' + repr(block.dtype) + ' ' + repr(block) +'\n'
        s += '}\n'
        print(s)


    def __repr__(self):
        s = self.get_name() + ' {\n'
        for i, block in self.block_i.items():
            s += '   ' + repr(i) + ' : ' + repr(block.shape) + ' ' + repr(block.dtype) + '\n'
        s += '}\n'
        return s

def meinsum(output_name, index_str, T1, T2, timer=None):
    input_index_str, output_index_str = index_str.split('->')
    index1, index2 = input_index_str.split(',')
    contraction_indices = list(set(index1)&set(index2))

    outref = []
    output_index_types = ''
    for idx in output_index_str:
        t1_idx = index1.find(idx)
        if t1_idx>=0:
            outref.append( (1, t1_idx) )
            output_index_types += T1.indextypes[t1_idx]
        else:
            t2_idx = index2.find(idx)
            outref.append( (2, t2_idx) )
            output_index_types += T2.indextypes[t2_idx]

    T3 = SparseTensor(output_name, output_index_types)

    #print('%s[%s] = %s[%s] * %s[%s] ' % (T3.get_name(), output_index_str, T1.get_name(), index1, T2.get_name(), index2))

    T1_i = tuple([ index1.find(idx) for idx in contraction_indices ])
    T2_i = tuple([ index2.find(idx) for idx in contraction_indices ])

    def get_out(indices1, indices2):
        s = []
        for id, index in outref:
            if id == 1:
                s.append(indices1[index])
            else:
                s.append(indices2[index])
        return tuple(s)

    for i1, block1 in T1.block_i.items():
        for i2, block2 in T2.block_i.items():
            fail = False
            for ii1, ii2 in zip(T1_i, T2_i):
                if i1[ii1] != i2[ii2]:
                    fail = True
            if fail:
                continue
            out_indices = get_out(i1, i2)
            #print('Einsum', i1, i2, index_str, block1.shape, block2.shape)
            #print('in',block1, 'in2', block2)
            if timer:
                with timer('einsum'):
                    value = np.einsum(index_str, block1, block2)
            else:
                value = np.einsum(index_str, block1, block2)
            #print('out', np.max(np.abs(value)))
            #input('prod')
            T3 += out_indices, value
    return T3



class RIPBC:
    def __init__(self, exx_fraction=None, screening_omega=None, N_k = (3,3,3), threshold=0.02, laux=2, lcomp=2):
        self.name = 'RI-PBC'
        self.exx_fraction = exx_fraction
        self.screening_omega = screening_omega
        self.fix_rho = False

        self.sdisp_Rc = []
        for x in range(-N_k[0]//2, N_k[0]//2+1):
            for y in range(-N_k[0]//2, N_k[0]//2+1):
                for z in range(-N_k[0]//2, N_k[0]//2+1):
                    self.sdisp_Rc.append( (x,y,z) )
        print('Real space cell displacements', self.sdisp_Rc)

        self.P_AMM = EvaluateAll_PAMM

    def initialize(self, density, hamiltonian, wfs):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.timer = hamiltonian.timer
        self.prepare_setups(density.setups)

    def prepare_setups(self, setups):
        if self.screening_omega != 0.0:
            print('Screening omega for setups')
            for setup in setups:
                setup.ri_M_pp = setup.calculate_screened_M_pp(self.screening_omega)
                setup.ri_X_p = setup.HSEX_p
        else:
            for setup in setups:
                setup.ri_M_pp = setup.M_pp
                setup.ri_X_p = setup.X_p


    def nlxc(self,
             H_MM,
             dH_asp,
             wfs,
             kpt, yy):

        #H_MM += kpt.exx_V_MM * yy
        pass

    def F_MM(self, a1, a2, R2):
        for (a3, a4, a5), locP1_AMM in P_AMM.blocks(M1=a1):
            assert a4 == a1
            for (a6, a7, a8), locP2_AMM in P_AMM.blocks(M1=a2):
                locW_AA = W_AA(a3, a6)
                rho_MM(a5, a8)

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac
        self.my_atoms = range(len(self.spos_ac))

        self.fock_matrix = RealSpaceFockMatrix()
        self.fock_matrix.prepare_evaluation(a1, a2, R)

        

        # F_MM( (a1, R1), (a2, R2) ) = 
        #    P_AMM( (a1 or a3), a1, a3 ) W_AA( (a1 or a3) ) P_AMM( (a2 or a4), a2, a4) rho_MM( a3, a4 )
        # F_MM( (a1, R1), (a2, R2) ) = 
        #    P_AMM( (a1,R1) or (a3,R3)), (a1,R1), (a3,R3) ) 
        #                 W_AA( ((a1,R1) or (a3, R3), ((a2,R2) or (a4,R4)) ) 
        #    P_AMM( ((a2,R2) or (a4,R4)), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )

        # Expand ors
        # 1. F_MM( (a1, R1), (a2, R2) ) = 
        #    P_AMM( (a1,R1), (a1,R1), (a3,R3) ) 
        #                 W_AA( (a1,R1), (a2, R2) ) ) 
        #    P_AMM( (a2,R2), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )

        # 2. F_MM( (a1, R1), (a2, R2) ) = 
        #    P_AMM( (a3,R3), (a1,R1), (a3,R3) ) 
        #                 W_AA( (a3,R3), (a2. R2) )
        #    P_AMM( (a2,R2), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )

        # 3. F_MM( (a1, R1), (a2, R2) ) = 
        #    P_AMM( (a1,R1), (a1,R1), (a3,R3) ) 
        #                 W_AA( (a1,R1), (a4, R4) ) 
        #    P_AMM( (a4,R4), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )

        # 4. F_MM( (a1, R1), (a2, R2) ) = 
        #    P_AMM( (a3,R3), (a1,R1), (a3,R3) ) 
        #                 W_AA( (a3,R3), (a4, R4) ) 
        #    P_AMM( (a4,R4), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )


        # Fix R1 = 0
        # 1. F_MM( (a1, 0), (a2, R2) ) = 
        #    P_AMM( (a1,0), (a1,0), (a3,R3) ) 
        #                 W_AA( (a1,0), (a2, R2) ) ) 
        #    P_AMM( (a2,R2), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )

        #    i)  Evaluate P_AMM( (a1,0), (a1,0), (a3,R3) ) for all overlapping (a3, R3)
        #    ii) Evaluate P_AMM( (a2,R2), (a2,R2), (a4,R4) ) for all overlapping (a4, R4)
        #    iii) Evaluate rho_MM for all pairs of (a3, R3), (a4, R3)
        #    iv) Contraction is now P_AMM( (a2,R2),(a2,R2), (a4,R4) ) * rho_MM( (a4, R4), (a3,R3) )
        #          yields PRHO_AMM( (a2,R2), (a2,R3), (a3, R3) )

        # 2. F_MM( (a1, 0), (a2, R2) ) = 
        #    P_AMM( (a3,R3), (a1,0), (a3,R3) ) 
        #                 W_AA( (a3,R3), (a2. R2) )
        #    P_AMM( (a2,R2), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )

        # 3. F_MM( (a1, 0), (a2, R2) ) = 
        #    P_AMM( (a1,0), (a1,0), (a3,R3) ) 
        #                 W_AA( (a1,0), (a4, R4) ) 
        #    P_AMM( (a4,R4), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )

        # 4. F_MM( (a1, 0), (a2, R2) ) = 
        #    P_AMM( (a3,R3), (a1,0), (a3,R3) ) 
        #                 W_AA( (a1,0), (a4, R4) ) 
        #    P_AMM( (a4,R4), (a2,R2), (a4,R4))
        #                rho_MM( (a3,R3), (a4, R4) )


        my_elements = []
        for R, sdisp_c in enumerate(self.sdisp_Rc):
            for a1 in self.my_atoms:
                for a2 in self.my_atoms:
                    my_elements.append( (a1, a2, R) )
                
        P_AMM_source.get(mu=a1)        

    def calculate_non_local(self):
        evv = 0.0
        ekin = -2*evv
        return evv, ekin

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        with self.timer('RI Local atomic corrections'):
            return paw_exx_correction(setup, D_sp, dH_sp, self.exx_fraction)

