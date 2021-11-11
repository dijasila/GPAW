"""

These classes are completely serial, and deal only with calculating certain 
blocks of the full matrices


Things to calculate:


     Local generalized-gaussian - generalized gaussian, will be be 
precalculated in the setup.py

     W_GG[a1,a2] = 
        '2 center integrals, asymptotic generalized-gaussian - generalized gaussian'
    
        *  will be calculated utilizing multipole expansions'

     W_GA[a1,a2] =
        '2 center integrals Generalized-gaussian - screened auxiliary, will be calculated using tci.'

     W_AA[a1,a2] =
        '2 center integrals Screened auxiliary - screened auxiliary, will 
         be calculated using tci.'

     I_AMM[a1,a2] = 
        '3 center integrals, where A and first M belong to atom a1, and second M belongs to atom a2.
         
Tasks:

     1. Create a list of all atom pairs with finite basis function overlap
        i) Obtain cutoffs for basis functions

        i) This information is in the AtomRegistry of tci

"""
import numpy as np
from gpaw.lcao.tci import get_cutoffs, split_setups_to_types, AtomPairRegistry

class MatrixElements:
    def __init__(self):
        pass

    def initialize(self, density, ham, wfs):
        setups = wfs.setups

        self.M_a = setups.M_a.copy()
        self.M_a.append(setups.nao)
        print('M_a', self.M_a)

        # I_a is an index for each atom identifying which setup type it has.
        # setup_for_atom_a = setups_I[I_a[a]]
        I_a, setups_I = split_setups_to_types(setups)

        # Build a list of basis function splines for each setup, and obtain
        # the maximum cut off for each setup type, phit_rcmax_I.
        phit_rcmax_I = get_cutoffs([s.phit_j for s in setups_I])

        # Obtain the maximum cutoff on per atom basis
        self.rcmax_a = [phit_rcmax_I[I] for I in I_a]

    def set_positions_and_cell(self, spos_ac, cell_cv, pbc_c):
        self.spos_ac = spos_ac
        self.cell_cv = cell_cv
        self.pbc_c = pbc_c

        # Build an atom pair registry for basis function overlap
        self.apr = AtomPairRegistry(self.rcmax_a, pbc_c, cell_cv, spos_ac)

        self.a1a2_p = [ (a1,a2) for a1,a2 in self.apr.get_atompairs() if a1<=a2 ]
        print(self.a1a2_p)
        print(self.rcmax_a)
        def log(name, quantity):
            print('%-70s %-5d' % (name, quantity))

        log('Number of atoms', len(self.rcmax_a))
        log('Atom pairs with overlapping basis functions', len(self.a1a2_p))
        self.a1a2_p = [ (a1,a2) for a1,a2 in self.apr.get_atompairs() if a1<=a2 ]

        #for a1, a2 in self.a1a2_p:
        #    for R_c in self.apr.get(a1,a2):
        #        print(a1,a2,np.linalg.norm(R_c), R_c)

        def a1a2a3a4_q():
            for p1, (a1,a2) in enumerate(self.a1a2_p):
                for a3, a4 in self.a1a2_p[p1:]:
                    yield a1,a2,a3,a4
 
        self.a1a2a3a4_q = a1a2a3a4_q

        # Even this takes minutes on large systems
        # log('Atom quadtuples with exchange of pairs', len([ aaaa for aaaa in self.a1a2a3a4_q() ]))

        log('Atom quadtuples with exchange of pairs', len(self.a1a2_p)*(len(self.a1a2_p)+1)/2)


    def set_parameters(self, parameters):
        self.parameters = parameters

    """
             /       ||               \
     I    =  | g (r) || φ (r') φ (r') |
      LMM    \  L    ||  M     M'     /,

      where double bar stands for Coulomb integral.

      a1 is always either a2 or a3.
      a2 and a3 are overlapping centers.
    """

    def evaluate_3ci_LMM(self, a1, a2, a3):
        
        R_c_and_offset_a = self.apr.get(a2, a3)
        if R_c_and_offset_a is None:
            return None
 
        print(a1,a2,a3)
        print(R_c_and_offset_a)
        raise NotImplementedError
        return 0.0
