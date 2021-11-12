from gpaw.auxlcao.algorithm import RIAlgorithm

"""
Implements reference implementation which evaluates all matrix elements inefficiently on the grid.

RIVFullBasisDebug calculates all 2 and 3-center matrix elements.
"""

from gpaw.lfc import LFC
import numpy as np
from gpaw.utilities.tools import tri2full
from gpaw.utilities import unpack2, pack2, packed_index


cutoff = 10

from timeit import default_timer

class MyTimer(object):
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose
        self.timer = default_timer
        
    def __enter__(self):
        self.start = self.timer()
        return self
        
    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        if self.verbose:
            print('%s elapsed time: %f ms' % (self.name, self.elapsed))



class RIVComparison:
    def __init__(self, algs):
        self.algs = algs

    def initialize(self, density, ham, wfs):
        for alg in algs:
            alg.initialize(density, ham, wfs)

    def set_positions(self, spos_ac):
        for alg in algs:
            alg.initialize(density, ham, wfs)
        self.compare_matrix_elements()
        xxx

    def compare_matrix_elements(self):
        for alg in algs[1:]:
            self.compare_matrix('I_AMM', alg.I_AMM, algs[0].I_AMM)
            self.compare_matrix('W_AA', alg.W_AA, algs[0].W_AA)


    def nlxc(self, H_MM, dH_asp, wfs, kpt):
        Hx_MM = []
        dH_xasp = []
        for x, alg in enumerate(algs):
            if x < len(algs):
                H1_MM = H_MM.copy()
                dH1_asp = dH_asp.copy()
            else:
                H1_MM, dH1_asp = H_MM, dH_asp 
            nlxc(H1_MM, dH_asp, wfs, kpt)
            Hx_MM.append(H1_MM)
            dH_xasp.append(dH_asp)


class RIVFullBasisDebug(RIAlgorithm):
    def initialize(self, density, ham, wfs):
        print("RIVFulLBasisDebug.initialize")

        self.ham = ham
        self.density = density
        self.wfs = wfs
        self.ecc = sum(setup.ExxC for setup in wfs.setups) * self.exx_fraction
        #self.symbols_a = atoms.get_chemical_symbols()
        assert wfs.world.size == wfs.gd.comm.size

        # Copy pasted from setup.py
        def H(rgd, n_g, l):
            v_g = rgd.poisson(n_g, l)
            v_g[1:] /= rgd.r_g[1:]
            v_g[0] = v_g[1]
            return v_g

        if 1:
            # For each atom
            auxt_aj = []
            for a, sphere in enumerate(wfs.basis_functions.sphere_a):
                auxt_j = []
                wauxt_j = []
                rgd = wfs.setups[a].rgd


                #ghat_l = []
                wghat_l = []
                g_lg = []
                for l in range(3):
                    spline = wfs.setups[a].ghat_l[l]
                    f_g = rgd.zeros()
                    for g, r in enumerate(rgd.r_g):
                        f_g[g] = spline(r) * r**l
                    g_lg.append(f_g)

                    # XXX Debug code no longer works, because ghat's are not aded to auxt_j
                    #ghat_l.append(wfs.setups[a].ghat_l[l])
                    #wghat_l.append(rgd.spline(H(rgd, f_g, l), cutoff, l, 100)) # XXX
                    
                    print('not adding compensation charges to aux basis')
                    continue
                    auxt_j.append(wfs.setups[a].ghat_l[l])
                    wauxt_j.append(rgd.spline(H(rgd, f_g, l), cutoff, l,500)) # XXX


                # This code naiively just creates an auxiliary basis function for each basis function pair
                # of the atom:
                #
                # \tilde \Theta_A(r) = \tilde \phi_i(r) \tilde \phi_j(r),
                #
                # Create auxiliary wave function space naiively just with
                for j1, spline1 in enumerate(sphere.spline_j):
                    for j2, spline2 in enumerate(sphere.spline_j):
                        if j1 > j2:
                            continue
                        f_g = rgd.zeros()
                        l1 = spline1.get_angular_momentum_number()
                        l2 = spline2.get_angular_momentum_number()
                        for g, r in enumerate(rgd.r_g):
                            f_g[g] = spline1(r) * spline2(r) * r ** (l1+l2)
                        for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                            if l > 2:
                                continue

                            if 0: # Not removing multipole
                                M = rgd.integrate(f_g * rgd.r_g**l) / (4*np.pi)
                                #print('Multipole before l=%d' % l, M)
                                f_g -= M*g_lg[l]

                            #print('Multipole after', rgd.integrate(f_g * rgd.r_g**l) / (4*np.pi))

                            #print('Not adding auxiliary basis functions')
                            #continue

                            auxt_j.append(rgd.spline(f_g, cutoff, l, 100))
                            wauxt_j.append(rgd.spline(H(rgd, f_g, l), cutoff, l, 100))
               
                auxt_aj.append(auxt_j)

                wfs.setups[a].auxt_j = auxt_j
                wfs.setups[a].wauxt_j = wauxt_j
                #wfs.setups[a].ghat_l = ghat_l
                wfs.setups[a].wghat_l = wghat_l

                x = 0
                wauxtphit_x = []
                for wauxt in wauxt_j:
                    for j1, spline1 in enumerate(sphere.spline_j):
                        f_g = rgd.zeros()
                        l1 = wauxt.get_angular_momentum_number()
                        l2 = spline1.get_angular_momentum_number()
                        for g, r in enumerate(rgd.r_g):
                            f_g[g] = spline1(r) * wauxt(r) * r ** (l1+l2)
                        for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                            wauxtphit_x.append(rgd.spline(f_g, cutoff, l))
                wfs.setups[a].wauxtphit_x = wauxtphit_x

            self.aux_lfc = LFC(density.finegd, auxt_aj)
            self.aux_lfc_coarse = LFC(density.gd, auxt_aj)

        self.timer = ham.timer

    def set_positions(self, spos_ac):
       if 1:
            #np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

            finegd = self.density.finegd
            self.aux_lfc.set_positions(spos_ac)
            self.aux_lfc_coarse.set_positions(spos_ac)

            # Count the number of auxiliary functions
            Atot = 0
            for a, setup in enumerate(self.wfs.setups):
                for j, auxt in enumerate(setup.auxt_j):
                    for m in range(2*auxt.l+1):
                        Atot += 1

            nao = self.wfs.setups.nao

            self.W_AA = np.zeros( (Atot, Atot) )
            self.I_AMM = np.zeros( (Atot, nao, nao) )
            self.G_AaL = []

            Aind = 0
            for a, setup in enumerate(self.wfs.setups):
                #print('Atom %d' % a)
                M = 0
                for j, aux in enumerate(setup.auxt_j):
                    print('   aux %d' % j)
                    for m in range(2*aux.l+1):
                        print('       m=%d' % (m-aux.l))
                        Q_aM = self.aux_lfc.dict(zero=True)
                        Q_aM[a][M] = 1.0

                        auxt_g = self.aux_lfc.gd.zeros()
                        wauxt_g = self.aux_lfc.gd.zeros()
                        self.aux_lfc.add(auxt_g, Q_aM)
                        self.ham.poisson.solve(wauxt_g, auxt_g, charge=None)

                        G_aL = self.density.ghat.dict(zero=True)
                        self.density.ghat.integrate(wauxt_g, G_aL)
                        self.G_AaL.append(G_aL)

                        # Calculate W_AA
                        W_aM = self.aux_lfc.dict(zero=True)
                        self.aux_lfc.integrate(wauxt_g, W_aM)

                        Astart = 0
                        for a2 in range(len(self.wfs.setups)):
                            W_M = W_aM[a2]
                            self.W_AA[Aind, Astart:Astart+len(W_M)] = W_M
                            Astart += len(W_M)

                        # Calculate W_Imm
                        wauxt_G = self.ham.gd.zeros()
                        self.ham.restrict(wauxt_g, wauxt_G)
                        V_MM = self.wfs.basis_functions.calculate_potential_matrices(wauxt_G)[0]
                        tri2full(V_MM)
                        self.I_AMM[Aind] = V_MM

                        M += 1
                        Aind += 1
            self.W_AA = (self.W_AA + self.W_AA.T)/2

            with open('RIVFullBasisDebug-W_AA.npy', 'wb') as f:
                np.save(f, self.W_AA)

            # I_AMM += \sum_L \sum_a \sum_ii Delta^a_Lii P_aMi P_aMi G_AaL

            with open('RIVFullBasisDebug-I_AMM.npy', 'wb') as f:
                np.save(f, self.I_AMM)

            for A in range(Atot):
                for a, setup in enumerate(self.wfs.setups):
                    G_L = self.G_AaL[A][a]
                    X_ii = np.dot(setup.Delta_iiL, G_L)
                    P_Mi = self.wfs.atomic_correction.P_aqMi[a][0]
                    self.I_AMM[A] += P_Mi @ X_ii @ P_Mi.T

            self.iW_AA = np.linalg.inv(self.W_AA)

    def cube_debug(self, atoms):
        from gpaw.auxlcao.tools import orbital_product_to_cube
        from ase.io.cube import write_cube

        C_AMM = np.einsum('AB,Bkl',
                           self.iW_AA,
                           self.I_AMM, optimize=True)

        nao = self.wfs.setups.nao
        for M1 in range(nao):
            for M2 in range(nao):
                orbital_product_to_cube('ref_%03d_%03d.cube' % (M1, M2), atoms, self.wfs, M1, M2)

                rho_MM = np.zeros( (nao, nao) )
                A = 0
                Q_aM = self.aux_lfc_coarse.dict(zero=True)
                for a, setup in enumerate(self.wfs.setups):
                    Aloc = 0
                    for j, aux in enumerate(setup.auxt_j):
                        for m in range(2*aux.l+1):
                            Q_aM[a][Aloc] = C_AMM[A, M1, M2]
                            A += 1
                            Aloc += 1
                fit_G = self.density.gd.zeros()
                self.aux_lfc_coarse.add(fit_G, Q_aM)

                write_cube(open('fit_%03d_%03d.cube' % (M1, M2),'w'), atoms, data=fit_G, comment='')

        with open('RIVFullBasisDebug-C_AMM.npy', 'wb') as f:
            np.save(f, C_AMM)
        xxx
        

    def nlxc(self, H_MM, dH_asp, wfs, kpt):
        self.evc = 0.0
        self.evv = 0.0
        self.ekin = 0.0

        rho_MM = wfs.ksl.calculate_density_matrix(kpt.f_n, kpt.C_nM)
        rho_MM[:] = 0.0
        rho_MM[0,0] = 1.0
        C_AMM = np.einsum('AB,Bkl,jl',
                          self.iW_AA,
                          self.I_AMM,
                          rho_MM, optimize=True)

        with open('RIVFullBasisDebug-C_AMM.npy', 'wb') as f:
            np.save(f, C_AMM)

        F_MM = -0.5 * np.einsum('Aij,AB,Bkl,jl',
                                self.I_AMM,
                                self.iW_AA,
                                self.I_AMM,
                                rho_MM, optimize=True)

        with open('RIVFullBasisDebug-F_MM.npy', 'wb') as f:
            np.save(f, F_MM)
            xxx

        H_MM += self.exx_fraction * F_MM
        self.evv = 0.5 * self.exx_fraction * np.einsum('ij,ij', F_MM, rho_MM)

        for a in dH_asp.keys():
            #print(a)
            D_ii = unpack2(self.density.D_asp[a][0]) / 2 # Check 1 or 2
            # Copy-pasted from hybrids/pw.py
            ni = len(D_ii)
            V_ii = np.empty((ni, ni))
            for i1 in range(ni):
                for i2 in range(ni):
                    V = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            V += self.density.setups[a].M_pp[p13, p24] * D_ii[i3, i4]
                    V_ii[i1, i2] = +V
            V_p = pack2(V_ii)
            dH_asp[a][0] += (-V_p - self.density.setups[a].X_p) * self.exx_fraction

            #print("Atomic Ex correction", np.dot(V_p, self.density.D_asp[a][0]) / 2)
            #print("Atomic Ex correction", np.trace(V_ii @ D_ii))
            self.evv -= self.exx_fraction * np.dot(V_p, self.density.D_asp[a][0]) / 2
            self.evc -= self.exx_fraction * np.dot(self.density.D_asp[a][0], self.density.setups[a].X_p)

        self.ekin = -2*self.evv - self.evc
        return self.evv, self.evc, self.ekin


class RIVRestrictedBasisDebug(RIAlgorithm):
    def initialize(self, density, ham, wfs):
        print("RIVRestrictedBasisDebug.initialize")

        self.ham = ham
        self.density = density
        self.wfs = wfs
        self.ecc = sum(setup.ExxC for setup in wfs.setups) * self.exx_fraction
        #self.symbols_a = atoms.get_chemical_symbols()
        assert wfs.world.size == wfs.gd.comm.size

        # Copy pasted from setup.py
        def H(rgd, n_g, l):
            v_g = rgd.poisson(n_g, l)
            v_g[1:] /= rgd.r_g[1:]
            v_g[0] = v_g[1]
            return v_g

        if 1:
            # For each atom
            auxt_aj = []
            for a, sphere in enumerate(wfs.basis_functions.sphere_a):
                auxt_j = []
                wauxt_j = []
                rgd = wfs.setups[a].rgd


                #ghat_l = []
                wghat_l = []
                g_lg = []
                for l in range(3):
                    spline = wfs.setups[a].ghat_l[l]
                    f_g = rgd.zeros()
                    for g, r in enumerate(rgd.r_g):
                        f_g[g] = spline(r) * r**l
                    g_lg.append(f_g)

                    auxt_j.append(wfs.setups[a].ghat_l[l])
                    wauxt_j.append(rgd.spline(H(rgd, f_g, l), cutoff, l,500)) # XXX


                # This code naiively just creates an auxiliary basis function for each basis function pair
                # of the atom:
                #
                # \tilde \Theta_A(r) = \tilde \phi_i(r) \tilde \phi_j(r),
                #
                # Create auxiliary wave function space naiively just with
                for j1, spline1 in enumerate(sphere.spline_j):
                    for j2, spline2 in enumerate(sphere.spline_j):
                        if j1 > j2:
                            continue
                        f_g = rgd.zeros()
                        l1 = spline1.get_angular_momentum_number()
                        l2 = spline2.get_angular_momentum_number()
                        for g, r in enumerate(rgd.r_g):
                            f_g[g] = spline1(r) * spline2(r) * r ** (l1+l2)
                        for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                            if l > 2:
                                continue
                            M = rgd.integrate(f_g * rgd.r_g**l) / (4*np.pi)
                            #print('Multipole before l=%d' % l, M)
                            f_g -= M*g_lg[l]
                            #print('Multipole after', rgd.integrate(f_g * rgd.r_g**l) / (4*np.pi))
                            print('Not adding aux orbital products')
                            #auxt_j.append(rgd.spline(f_g, cutoff, l, 100))
                            #wauxt_j.append(rgd.spline(H(rgd, f_g, l), cutoff, l, 100))
               
                auxt_aj.append(auxt_j)

                wfs.setups[a].auxt_j = auxt_j
                wfs.setups[a].wauxt_j = wauxt_j
                #wfs.setups[a].ghat_l = ghat_l
                wfs.setups[a].wghat_l = wghat_l

                x = 0
                wauxtphit_x = []
                for wauxt in wauxt_j:
                    for j1, spline1 in enumerate(sphere.spline_j):
                        f_g = rgd.zeros()
                        l1 = wauxt.get_angular_momentum_number()
                        l2 = spline1.get_angular_momentum_number()
                        for g, r in enumerate(rgd.r_g):
                            f_g[g] = spline1(r) * wauxt(r) * r ** (l1+l2)
                        for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                            wauxtphit_x.append(rgd.spline(f_g, cutoff, l))
                wfs.setups[a].wauxtphit_x = wauxtphit_x

            self.aux_lfc = LFC(density.finegd, auxt_aj)

        self.timer = ham.timer

    def set_positions(self, spos_ac):
       # TODO: Faster loop, do not loop non-overlapping atoms
       def loop_pairs():
           for a1, setup in enumerate(self.wfs.setups):
               for a2, setup in enumerate(self.wfs.setups):
                   yield a1,a2

       self.loop_pairs = loop_pairs

       # For each pair of sites, we need to create projection matrix
       # P_AMM = Wloc_AA^(-1) I_AMM

       if 1:
            #np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

            finegd = self.density.finegd
            self.aux_lfc.set_positions(spos_ac)

            self.Astart_a = []
            # Count the number of auxiliary functions
            Atot = 0
            for a, setup in enumerate(self.wfs.setups):
                self.Astart_a.append(Atot)
                for j, auxt in enumerate(setup.auxt_j):
                    Atot += 2*auxt.l+1
            self.Astart_a.append(Atot)

            self.Mstart_a = []
            # Count the number of basis functions
            Mtot = 0
            for a, setup in enumerate(self.wfs.setups):
                self.Mstart_a.append(Mtot)
                for j, phit in enumerate(setup.phit_j):
                    Mtot += 2*phit.l+1
            self.Mstart_a.append(Mtot)

            nao = self.wfs.setups.nao

            self.W_AA = np.zeros( (Atot, Atot) )
            self.I_AMM = np.zeros( (Atot, nao, nao) )
            self.G_AaL = []

            Aind = 0
            for a, setup in enumerate(self.wfs.setups):
                #print('Atom %d' % a)
                M = 0
                for j, aux in enumerate(setup.auxt_j):
                    print('   aux %d' % j)
                    for m in range(2*aux.l+1):
                        print('       m=%d' % (m-aux.l))
                        Q_aM = self.aux_lfc.dict(zero=True)
                        Q_aM[a][M] = 1.0

                        auxt_g = self.aux_lfc.gd.zeros()
                        wauxt_g = self.aux_lfc.gd.zeros()
                        self.aux_lfc.add(auxt_g, Q_aM)
                        self.ham.poisson.solve(wauxt_g, auxt_g, charge=None)

                        G_aL = self.density.ghat.dict(zero=True)
                        self.density.ghat.integrate(wauxt_g, G_aL)
                        self.G_AaL.append(G_aL)

                        # Calculate W_AA
                        W_aM = self.aux_lfc.dict(zero=True)
                        self.aux_lfc.integrate(wauxt_g, W_aM)

                        Astart = 0
                        for a2 in range(len(self.wfs.setups)):
                            W_M = W_aM[a2]
                            self.W_AA[Aind, Astart:Astart+len(W_M)] = W_M
                            Astart += len(W_M)

                        # Calculate W_Imm
                        wauxt_G = self.ham.gd.zeros()
                        self.ham.restrict(wauxt_g, wauxt_G)
                        V_MM = self.wfs.basis_functions.calculate_potential_matrices(wauxt_G)[0]
                        tri2full(V_MM)
                        self.I_AMM[Aind] = V_MM

                        M += 1
                        Aind += 1

            self.W_AA = (self.W_AA + self.W_AA.T)/2
            self.iW_AA = np.linalg.pinv(self.W_AA, hermitian=True, rcond=1e-14)
            self.iW_AA = (self.iW_AA + self.iW_AA.T)/2

            print(np.linalg.eig(self.W_AA),"eigs of W_AA")
            print(self.iW_AA,"iW_AA")
            print(self.W_AA,"W_AA")
            print(self.W_AA - self.W_AA.T,"W_AA - W_AA.T")

            print(self.iW_AA-self.iW_AA.T,"asym iW_AA")
            # I_AMM += \sum_L \sum_a \sum_ii Delta^a_Lii P_aMi P_aMi G_AaL

            """
            for A in range(Atot):
                for a, setup in enumerate(self.wfs.setups):
                    G_L = self.G_AaL[A][a]
                    X_ii = np.dot(setup.Delta_iiL, G_L)
                    #print(X_ii,'X_ii')
                    P_Mi = self.wfs.atomic_correction.P_aqMi[a][0]
                    self.I_AMM[A] += P_Mi @ X_ii @ P_Mi.T
                self.I_AMM[A] = (self.I_AMM[A] + self.I_AMM[A].T)/2
           
            """

    def nlxc(self, H_MM, dH_asp, wfs, kpt):
        self.evc = 0.0
        self.evv = 0.0
        self.ekin = 0.0

        # XXX Remove this recalculation, problem is that density matrix is not really stored anywhere,
        # it only exists in the create density loop
        rho_MM = wfs.ksl.calculate_density_matrix(kpt.f_n, kpt.C_nM)

        assert(np.linalg.norm(rho_MM-rho_MM.T)<1e-12)

        M_a = self.Mstart_a
        A_a = self.Astart_a

        """             __     __
              a1 a2     \      \     a1 a3 a2 a4  a3 a4
             P       =  /_     /_   K            P
              i  j      a3,a4   kl   i  k  j  l   k  l

                       __     __   a1 a3 (a1,a3)          (a2,a4)   a2 a4
                     = \      \           -1             -1                 a3 a4
                       /_     /_  I      W        W      W         I       P
                       a3,a4  kl   Aik    AA       AA      AA       A'jl    k  l


             I_AMM((a1,a3), a1,a3)  inv(W_AA((a1,a3),(a1,a3)) W_AA( (a1,a3),(a2,a4) ) inv(W_AA((a2,a4),(a2,a4)) I_AMM(a2,a4) P_MM(a3, a4)

             1) Generate all overlapping atomic pairs, represent equal pairs as a single tuple
              
             2) For each pair, generate
                I_AMM = I_AMM_p[pair], where

        """       
        
        


        """
        with MyTimer('Atomwise loop'):
            F_MM = np.zeros_like(rho_MM)
            for a3, setup in enumerate(self.wfs.setups):       # k
                for a4, setup in enumerate(self.wfs.setups):   # l
                    rho_a3a4_MM = rho_MM[M_a[a3]:M_a[a3+1],M_a[a4]:M_a[a4+1]]
                    for a1, setup in enumerate(self.wfs.setups):               # i
                        I_a1a3_AMM = self.I_AMM[:, M_a[a1]:M_a[a1+1], M_a[a3]:M_a[a3+1]]
                        for a2, setup in enumerate(self.wfs.setups):           # j
                            I_a2a4_AMM = self.I_AMM[:, Mstart_a[a2]:Mstart_a[a2+1], Mstart_a[a4]:Mstart_a[a4+1]]
                            iW_AA = self.iW_AA
                            F_MM[Mstart_a[a1]:Mstart_a[a1+1],Mstart_a[a2]:Mstart_a[a2+1]] += \
                                 -0.5 * np.einsum('Aik,AB,Bjl,kl',
                                                  I_a1a3_AMM,
                                                  iW_AA,
                                                  I_a2a4_AMM,
                                                  rho_a3a4_MM, optimize=True)
        """

        def get_rho_MM(a,ap):
            return rho_MM[M_a[a]:M_a[a+1],M_a[ap]:M_a[ap+1]]

        def get_I_AMM(a,ap,app):
            return self.I_AMM[A_a[a]:A_a[a+1],M_a[ap]:M_a[ap+1], M_a[app]:M_a[app+1]]

        def get_W_AA(a,ap):
            return self.W_AA[A_a[a]:A_a[a+1], A_a[ap]:A_a[ap+1]]

        def safe_inv(W_AA):
            print(np.linalg.eigvalsh(W_AA), 'Eigenvalues of W_AA before inverse')
            #iW_AA = np.linalg.pinv(W_AA, hermitian=True, rcond=1e-6)
            iW_AA = np.linalg.inv(W_AA)
            print('Non hermitian part of inverse', iW_AA-iW_AA.T)
            iW_AA = (iW_AA + iW_AA.T)/2
            return iW_AA

        F_MM = np.zeros_like(rho_MM)
        # a1 = a2 = a3 = a4
        with MyTimer('a1 = a2 = a3 = a4'):
            for a, setup in enumerate(self.wfs.setups): # k
                rho_aa_MM = get_rho_MM(a, a)
                I_AMM = get_I_AMM(a,a,a)
                iW_AA = safe_inv(get_W_AA(a,a))
                F_MM[M_a[a]:M_a[a+1],M_a[a]:M_a[a+1]] = \
                       -0.5 * np.einsum('Aij,AB,Bkl,jl',
                                         I_AMM,
                                         iW_AA,
                                         I_AMM,
                                         rho_aa_MM, optimize=True)

       

        # One P matrix for all overlapping pairs


        P1_aMMM = {}
        with MyTimer('Single center projection operators'):
            # Calculate single center projection operators
            for a, setup in enumerate(self.wfs.setups):
                P1_aAMM[a] = np.einsum('AB,Bij', safe_inv(get_W_AA(a,a)), get_I_AMM(a,a,a))

        with MyTimer('a2=a4 , a1=a3, a1 != a2'):
            C_aMM = {}
            for a1, setup in enumerate(self.wfs.setups):
                a3 = a1
                for a2, setup in enumerate(self.wfs.setups):
                    a4 = a2
                    rho_a3a4_MM = get_rho_MM(a3, a4) # a3 != a4
                    W_AA = get_W_AA(a1,a2)
                    F_MM[M_a[a1]:M_a[a1+1],M_a[a2]:M_a[a2+1]] = -0.5 * \
                        np.einsum('Aij,AB,Bkl,jl', P1_aAMM[a1], W_AA, P1_aAMM[a4], rho_a3a4_MM, optimize=True)

        """
        with MyTimer('Atomwise loop'):
            F_MM = np.zeros_like(rho_MM)
            for a3, setup in enumerate(self.wfs.setups):                      # k
                for a4, setup in enumerate(self.wfs.setups):                  # l
                    rho_a3a4_MM = get_rho_MM(a3, a4)
                    for a2, setup in enumerate(self.wfs.setups):              # j
                        I_a2a2a4_AMM = get_I_AMM(a2, a2, a4)
                        # 1st contraction: I_a2a4_AMM rho_a3a4_MM
                        Irho_a2a2a3_AMM = np.einsum('Ajl,kl', I_a2a2a4_AMM, rho_a3a4_MM)

                        if a2 == a4:
                            iW_a2a2_AA = safe_inv(get_W_AA(a2,a2))
                            #iWIrho_a2a2a3_AMM = np.dot(iW_a2a2_AA, Irho_a2a2a3_AMM)
                            iWIrho_a2a2a3_AMM = np.einsum('AB,Bij',iW_a2a2_AA, Irho_a2a2a3_AMM)

                            for a1, setup in enumerate(self.wfs.setups):               # i
                                if a1 == a3:
                                    W_a1a2_AA = get_W_AA(a1, a2)
                                    iWW_a1a2_AA = np.dot(safe_inv(get_W_AA(a1,a1)), W_a1a2_AA)
                                    #WiWIrho_a1a2a3_AMM = np.dot(iWW_a1a2_AA, iWIrho_a2a2a3_AMM)
                                    WiWIrho_a1a2a3_AMM = np.einsum('AB,Bij', iWW_a1a2_AA, iWIrho_a2a2a3_AMM)
                                    I_a1a1a3_AMM = get_I_AMM(a1, a1, a3)
                                    F_MM[M_a[a1]:M_a[a1+1],M_a[a2]:M_a[a2+1]] += \
                                       -0.5 * np.einsum('Aik,Ajk', I_a1a1a3_AMM, WiWIrho_a1a2a3_AMM)
                        else:
                            raise NotImplementedError
        """
        with MyTimer('Full einsum loop'):
            F2_MM = np.zeros_like(rho_MM)
            F2_MM = -0.5 * np.einsum('Aij,AB,Bkl,jl',
                                    self.I_AMM,
                                    self.iW_AA,
                                    self.I_AMM,
                                    rho_MM, optimize=True)
        print('Fock matrix error', np.linalg.norm(F_MM-F2_MM))
        print('F_MM',F_MM)
        print(self.iW_AA,"iW_AA")
        print(self.iW_AA-self.iW_AA.T,"iW_AA-iW_AA.T")
         
        print('F2_MM',F2_MM)
        print('F_MM-F2',F_MM-F2_MM)
        print('F_MM / F2',F_MM / F2_MM)
        print('F2_MM-F2_MM.T',F2_MM-F2_MM.T)
        print('F_MM-F_MM.T',F_MM-F_MM.T)
        H_MM += self.exx_fraction * F_MM

        self.evv = 0.5 * self.exx_fraction * np.einsum('ij,ij', F_MM, rho_MM)

        for a in dH_asp.keys():
            #print(a)
            D_ii = unpack2(self.density.D_asp[a][0]) / 2 # Check 1 or 2
            # Copy-pasted from hybrids/pw.py
            ni = len(D_ii)
            V_ii = np.empty((ni, ni))
            for i1 in range(ni):
                for i2 in range(ni):
                    V = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            V += self.density.setups[a].M_pp[p13, p24] * D_ii[i3, i4]
                    V_ii[i1, i2] = +V
            V_p = pack2(V_ii)
            dH_asp[a][0] += (-V_p - self.density.setups[a].X_p) * self.exx_fraction

            #print("Atomic Ex correction", np.dot(V_p, self.density.D_asp[a][0]) / 2)
            #print("Atomic Ex correction", np.trace(V_ii @ D_ii))
            self.evv -= self.exx_fraction * np.dot(V_p, self.density.D_asp[a][0]) / 2
            self.evc -= self.exx_fraction * np.dot(self.density.D_asp[a][0], self.density.setups[a].X_p)

        self.ekin = -2*self.evv -self.evc
        return self.evv, self.evc, self.ekin
