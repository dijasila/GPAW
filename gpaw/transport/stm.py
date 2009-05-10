import numpy as np
from ase.units import Bohr

from gpaw.grid_descriptor import GridDescriptor
from gpaw.transformers import Transformer
from gpaw.lfc import NewLocalizedFunctionsCollection as LFC
import gpaw.mpi as mpi

class Transformer:
    def __init__(self,gd, nn):
        self.gd = gd
        self.nn=nn

    def apply(self,f_ig,Tf_iG):
        nn = self.nn
        gd = self.gd
        h_c= gd.h_c

        f_iG=f_ig[:, nn:-nn, nn:-nn, nn:-nn]
        
        Tf_ix=np.zeros_like(Tf_iG)
        Tf_iy=np.zeros_like(Tf_iG)
        Tf_iz=np.zeros_like(Tf_iG)
                        
        c_1 =[None,1,2,3,4]
        c_2 =[-4,-3,-2,-1,None]
        c_3 = [-1,16,-30,16,-1] #coefficients for a five point stencil
        for i,j,k in zip(c_1,c_2,c_3):
            Tf_ix[:,i:j,nn:-nn,nn:-nn]+=k*f_iG
            Tf_iy[:,nn:-nn,i:j,nn:-nn]+=k*f_iG
            Tf_iz[:,nn:-nn,nn:-nn,i:j]+=k*f_iG
        
        T=(Tf_ix/h_c[0]**2+Tf_iy/h_c[1]**2+Tf_iz/h_c[2]**2)/12.0
        
        Tf_iG[:,:,:,:]=T[:,:,:,:]


class LocalizedFunctions:
    def __init__(self, gd, f_iG, corner_c, index=None, vt_G=None):
        self.gd = gd
        #assert gd.is_orthogonal()
        assert not gd.is_non_orthogonal()
        self.size_c = np.array(f_iG.shape[1:4])
        self.f_iG = f_iG
        self.corner_c = corner_c
        self.index = index
        self.vt_G = vt_G
        
        #check boundary conditions and find translation vectors
        v1_c = -np.sign(corner_c[:2])
        v2_c = -np.sign(corner_c[:2]+self.size_c[:2]\
               -(gd.end_c[:2]-np.array([1,1])))
        trans_v = [np.array([0,0,0])]
        for i in range(2):
          if v1_c[i]==1:
            v = np.zeros(3,dtype=int)
            v[i]=1
            trans_v.append(v)
          if v2_c[i]==-1:
            v = np.zeros(3,dtype=int)
            v[i]=-1
            trans_v.append(v)
        if len(trans_v)==3:
          v = trans_v[1]+trans_v[2]
          trans_v.append(v)
        trans_v[:]*=(self.gd.N_c-np.array([1,1,1]))
        #List of corners of all periodic translations
        self.periodic_list = trans_v+corner_c
        
    def get_periodic_list(self):
        list = []
        for corner in self.periodic_list:
          list.append(LocalizedFunctions(self.gd,self.f_iG,
                                        corner_c=corner,
                                        index=self.index,
                                        vt_G=self.vt_G))
        return list


    def __len__(self):
        return len(self.f_iG)

    def apply_t(self):
        """Apply kinetic energy operator and return new object."""
        p = 2  # padding
        newsize_c = self.size_c + 2 * p
        gd = GridDescriptor(N_c=newsize_c + 1,
                            cell_cv=self.gd.h_c * (newsize_c + 1),
                            pbc_c=False,
                            comm=mpi.serial_comm)
        T = Transformer(gd, nn=p)
        f_ig = np.zeros((len(self.f_iG),) + tuple(newsize_c))
        f_ig[:, p:-p, p:-p, p:-p] = self.f_iG
        Tf_iG = np.empty_like(f_ig)
        T.apply(f_ig, Tf_iG)
        return LocalizedFunctions(self.gd, Tf_iG, self.corner_c - p,
                                  self.index)
        
    def overlap(self, other):
        start_c = np.maximum(self.corner_c, other.corner_c)
        stop_c = np.minimum(self.corner_c + self.size_c,
                            other.corner_c + other.size_c)
        if (start_c < stop_c).all():
            astart_c = start_c - self.corner_c
            astop_c = stop_c - self.corner_c
            a_iG = self.f_iG[:,
                astart_c[0]:astop_c[0],
                astart_c[1]:astop_c[1],
                astart_c[2]:astop_c[2]].reshape((len(self.f_iG), -1))
            bstart_c = start_c - other.corner_c
            bstop_c = stop_c - other.corner_c
            b_iG = other.f_iG[:,
                bstart_c[0]:bstop_c[0],
                bstart_c[1]:bstop_c[1],
                bstart_c[2]:bstop_c[2]].reshape((len(other.f_iG), -1))
            if self.vt_G is not None:
                a_iG *= self.vt_G[start_c[0]:stop_c[0],
                                  start_c[1]:stop_c[1],
                                  start_c[2]:stop_c[2]].reshape((-1,))
            return self.gd.dv * np.inner(a_iG, b_iG)
        else:
            return None

    def __or__(self, other):
        if isinstance(other, LocalizedFunctions):
            return self.overlap(other)

        # other is a potential:
        vt_G = other
        return LocalizedFunctions(self.gd, self.f_iG, self.corner_c,
                                  self.index, vt_G)

class WannierFunction(LocalizedFunctions):
    def __init__(self, gd, wanf_G, corner_c, index=None):
        LocalizedFunctions.__init__(self, gd, wanf_G[np.newaxis, :, :, :],
                                    corner_c, index)

class AtomCenteredFunctions(LocalizedFunctions):
    def __init__(self, gd, spline_j, spos_c, index=None):
        rcut = max([spline.get_cutoff() for spline in spline_j])
        corner_c = np.ceil(spos_c * gd.N_c - rcut / gd.h_c).astype(int)
        size_c = np.ceil(spos_c * gd.N_c + rcut / gd.h_c).astype(int) - corner_c
        smallgd = GridDescriptor(N_c=size_c + 1,
                                 cell_cv=gd.h_c * (size_c + 1),
                                 pbc_c=False,
                                 comm=mpi.serial_comm)
        lfc = LFC(smallgd, [spline_j])
        lfc.set_positions((spos_c[np.newaxis, :] * gd.N_c - corner_c + 1) /
                          smallgd.N_c)
        ni = lfc.Mmax
        f_iG = smallgd.zeros(ni)
        lfc.add(f_iG, {0: np.eye(ni)})
        LocalizedFunctions.__init__(self, gd, f_iG, corner_c, index)

class STM:
    def __init__(self, tip, surface):
        self.tip = tip
        self.srf = surface

        tgd = tip.gd
        sgd = surface.gd

        #assert tgd.h_cv == sgd.h_cv
        assert not (tgd.h_c - sgd.h_c).any()

    def initialize(self, tip_atom_index, dmin=2.0):
        dmin /= Bohr
        tip_pos_av = self.tip.atoms.get_positions() / Bohr
        srf_pos_av = self.srf.atoms.get_positions() / Bohr
        tip_zmin = tip_pos_av[tip_atom_index, 2]
        srf_zmax = srf_pos_av[:, 2].max()

        offset_c = (tip_pos_av[tip_atom_index] / self.tip.gd.h_c).astype(int)

        tip_zmin_a = np.empty(len(tip_pos_av))
        
        for a, setup in enumerate(self.tip.wfs.setups):
            rcutmax = max([phit.get_cutoff() for phit in setup.phit_j])
            tip_zmin_a[a] = tip_pos_av[a, 2] - rcutmax - tip_zmin

        srf_zmax_a = np.empty(len(srf_pos_av))
        for a, setup in enumerate(self.srf.wfs.setups):
            rcutmax = max([phit.get_cutoff() for phit in setup.phit_j])
            srf_zmax_a[a] = srf_pos_av[a, 2] + rcutmax - srf_zmax

        tip_indices = np.where(tip_zmin_a < srf_zmax_a.max() - dmin)[0]  
        srf_indices = np.where(srf_zmax_a > tip_zmin_a.min() + dmin)[0]  
        print 'Tip atoms:', tip_indices
        print 'Surface atoms:', srf_indices
        
        self.tip_functions = []
        i = 0
        for a in tip_indices:
            setup = self.tip.wfs.setups[a]
            spos_c = tip_pos_av[a] / self.tip.gd.cell_c
            for phit in setup.phit_j:
                f = AtomCenteredFunctions(self.tip.gd, [phit], spos_c, i)
                self.tip_functions.append(f)
                i += len(f.f_iG)
        self.ni = i

        # Apply kinetic energy:
        self.tip_functions_kin = []
        #for f in self.tip_functions:
        #    self.tip_functions_kin.append(f.apply_t())

        self.srf_functions = []
        j = 0
        for a in srf_indices:
            setup = self.srf.wfs.setups[a]
            spos_c = srf_pos_av[a] / self.srf.gd.cell_c
            for phit in setup.phit_j:
                f = AtomCenteredFunctions(self.srf.gd, [phit], spos_c, j)
                self.srf_functions.append(f)

                # Boundary conditions!!!?
                #if f outside box:
                #    self.srf_functions.append(f.translate())
                    
                
                j += len(f.f_iG)
        self.nj = j
        
        #self.gd = GridDescriptor()

        #S_ij = ...

    def get_s(self, ):
        S_ij = np.zeros((self.ni, self.nj))
        for s in self.srf_functions:
            j1 = s.index
            j2 = j1 + len(s)
            for t in self.tip_functions:
                i1 = t.index
                i2 = i1 + len(t)
                overlap = (t | vt_G | s) # + kinetic energy XXX
                if overlap is not None:
                    S_ij[i1:i2, j1:j2] += overlap
  
        return S_ij
