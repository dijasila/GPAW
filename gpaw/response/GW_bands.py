import numpy as np
from fractions import gcd
import pickle
import sys

from ase.dft.kpoints import ibz_points, get_bandpath
from ase.units import Bohr,Hartree

from gpaw import GPAW, PW, FermiDirac
from gpaw.spinorbit import get_spinorbit_eigenvalues
from gpaw.response.cell import get_primitive_cell
from gpaw.kpt_descriptor import to1bz

from scipy.interpolate import InterpolatedUnivariateSpline

class GW_bands:
    """This class defines the GW_bands properties"""

    def __init__(self,
                 calc=None,
                 gw_file=None,
                 kpoints=None): 

            self.calc = GPAW(calc)
            if gw_file is not None:            
                self.gw_file = pickle.load(open(gw_file))
            self.kpoints = kpoints

            self.gd = self.calc.wfs.gd.new_descriptor()
            self.kd = self.calc.wfs.kd
            
            rpad = np.ones(3, int)                                                                            
            self.acell_cv = self.gd.cell_cv
            self.acell_cv, self.bcell_cv, self.vol, self.BZvol = get_primitive_cell(self.acell_cv,rpad=rpad)
           

    # Finding the k-points along the bandpath
    #------------------------------------------------------------------------------
    def find_k_along_path(self,plot_BZ=True):
        kd = self.kd
        acell_cv = self.acell_cv
        bcell_cv = self.bcell_cv
        kpoints = self.kpoints

        if plot_BZ is True:
            #--------------------------------------------
            # Plotting the points in the Brillouin Zone
            #--------------------------------------------
            kp_1bz = to1bz(kd.bzk_kc,acell_cv)
            
            bzk_kcv=np.dot(kd.bzk_kc,bcell_cv)
            kp_1bz_v=np.dot(kp_1bz,bcell_cv)
            
            import matplotlib.pyplot as plt
            plt.plot(bzk_kcv[:,0],bzk_kcv[:,1],'xg')
            plt.plot(kp_1bz_v[:,0],kp_1bz_v[:,1],'ob')
            for ik in range(1,len(kpoints)):    
                kpoint1_v = np.dot(kpoints[ik],bcell_cv)
                kpoint2_v = np.dot(kpoints[ik-1],bcell_cv)
                plt.plot([kpoint1_v[0],kpoint2_v[0]],[kpoint1_v[1],kpoint2_v[1]],'--vr')


        #--------------------------------------------
        # Finding the points along given directions
        #--------------------------------------------      
        op_scc = kd.symmetry.op_scc
        N_c = kd.N_c
        wpts_xc = kpoints

        x_x = []
        k_xc = []
        k_x = []
        x = 0.
        X = []
        for nwpt in range(1, len(wpts_xc)):
            X.append(x)
            to_c = wpts_xc[nwpt]
            from_c = wpts_xc[nwpt - 1]
            vec_c = to_c - from_c
            print 'From ',from_c,' to ',to_c
            Nv_c = (vec_c * N_c).round().astype(int)
            Nv = abs(gcd(gcd(Nv_c[0], Nv_c[1]), Nv_c[2]))
            print Nv,' points found'
            dv_c = vec_c / Nv
            dv_v = np.dot(dv_c, bcell_cv)
            dx = np.linalg.norm(dv_v)
            if nwpt == len(wpts_xc) - 1:
                #X.append(Nv * dx)
                Nv += 1
            for n in range(Nv):
                k_c = from_c + n * dv_c
                bzk_c = to1bz(np.array([k_c]), acell_cv)[0]
                ikpt = kd.where_is_q(bzk_c,kd.bzk_kc)
                x_x.append(x)
                k_xc.append(k_c)
                k_x.append(ikpt)
                x += dx
        X.append(x_x[-1])
        if plot_BZ is True:      
            for ik in range(len(k_xc)):
                ktemp_xcv = np.dot(k_xc[ik],bcell_cv)
                plt.plot(ktemp_xcv[0],ktemp_xcv[1],'xr',markersize=10)
            plt.show()

        return x_x,k_xc,k_x,X

    # Finding the vacuum level through Hartree potential
    #------------------------------------------------------------------------------
    def get_vacuum_level(self,plot_pot=False):
        calc = self.calc

        calc.restore_state()
        vHt_g = calc.hamiltonian.pd3.ifft(calc.hamiltonian.vHt_q) * Hartree
        vHt_z = np.mean(np.mean(vHt_g, axis=0), axis=0)
        
        if plot_pot==True:
            import matplotlib.pyplot as plt
            plt.plot(vHt_z)
            plt.show()
        return vHt_z[0]

    # Getting SpinOrbit eigenvalues
    #------------------------------------------------------------------------------
    def get_spinorbit_corrections(self, return_spin=True, return_wfs=False, bands=None):
        calc = self.calc
        e_skn = self.gw_file['qp_skn']    
        
        bandrange = self.gw_file['bandrange']

        eSO_nk, s_nk, v_knm = get_spinorbit_eigenvalues(calc, return_spin=return_spin, return_wfs=return_wfs,
                                                 bands=range(bandrange[0],bandrange[-1]),
                                                 GW=True, eGW_skn=e_skn*Hartree)       
        #eSO_kn = np.sort(e_skn,axis=2)
        e_kn = eSO_nk.T/Hartree
        return e_kn, v_knm

    # Getting Eigenvalues along the path
    #------------------------------------------------------------------------------
    def get_GW_bands(self,nk_Int=50,Interpolate=False,SO=False):
        kd = self.kd
        if SO == True:
            e_kn, v_knm = self.get_spinorbit_corrections(return_wfs=True)
        else:
            e_kn = self.gw_file['qp_skn'][0]
            #e_kn = np.sort(e_kn,axis=2)
        bandrange = self.gw_file['bandrange']

        ef = self.calc.get_fermi_level()
        evac = self.get_vacuum_level()
        x_x,k_xc,k_x,X = self.find_k_along_path(plot_BZ=False)
        
        k_ibz_x = np.zeros_like(k_x)
        eGW_kn = np.zeros((len(k_x),e_kn.shape[1]))
        for n in range(e_kn.shape[1]):
            for ik in range(len(k_x)):
                ibzkpt = kd.bz2ibz_k[k_x[ik]]
                k_ibz_x[ik] = ibzkpt
                eGW_kn[ik,n] = e_kn[ibzkpt,n]*Hartree        

        N_occ = (eGW_kn[0] < ef).sum()
        print ' ' 
        print 'The number of Occupied bands is:', N_occ+bandrange[0]
        gap = (eGW_kn[:,N_occ].min()-eGW_kn[:,N_occ-1].max())
        print 'The bandgap is: %f'%gap
        print 'The valence band is at k=', x_x[eGW_kn[:,N_occ-1].argmax()]
        print 'The the conduction band is at k=', x_x[eGW_kn[:,N_occ].argmin()]
        vbm = eGW_kn[abs(x_x-X[2]).argmin(),N_occ-1]-evac
        cbm = eGW_kn[abs(x_x-X[2]).argmin(),N_occ]-evac
        print 'The valence band at K is=', vbm
        print 'The conduction band at K is=', cbm
        vbm2 = eGW_kn[abs(x_x-X[2]).argmin(),N_occ-3]-evac
        cbm2 = eGW_kn[abs(x_x-X[2]).argmin(),N_occ+2]-evac
        

        if Interpolate==True:
            xfit_k = np.linspace(x_x[0],x_x[-1],nk_Int)
            efit_kn = np.zeros((nk_Int,eGW_kn.shape[1]))
            for n in range(eGW_kn.shape[1]):
                fit_e = InterpolatedUnivariateSpline(x_x,eGW_kn[:,n])
                efit_kn[:,n] = fit_e(xfit_k)
            if SO == False:
                return xfit_k,X,efit_kn-evac,ef-evac,gap,vbm,cbm,vbm2,cbm2
            else:
                print('At the moment I cannot return the interpolated wavefuctions with SO=True, so be happy with what you got!')
                return xfit_k,X,efit_kn-evac,ef-evac,gap,vbm,cbm,vbm2,cbm2
        else:
            if SO == False:    
                return x_x,X,k_ibz_x,eGW_kn-evac,ef-evac,gap,vbm,cbm,vbm2,cbm2
            else:
                return x_x,X,k_ibz_x,eGW_kn-evac,v_knm,ef-evac,gap,vbm,cbm,vbm2,cbm2

