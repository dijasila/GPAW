from gpaw.response.pair import PairDensity, KPointPair
from gpaw import GPAW
import numpy as np
from gpaw.spinorbit import BZWaveFunctions, soc_eigenstates
from ase.dft.kpoints import bandpath
from ase.units import Bohr, Hartree, kJ, _me, _e, _hplanck
from gpaw.mpi import world, serial_comm, broadcast, rank
from gpaw.occupations import fermi_dirac

def parallelisation_sizes(kpts_k, rank=None):
    if rank is None:
        rank = world.rank
    nK = len(kpts_k) 
    myKsize = -(-nK // world.size)
    myKrange = range(rank * myKsize,
                     min((rank + 1) * myKsize, nK))
    myKsize = len(myKrange)
    return myKrange, myKsize


def get_berrycurvature(calc, kpts_k=None, n1=0, n2=None, mi=None, mf=None, width=None,
                              theta=0.0, phi=0.0, scale=1.0, fermi_shift=None):
    kpts_kc = calc.get_bz_k_points()
    if kpts_k == None:
        kpts_k = range(len(kpts_kc))

    if width is not None:
        calc.wfs.occupations._width = width

    if n2 == None:
        n2 = calc.get_number_of_bands()
    bands_n = range(n1, n2)

    pdensity = PairDensity(calc, txt=None, communicator=serial_comm)

    myKrange, myKsize = parallelisation_sizes(kpts_k)

    soc = soc_eigenstates(calc, scale=scale, theta=theta, phi=phi, n1=n1, n2=n2) 

    if fermi_shift == None:
        wfs_k = soc.wfs
    else:
        wfs_k = new_occupations(fermi_shift, calc, kpts_k, soc, width=calc.wfs.occupations._width)

    v_kmsn = soc.eigenvectors()
    e_km = soc.eigenvalues() / Hartree
    Ns = calc.get_number_of_spins()
    Nn = len(bands_n)
    
    berryc_xy_k = np.zeros(len(kpts_k))
    berryc_yz_k = np.zeros(len(kpts_k))
    berryc_zx_k = np.zeros(len(kpts_k))

    threshold = 1

    for i in myKrange:
        ik = kpts_k[i]
        k_c = kpts_kc[ik]

        if mf is not None:
            f_m[:mf+1-2*n1] = 1
            f_m[mf+1-2*n1:] = 0
        if mi is not None:
            f_m[:mi-2*n1] = 0

        f_m = wfs_k[ik].f_m
        f_mm = (f_m[:, np.newaxis] - f_m[:])
        
        deps_mm = e_km[ik,:, np.newaxis] - e_km[ik, :]
        deps_mm[deps_mm == 0.0] = np.inf

        smallness_mm = np.abs(-1e-3 / deps_mm)
        inds_mm = (np.logical_and(np.inf > smallness_mm,
                                  smallness_mm > threshold))

        frac_mm = f_mm / np.square(deps_mm)
        frac_mm[inds_mm] = 0
        frac_mm = np.nan_to_num(frac_mm)

        rho_snnv = np.zeros((2, Nn, Nn, 3), complex)
        for s in range(Ns):
            kpt = pdensity.get_k_point(s, k_c, n1, n2)
            for n in bands_n:
                rho_snnv[s, n] = pdensity.optical_pair_velocity(n, kpt, kpt)

        if Ns == 1:
            rho_snnv[1] = rho_snnv[0]
        
        v_msn = v_kmsn[ik]
        rho_mmv = np.dot(v_msn[:, 0].conj(),                                       
                           np.dot(v_msn[:, 0], rho_snnv[0]))                             
        rho_mmv += np.dot(v_msn[:, 1].conj(),                                       
                          np.dot(v_msn[:, 1], rho_snnv[1]))                             

        
        A_mm = rho_mmv[:, :, 0].conj() * rho_mmv[:, :, 1]
        berryc_xy_k[i] = np.einsum('mn, mn', A_mm, frac_mm).imag
        
        A_mm = rho_mmv[:, :, 1].conj() * rho_mmv[:, :, 2]
        berryc_yz_k[i] = np.einsum('mn, mn', A_mm, frac_mm).imag
        
        A_mm = rho_mmv[:, :, 2].conj() * rho_mmv[:, :, 0]
        berryc_zx_k[i] = np.einsum('mn, mn', A_mm, frac_mm).imag

    berryc_xy_k = berryc_xy_k * Bohr**2
    berryc_yz_k = berryc_yz_k * Bohr**2
    berryc_zx_k = berryc_zx_k * Bohr**2

    world.sum(berryc_xy_k)
    world.sum(berryc_yz_k)
    world.sum(berryc_zx_k)

    return berryc_xy_k, berryc_yz_k, berryc_zx_k

def get_optical_conductivity(calc, kpts_k=None, n1=0, n2=None, mi=None, mf=None, width=None,
                              theta=0.0, phi=0.0, scale=1.0, fermi_shift=None):
    berryc_xy_k, berryc_yz_k, berryc_zx_k = get_berrycurvature(calc, kpts_k=None, n1=0, n2=None, mi=None, mf=None, width=None,
                                                               theta=0.0, phi=0.0, scale=1.0, fermi_shift=fermi_shift)
    
    sigma_xy = (_e**2/_hplanck)*berryc_xy_k
    sigma_yz = (_e**2/_hplanck)*berryc_yz_k
    sigma_zx = (_e**2/_hplanck)*berryc_zx_k

    return sigma_xy, sigma_yz, sigma_zx

def new_occupations(fermi_shift, calc, kpts_k, soc, width):

    e_km = soc.eigenvalues()
    efermi = soc._calculate_occ_numbers_and_fermi_level()
    new_efermi = efermi + fermi_shift
    
    for i in kpts_k:            
        soc.wfs[i].f_m = fermi_dirac(e_km[i], new_efermi, width)[0]
        
    return soc.wfs
    
