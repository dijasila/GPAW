from numpy import array,zeros,sum,abs,dot,where,reshape,argmax
import gpaw.mpi as mpi

"""
def dscf_find_lumo(calc,band):

    # http://trac.fysik.dtu.dk/projects/gpaw/browser/trunk/doc/documentation/dscf/lumo.py?format=raw

    assert band in [5,6]

    #Find band corresponding to lumo
    lumo = calc.get_pseudo_wave_function(band=band, kpt=0, spin=0)
    lumo = reshape(lumo, -1)

    wf1_k = [calc.get_pseudo_wave_function(band=5, kpt=k, spin=0) for k in range(calc.nkpts)]
    wf2_k = [calc.get_pseudo_wave_function(band=6, kpt=k, spin=0) for k in range(calc.nkpts)]

    band_k = []
    for k in range(calc.nkpts):
        wf1 = reshape(wf1_k[k], -1)
        wf2 = reshape(wf2_k[k], -1)
        p1 = abs(dot(wf1, lumo))
        p2 = abs(dot(wf2, lumo))

        if p1 > p2:
            band_k.append(5)
        else:
            band_k.append(6)

    return band_k
"""

# -------------------------------------------------------------------

def mpi_debug(text):
    if isinstance(text,list):
        for t in text:
            mpi_debug(t)
    else:
        print 'mpi.rank=%d, %s' % (mpi.rank,text)

# -------------------------------------------------------------------

def dscf_find_atoms(atoms,symbol):
    chemsyms = atoms.get_chemical_symbols()
    return where(map(lambda s: s==symbol,chemsyms))[0]

# -------------------------------------------------------------------

def dscf_find_bands(calc,bands,datas=None,allspins=False,debug=False):
    """Entirely serial, but works regardless of parallelization. DOES NOT WORK WITH DOMAIN-DECOMPOSITION IN GPAW v0.5.2725 """ #TODO!

    if datas is None:
        datas = range(len(bands))
    else:
        assert len(bands)==len(datas), 'Length mismatch.'

    if allspins:
        raise NotImplementedError, 'Currently only the spin-down case is considered...' #TODO!

    # Extract wave functions for each band and k-point
    wf_knG = []
    for k in range(calc.nkpts):
        wf_knG.append([reshape(calc.get_pseudo_wave_function(band=n,kpt=k,spin=0),-1) for n in bands]) #calc.get_pseudo fails with domain-decomposition from tar-file

    # Extract wave function for each band of the Gamma point
    gamma_nG = wf_knG[0]

    if debug: mpi_debug('wf_knG='+str(wf_knG))

    band_kn = []
    data_kn = []

    for k in range(calc.nkpts):
        band_n = []
        data_n = []

        for n in range(len(bands)):
            # Find the band for this k-point which corresponds to bands[n] of the Gamma point
            wf = wf_knG[k][n]
            p = argmax([abs(dot(wf,gamma_nG[m])) for m in range(len(bands))])
            band_n.append(bands[p])
            data_n.append(datas[p])

        band_kn.append(band_n)
        data_kn.append(data_n)

    return (band_kn,data_kn,)

# -------------------------------------------------------------------

def dscf_linear_combination(calc, molecule, bands, coefficients, debug=False):
    """Full parallelization over k-point - grid-decomposition parallelization needs heavy testing.""" #TODO!

    if debug: dumpkey = mpi.world.size == 1 and 'serial' or 'mpi'

    (band_kn,coeff_kn,) = dscf_find_bands(calc,bands,coefficients)

    if debug: mpi_debug('nkpts=%d, ni=%d, len(calc.kpt_u)=%d' % (calc.nkpts,calc.setups[0].ni, len(calc.kpt_u)))
    if debug: mpi_debug('band_kn=%s, coeff_kn=%s' % (str(band_kn),str(coeff_kn)))
    if debug: mpi_debug(['u=%d, k=%d, s=%d, kpt.comm.rank=%d, k_c=%s' % (kpt.u,kpt.k,kpt.s,kpt.comm.rank,kpt.k_c) for kpt in calc.kpt_u])

    P_aui = zeros((len(molecule),len(calc.kpt_u),calc.setups[0].ni),dtype=complex)

    for (m,a) in enumerate(molecule):
        if calc.nuclei[a].in_this_domain:
            """
            if debug:
                for n in bands:
                    print 'mpi.rank=%d, calc.nuclei[%d].P_uni[:,%d,:].shape=' % (mpi.rank,a,n), calc.nuclei[a].P_uni[:,n,:].shape

                print 'mpi.rank=%d, test.shape=' % mpi.rank, sum([c*calc.nuclei[a].P_uni[:,n,:] for (c,n) in zip(coefficients,bands)],axis=0).shape
            """
            if debug: mpi_debug(['calc.nuclei[%d].P_uni[:,%d,:].shape=%s' % (a,n,str(calc.nuclei[a].P_uni[:,n,:].shape)) for n in bands])

            #P_aui[m] += sum([c*calc.nuclei[a].P_uni[:,n,:] for (c,n) in zip(coefficients,bands)],axis=0)

            for kpt in calc.kpt_u:
                band_n = band_kn[kpt.k]
                coeff_n = coeff_kn[kpt.k]
                P_aui[m][kpt.u] += sum([c*calc.nuclei[a].P_uni[kpt.u,n,:] for (c,n) in zip(coeff_n,band_n)],axis=0)

                if debug: calc.nuclei[a].P_uni[kpt.u,:,:].dump('dscf_tool_P_uni_a%01d_k%01ds%01d_%s%02d.pickle' % (a,kpt.k,kpt.s,dumpkey,mpi.rank))

    calc.gd.comm.sum(P_aui)

    if debug: P_aui.dump('dscf_tool_P_aui_%s%02d.pickle' % (dumpkey,mpi.rank))

    """
    if debug and mpi.rank == 0:
        print 'P_aui.shape=',P_aui.shape

        for (a,P_ui) in enumerate(P_aui):
            print 'P_aui[%d].shape=' % a,P_ui.shape

        print 'P_aui=',P_aui

        print 'gd.Nc=',calc.gd.N_c
    """

    if debug: mpi_debug('P_aui.shape='+str(P_aui.shape))

    #wf_u = [sum([c*calc.kpt_u[u].psit_nG[n] for (c,n) in zip(coefficients,bands)],axis=0) for u in range(0,len(calc.kpt_u))]
    #wf_u = zeros((calc.nkpts,calc.gd.N_c[0]-1,calc.gd.N_c[1]-1,calc.gd.N_c[2]-1))#,dtype=complex)
    wf_u = calc.gd.zeros(len(calc.kpt_u),dtype=complex)

    gd_slice = calc.gd.get_slice()

    if debug: mpi_debug('gd_slice='+str(gd_slice))

    for kpt in calc.kpt_u:
        if debug: mpi_debug('u=%d, k=%d, s=%d, kpt.comm.rank=%d, calc.kpt_comm.rank=%d, gd.shape=%s, psit.shape=%s' % (kpt.u,kpt.k,kpt.s,kpt.comm.rank,calc.kpt_comm.rank,str(wf_u[0].shape),str(array(kpt.psit_nG[0])[gd_slice].shape)))

        #wf_u[kpt.u] += sum([c*array(kpt.psit_nG[n])[gd_slice] for (c,n) in zip(coefficients,bands)],axis=0)

        band_n = band_kn[kpt.k]
        coeff_n = coeff_kn[kpt.k]
        wf_u[kpt.u] += sum([c*array(kpt.psit_nG[n])[gd_slice] for (c,n) in zip(coeff_n,band_n)],axis=0)

    #calc.gd.comm.sum(wf_u)

    if debug: mpi_debug('|wf_u|^2=%s' % str([sum(abs(wf.flatten())**2) for wf in wf_u]))

    """
    if debug and mpi.rank == 0:
        print 'wf_u.shape=',wf_u.shape

        for (u,wf) in enumerate(wf_u):
            print 'wf[%d].shape=' % u,wf.shape

        for (u,wf) in enumerate(wf_u):
            print 'wf_u[%d].shape=' % u,wf.shape
    """

    return (P_aui,wf_u,)

