from __future__ import print_function
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.xc.libvdwxc import vdw_df, vdw_df2, vdw_df_cx, \
    vdw_optPBE, vdw_optB88, vdw_C09, vdw_beef, \
    libvdwxc_has_mpi, libvdwxc_has_pfft

# This test verifies that the results returned by the van der Waals
# functionals implemented in libvdwxc do not change.

N_c = np.array([23, 10, 6])
gd = GridDescriptor(N_c, N_c * 0.2, pbc_c=(1, 0, 1))

n_sg = gd.zeros(1)
nG_sg = gd.collect(n_sg)
if gd.comm.rank == 0:
    gen = np.random.RandomState(0)
    nG_sg[:] = gen.rand(*nG_sg.shape)
gd.distribute(nG_sg, n_sg)

for mode in ['serial', 'mpi', 'pfft']:
    if mode == 'serial' and gd.comm.size > 1:
        continue
    if mode == 'mpi' and not libvdwxc_has_mpi():
        continue
    if mode == 'pfft' and not libvdwxc_has_pfft():
        continue

    errs = []

    def test(vdwxcclass, Eref=np.nan, nvref=np.nan):
        print('')
        xc = vdwxcclass(mode=mode)
        xc.initialize_backend(gd)
        if gd.comm.rank == 0:
            print(xc.libvdwxc.tostring())
        v_sg = gd.zeros(1)
        E = xc.calculate(gd, n_sg, v_sg)
        nv = gd.integrate(n_sg * v_sg, global_integral=True)
        nv = float(nv)  # Comes out as an array due to spin axis

        Eerr = abs(E - Eref)
        nverr = abs(nv - nvref)
        errs.append((vdwxcclass.__name__, Eerr, nverr))

        if gd.comm.rank == 0:
            name = xc.name
            print(name)
            print('=' * len(name))
            print('E  = %19.16f vs ref = %19.16f :: err = %10.6e'
                  % (E, Eref, Eerr))
            print('nv = %19.16f vs ref = %19.16f :: err = %10.6e'
                  % (nv, nvref, nverr))
            print()
        gd.comm.barrier()

        print('Update:')
        print('    test({}, {!r}, {!r})'.format(vdwxcclass.__name__,
                                                E, nv))

    test(vdw_df, -3.7373237065604763, -4.7766307896135345)
    test(vdw_df2, -3.756806531923722, -4.791445795325973)
    test(vdw_df_cx, -3.6297376828922516, -4.675348843676276)
    test(vdw_optPBE, -3.6836013806903409, -4.729000723771955)
    test(vdw_optB88, -3.7182162928044207, -4.7586587646972545)
    test(vdw_C09, -3.6178542857032827, -4.6660965477389125)
    test(vdw_beef, -3.7742681088117678, -4.852078112171113)

    if any(err[1] > 1e-14 or err[2] > 1e-14 for err in errs):
        # Try old values (compatibility)
        del errs[:]

        test(vdw_df, -3.7373236650435588, -4.77663026883604)
        test(vdw_df2, -3.7568066347104221, -4.791445146559045)
        test(vdw_df_cx, -3.6297376413753346, -4.67534832289878)
        test(vdw_optPBE, -3.6836013391734239, -4.729000202994461)
        test(vdw_optB88, -3.7182162512875037, -4.758658243919759)
        test(vdw_C09, -3.6178542441863657, -4.666096026961417)
        test(vdw_beef, -3.7742682115984687, -4.8520774634041866)

        for name, Eerr, nverr in errs:
            assert Eerr < 1e-14 and nverr < 1e-14, (name, Eerr, nverr)
