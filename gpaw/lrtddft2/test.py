### run with at least 24 mpi-procs

import sys
from ase import *
from ase.io import *
from gpaw import *
import gpaw.mpi

from gpaw.lrtddft import *
from lrigpaw.lrtddft2 import *


nanal = 9

###############################################################################
# GS
###############################################################################
if not os.path.exists('h2s2.gpw'):
    atoms = Atoms( 'H2S2',
                   positions=[ (0.90, -0.30,  -1.35),
                               (0.90,  0.30,   1.35),
                               (0.00,  0.00,  -1.00),
                               (0.00,  0.00,   1.00) ],
                   cell = (11.2, 11.2, 12.8) )
    atoms.center()

    calc=GPAW(h=0.2, xc='PBE', nbands=30, eigensolver='cg', convergence={'bands': 20})

    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    calc.write('h2s2.gpw',mode='all')


###############################################################################
def do_new(istart, pend, ediff, dd, eh, sl=None):
    world = gpaw.mpi.world.new_communicator(range(dd*eh))
    if world is None: return
    dd_comm, eh_comm = lr_communicators(world, dd, eh)

    txtid = "i%04d_p%04d_ediff%06.3lf_dd%04d_eh%04d" % (istart, pend, ediff,
                                                        dd, eh)
    if sl is not None: txtid += "_%03d,%03d,%03d" % (sl[0],sl[1],sl[2])


    
    if gpaw.mpi.world.rank == 0: txt = 'h2s2_lri_' + txtid + '.txt'
    else:                        txt = '/dev/null'


    if sl is not None:
        calc = GPAW('h2s2.gpw', communicator=dd_comm, txt=txt,
                    parallel={'sl_lrtddft': sl})
    else:
        calc = GPAW('h2s2.gpw', communicator=dd_comm, txt=txt)


    lr = LrTDDFTindexed( 'h2s2_lri_' + txtid,
                         calc = calc,
                         xc = 'PBE',
                         min_occ=istart,
                         max_occ=pend,
                         min_unocc=istart,
                         max_unocc=pend,
                         max_energy_diff=ediff,
                         eh_communicator=eh_comm,
                         recalculate='all' )

    lr.calculate_excitations()

    trans = []
    for k in range(nanal):
        w = lr.get_excitation_energy(k)
        f = lr.get_oscillator_strength(k)
        R = lr.get_rotatory_strength(k)
        trans.append([w,f,R])

    if gpaw.mpi.world.rank == 0:
        for (k,t) in enumerate(trans):
            [w,f,R] = t
            print >> calc.txt, "LrTrans #%04d   %18.7lf %18.7lf %18.7lf" % (k, w*27.211,f,R*64604.8164)

    return trans



###############################################################################
def do_rst(istart, pend, ediff, dd, eh, istart2, pend2, ediff2, sl=None):
    world = gpaw.mpi.world.new_communicator(range(dd*eh))
    if world is None: return
    dd_comm, eh_comm = lr_communicators(world, dd, eh)

    txtid = "i%04d_p%04d_ediff%06.3lf_dd%04d_eh%04d" % (istart, pend, ediff, dd, eh)
    
    txtid2 = "i%04d_p%04d_ediff%06.3lf_dd%04d_eh%04d_ii%04d_pp%04d" % (istart, pend, ediff, dd, eh, istart2, pend2)
    
    if sl is not None: txtid += "_%03d,%03d,%03d" % (sl[0],sl[1],sl[2])
    if sl is not None: txtid2 += "_%03d,%03d,%03d" % (sl[0],sl[1],sl[2])


    
    if gpaw.mpi.world.rank == 0: txt = 'h2s2_lri_rst_' + txtid2 + '.txt'
    else:                        txt = '/dev/null'


    if sl is not None:
        calc = GPAW('h2s2.gpw', communicator=dd_comm, txt=txt,
                    parallel={'sl_lrtddft': sl})
    else:
        calc = GPAW('h2s2.gpw', communicator=dd_comm, txt=txt)


    lr = LrTDDFTindexed( 'h2s2_lri_' + txtid,
                         calc = calc,
                         xc = 'PBE',
                         min_occ=istart2,
                         max_occ=pend2,
                         min_unocc=istart2,
                         max_unocc=pend2,
                         max_energy_diff=ediff2,
                         eh_communicator=eh_comm )

    lr.calculate_excitations()

    trans = []
    for k in range(nanal):
        w = lr.get_excitation_energy(k)
        f = lr.get_oscillator_strength(k)
        R = lr.get_rotatory_strength(k)
        trans.append([w,f,R])

    if gpaw.mpi.world.rank == 0:
        for (k,t) in enumerate(trans):
            [w,f,R] = t
            print >> calc.txt, "LrTrans #%04d   %18.7lf %18.7lf %18.7lf" % (k, w*27.211,f,R*64604.8164)

    return trans



###############################################################################
def do_eig(istart, pend, ediff, dd, eh, sl=None):
    world = gpaw.mpi.world.new_communicator(range(dd*eh))
    if world is None: return
    dd_comm, eh_comm = lr_communicators(world, 1, dd*eh)

    txtid = "i%04d_p%04d_ediff%06.3lf_dd%04d_eh%04d" % (istart, pend, ediff,
                                                        dd, eh)
    if sl is not None: txtid += "_%03d,%03d,%03d" % (sl[0],sl[1],sl[2])


    
    if gpaw.mpi.world.rank == 0: txt = 'h2s2_lri_eig_' + txtid + '.txt'
    else:                        txt = '/dev/null'


    if sl is not None:
        calc = GPAW('h2s2.gpw', communicator=dd_comm, txt=txt,
                    parallel={'sl_lrtddft': sl})
    else:
        calc = GPAW('h2s2.gpw', communicator=dd_comm, txt=txt)


    lr = LrTDDFTindexed( 'h2s2_lri_' + txtid,
                         calc = calc,
                         xc = 'PBE',
                         min_occ=istart,
                         max_occ=pend,
                         min_unocc=istart,
                         max_unocc=pend,
                         max_energy_diff=ediff,
                         eh_communicator=eh_comm,
                         recalculate='eigen' )

    lr.calculate_excitations()

    trans = []
    for k in range(nanal):
        w = lr.get_excitation_energy(k)
        f = lr.get_oscillator_strength(k)
        R = lr.get_rotatory_strength(k)
        trans.append([w,f,R])

    if gpaw.mpi.world.rank == 0:
        for (k,t) in enumerate(trans):
            [w,f,R] = t
            print >> calc.txt, "LrTrans #%04d   %18.7lf %18.7lf %18.7lf" % (k, w*27.211,f,R*64604.8164)

    return trans



###############################################################################
def do_old(istart, pend, ediff, dd, eh, sl):
    world = gpaw.mpi.world.new_communicator(range(dd*eh))
    if world is None: return
    dd_comm, eh_comm = lr_communicators(world, dd, eh)
        
    txtid = "i%04d_p%04d_ediff%06.3lf_dd%04d_eh%04d" % (istart, pend, ediff,
                                                        dd, eh)
    

    if gpaw.mpi.world.rank == 0: gstxt = 'h2s2_gs_old_' + txtid + '.txt'
    else:                        gstxt = '/dev/null'

    txt = 'h2s2_lr_old_' + txtid + '.txt'

    lr = LrTDDFT( GPAW('h2s2.gpw', communicator=dd_comm, txt=gstxt),
                  xc = 'PBE',
                  istart=istart,
                  jend=pend,
                  #energy_range=ediff,
                  txt=txt )

    lr.diagonalize()
    lr.write('h2s2_lr_old' + txtid + '.lrdat')

    trans = []
    for k in range(nanal):
        w = lr[k].energy
        f = lr[k].get_oscillator_strength()[0]
        R = lr[k].get_rotatory_strength(units='a.u.')
        trans.append([w,f,R])

    if gpaw.mpi.world.rank == 0:
        for (k,t) in enumerate(trans):
            [w,f,R] = t
            print >> lr.txt, "LrTrans #%04d   %18.7lf %18.7lf %18.7lf" % (k, w*27.211,f,R*64604.8164)


    return trans


###############################################################################



###############################################################################
#
# Following tests:
# 1) old LrTDDFT vs new LrTDDFT
#    - istart=4 => pend=9 and istart=3 => pend=11
# 2) Lapack vs Scalapack
#    - istart=4 => pend=9 and istart=3 => pend=11
#    - clean   dd=8, eh=3: -sl_lrtddft=1,1,1 / 3,1,3 / 4,4,1 / 3,7,1
#    - restart dd=8, eh=2: -sl_lrtddft=1,1,1 / 3,1,3 / 3,7,1
# 3) Size change
#    - from istart=4 => pend=9 to istart=3 => pend=11
#    - from istart=3 => pend=11 to istart=4 => pend=9
# 3) Scalapack size change
#    - from istart=4 => pend=9 to istart=3 => pend=11
#    - from istart=3 => pend=11 to istart=4 => pend=9
#    - dd=8, eh=3: -sl_lrtddft=1,1,1 / 3,5,2
###############################################################################


data = {}

data[(4,9)] = [ [[  3.8307029,          0.0033744,          0.6852403],
                 [  5.0140338,          0.0091942,         56.4963774],
                 [  5.9941797,          0.0000016,          0.0585058],
                 [  6.2012587,          0.0000768,         -2.6735164],
                 [  7.3217155,          0.0834718,       -114.8496973],
                 [  8.1095916,          0.1802103,         62.1772987],
                 [  8.6754220,          0.0950949,          1.4506092],
                 [  9.4443891,          0.9660643,          3.8178493],
                 [ 10.1910041,          0.0122456,        -13.0075263] ] ]

data[(3,11)] = [ [ [   3.8216832,          0.0036721,          0.4962022],
                   [   5.0136285,          0.0090210,         55.9136106],
                   [   5.9827652,          0.0000695,          0.2920758],
                   [   6.1926338,          0.0001338,         -4.1875160],
                   [   6.2262262,          0.0462699,         21.4012398],
                   [   6.2763805,          0.0000161,          0.6371072],
                   [   7.3206578,          0.0816187,       -112.6221885],
                   [   7.7192300,          0.0042342,         12.7233311],
                   [   8.0121517,          0.1973411,         74.8663029] ] ]

for d in data[(4,9)]+data[(3,11)]:
    for t in d:
        t[0] /= 27.211
        t[1] /= 1.
        t[2] /= 64604.8164


ndd = 8
neh = 3

# 4 9
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, 1, None)
otrans = do_old(istart, pend, ediff, dd, eh, sl);   data[(istart,pend)].append(otrans)
trans = do_new(istart, pend, ediff, dd, eh, sl);    data[(istart,pend)].append(trans)
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, None)
trans = do_new(istart, pend, ediff, dd, eh, sl);    data[(istart,pend)].append(trans)

# 3 11
(istart,pend,ediff,dd,eh,sl) = (3,11,100., ndd, 1, None)
otrans = do_old(istart, pend, ediff, dd, eh, sl);    data[(istart,pend)].append(otrans)
trans = do_new(istart, pend, ediff, dd, eh, sl);     data[(istart,pend)].append(trans)


# Scalapack 4 9
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (1,1,1))
trans = do_new(istart, pend, ediff, dd, eh, sl);       data[(istart,pend)].append(trans)
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (3,1,3))
trans = do_new(istart, pend, ediff, dd, eh, sl);       data[(istart,pend)].append(trans)
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (4,4,1))
trans = do_new(istart, pend, ediff, dd, eh, sl);       data[(istart,pend)].append(trans)
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (3,7,1))
trans = do_new(istart, pend, ediff, dd, eh, sl);       data[(istart,pend)].append(trans)

# Scalapack eigen
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (1,1,1))
trans = do_eig(istart, pend, ediff, dd, eh, sl);       data[(istart,pend)].append(trans)
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (3,1,3))
trans = do_eig(istart, pend, ediff, dd, eh, sl);       data[(istart,pend)].append(trans)
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (3,7,1))
trans = do_eig(istart, pend, ediff, dd, eh, sl);       data[(istart,pend)].append(trans)


# Size change
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, None)
(istart2,pend2,ediff2) = (3,11,100.)
trans = do_rst(istart, pend, ediff, dd, eh, istart2, pend2, ediff2, sl);       data[(istart2,pend2)].append(trans)

(istart,pend,ediff,dd,eh,sl) = (3,11,100., ndd, neh, None)
(istart2,pend2,ediff2) = (4,9,100.)
trans = do_rst(istart, pend, ediff, dd, eh, istart2, pend2, ediff2, sl);       data[(istart2,pend2)].append(trans)


# Scalapack size changes
(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (3,1,3))
(istart2,pend2,ediff2) = (3,11,100.)
trans = do_rst(istart, pend, ediff, dd, eh, istart2, pend2, ediff2, sl);       data[(istart2,pend2)].append(trans)

(istart,pend,ediff,dd,eh,sl) = (4,9,100., ndd, neh, (3,7,1))
(istart2,pend2,ediff2) = (3,11,100.)
trans = do_rst(istart, pend, ediff, dd, eh, istart2, pend2, ediff2, sl);       data[(istart2,pend2)].append(trans)

(istart,pend,ediff,dd,eh,sl) = (3,11,100., ndd, neh, (3,7,1))
(istart2,pend2,ediff2) = (4,9,100.)
trans = do_rst(istart, pend, ediff, dd, eh, istart2, pend2, ediff2, sl);       data[(istart2,pend2)].append(trans)


if gpaw.mpi.world.rank == 0:
    d0 = data[(4,9)][0]
    for d in data[(4,9)][1:]:
        for (t,t0) in zip(d,d0):
            dw = t[0] - t0[0]
            df = t[1] - t0[1]
            dR = t[2] - t0[2]
            if abs(dw) > 1e-5 or abs(df) > 1e-5 or abs(dR) > 1e-4:
                print "%10.3le %10.3le %10.3le <= TOO LARGE" % (t[0]-t0[0], t[1]-t0[1], t[2]-t0[2])
            else:
                print "%10.3le %10.3le %10.3le" % (t[0]-t0[0], t[1]-t0[1], t[2]-t0[2])
        print

if gpaw.mpi.world.rank == 0:
    d0 = data[(3,11)][0]
    for d in data[(3,11)][1:]:
        for (t,t0) in zip(d,d0):
            dw = t[0] - t0[0]
            df = t[1] - t0[1]
            dR = t[2] - t0[2]
            if abs(dw) > 1e-5 or abs(df) > 1e-5 or abs(dR) > 1e-4:
                print "%10.3le %10.3le %10.3le <= TOO LARGE" % (t[0]-t0[0], t[1]-t0[1], t[2]-t0[2])
            else:
                print "%10.3le %10.3le %10.3le" % (t[0]-t0[0], t[1]-t0[1], t[2]-t0[2])
        print



