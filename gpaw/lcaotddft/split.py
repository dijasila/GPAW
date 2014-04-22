from gpaw.mpi import world
import numpy as np
import time
from ase.io import read
from ase import Atoms
from gpaw.lfc import LFC
from gpaw.mixer import Mixer
from gpaw.analyse.observers import Observer 
from math import sqrt, pi

class VRespCollector(Observer):
    def __init__(self, filename, lcao):
        Observer.__init__(self)
        self.lcao = lcao
        self.filename = filename
       
    def update(self):
        x_sg = self.lcao.hamiltonian.xc.xcs['RESPONSE'].vt_sG        
        x_sg = self.lcao.density.gd.collect(x_sg, broadcast=False)
        iter = self.niter
        if world.rank==0:
             fname = self.filename+"."+str(iter)+".vresp"
             x_sg.astype(np.float32).tofile(fname)

class DensityCollector(Observer):
    def __init__(self, filename, lcao):
        Observer.__init__(self)
        self.lcao = lcao
        self.filename = filename
       
    def update(self):
        self.lcao.timer.start('Dump density')
        x_g = self.lcao.density.rhot_g
        x_G = self.lcao.density.gd.zeros()
        self.lcao.hamiltonian.restrict(x_g, x_G)
        x_G = self.lcao.density.gd.collect(x_G, broadcast=False)
        iter = self.niter
        if world.rank==0:
             fname = self.filename+"."+str(iter)+".density"
             x_G.astype(np.float32).tofile(fname)
        self.lcao.timer.stop('Dump density')


class SplitDensityCollector(Observer):
    def __init__(self, filename, lcao, nsplit):
        Observer.__init__(self)
        self.lcao = lcao
        self.lcao.timer.start('Split density init')
        self.filename = filename
        self.nsplit = nsplit
        # Density arrays, files, and occupation numbers
        self.n_xsg = []
        #self.f_x = []
        self.f_xn = []
        #for i in range(2):
        #    if world.rank == 0:
        #        self.f_x.append(open(filename+'.'+str(i)+'.density','w'))
        #    else:
        #        self.f_x.append(None)
        for i in range(2):
            self.n_xsg.append(lcao.density.gd.zeros(1))
        f_n = lcao.wfs.kpt_u[0].f_n.copy()
        f2_n = lcao.wfs.kpt_u[0].f_n.copy()
        for n in range(lcao.wfs.bd.nbands):
            band_rank, myn = lcao.wfs.bd.who_has(n)
            if lcao.wfs.bd.rank == band_rank:
                if n < nsplit:
                    f_n[myn] = 0.0
                else:
                    f2_n[myn] = 0.0
        self.f_xn.append(f_n)
        self.f_xn.append(f2_n)
        #self.lcao.timer.start('Split density create LFC')
        #print "Creating lfc"
        #self.ghat = LFC(self.lcao.wfs.gd, [setup.ghat_l for setup in self.lcao.density.setups],
        #                integral=sqrt(4 * pi), forces=False)
        #spos_ac = self.lcao.atoms.get_scaled_positions() % 1.0
        #self.ghat.set_positions(spos_ac)
        #print "LFC created"
        self.lcao.timer.stop('Split density init')

        self.update()
  
    def update(self):
        self.lcao.timer.start('Split density dump')
        iter = self.niter
        if iter % 5 != 0:
            return
        start = time.time() 
        kpt = self.lcao.wfs.kpt_u[0]
        for i, (f_n, n_sg) in enumerate(zip(self.f_xn, self.n_xsg)):
            n_sg[:] = 0.0 # XXX n_sg just temporary here
            self.lcao.wfs.add_to_density_from_k_point_with_occupation(n_sg, kpt, f_n)

            D_asp={}
            for a in self.lcao.density.D_asp:
                ni = self.lcao.density.setups[a].ni
                D_asp[a] = np.zeros((1, ni * (ni + 1) // 2))
            self.lcao.wfs.calculate_atomic_density_matrices_with_occupation(D_asp, f_n)
            for a, D_sp in D_asp.items():
                Q_aL = np.dot(D_sp.sum(0),
                                   self.lcao.density.setups[a].Delta_pL)
                qfile = open("%s.%i.atom.%d" % (self.filename,i,a),'a+')
                Q_aL.astype(np.float32).tofile(qfile)
                qfile.close()
            if self.lcao.wfs.bd.comm.rank == 0:
                gd = self.lcao.wfs.gd
                fname = self.filename+"."+str(i)+".%.5f.%d.%d.%d-%d.%d.%d.density" % (self.lcao.time, gd.beg_c[0], gd.beg_c[1], gd.beg_c[2], gd.end_c[0], gd.end_c[1], gd.end_c[2])
                n_sg.astype(np.float32).tofile(fname)
        took = time.time()-start
        if world.rank == 0:
            print "Density dump took:", took, "s"
        self.lcao.timer.stop('Split density dump')


"""
atoms = read('../geometries/ag13.traj')
atoms.center(vacuum=5)

calc = LCAOTDDFT(mode='lcao', dtype=complex, width=0.01, basis='LDA.dz+5p', xc='LDA', h=0.3, 
                 mixer=Mixer(0.1,4,weight=100))
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.absorption_kick([0.001,0.00,0.00])
split = SplitDensityCollector('ag13', calc, 13*5)
calc.attach(split, 1)
calc.propagate(10, 1000)

"""
