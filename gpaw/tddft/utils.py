# Written by Lauri Lehtovaara 2008

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc
from gpaw.utilities.blas import dotu

import _gpaw

from gpaw import debug_cuda,debug_cuda_reltol,debug_cuda_abstol
import pycuda.gpuarray as gpuarray

class MultiBlas:
    def __init__(self, gd, timer = None):
        self.gd = gd
        #self.timer = None
        self.timer = timer

    # Multivector ZAXPY: a x + y => y
    #def multi_zaxpy(self, a,x,y, nvec):
    #    if isinstance(a, (float, complex)):
            #for i in range(nvec):
            #    axpy(a*(1+0J), x[i], y[i])
    #    else:
    #        for i in range(nvec):
    #            axpy(a[i]*(1.0+0.0J), x[i], y[i])

    def multi_zaxpy_cpu(self, a,x,y, nvec):
        for i in range(nvec):
            axpy(a[i]*(1+0J), x[i], y[i])
            
    def multi_zaxpy(self, a,x,y, nvec):
        assert type(x) == type(y)
        if self.timer is not None:
            self.timer.start('multi_axpy')
        if isinstance(a, (float, complex)):
            axpy(a*(1+0J), x, y)
        else:
            if isinstance(x,gpuarray.GPUArray):                
                if debug_cuda:
                    #if 1:
                    x_cpu=x.get()
                    y_cpu=y.get()
                    self.multi_zaxpy_cpu(a,x_cpu,y_cpu, nvec)
                a_gpu=a*(1+0J)
                _gpaw.multi_axpy_cuda_gpu(a_gpu, x.gpudata, x.shape, y.gpudata, y.shape, x.dtype,nvec)
                #if 1:
                if debug_cuda:
                    if not  np.allclose(y_cpu,y.get(),
                                        debug_cuda_reltol,debug_cuda_abstol):
                        diff=abs(y_cpu-y.get())
                        error_i=np.unravel_index(np.argmax(diff - debug_cuda_reltol * abs(y_cpu)),diff.shape)
                        print "Debug cuda: multi_zaxpy max rel error: ",error_i,y_cpu[error_i],y.get()[error_i],abs(y_cpu[error_i]-y.get()[error_i])
                        error_i=np.unravel_index(np.argmax(diff),diff.shape)
                        print "Debug cuda: multi_zaxpy max abs error: ",error_i, y_cpu[error_i],y.get()[error_i],abs(y_cpu[error_i]-y.get()[error_i])
            else:
                self.multi_zaxpy_cpu(a,x,y, nvec)
        if self.timer is not None:
            self.timer.stop('multi_axpy')



    # Multivector dot product, a^H b, where ^H is transpose
    def multi_zdotc_cpu(self, s, x,y, nvec):
        for i in range(nvec):
            s[i] = dotc(x[i],y[i])
        return s
    
    def multi_zdotc(self, s, x,y, nvec):
        if self.timer is not None:
            self.timer.start('multi_zdotc')
        if isinstance(x,gpuarray.GPUArray):
            _gpaw.multi_dotc_cuda_gpu(x.gpudata, x.shape, y.gpudata, x.dtype, s)
        else:
            s=self.multi_zdotc_cpu(s, x,y, nvec)            
        self.gd.comm.sum(s)
        if self.timer is not None:
            self.timer.stop('multi_zdotc')
        return s

    def multi_zdotu_cpu(self, s, x,y, nvec):
        for i in range(nvec):
            s[i] = dotu(x[i],y[i])
        return s
    
    def multi_zdotu(self, s, x,y, nvec):
        if self.timer is not None:
            self.timer.start('multi_zdotu')
        if isinstance(x,gpuarray.GPUArray):
            _gpaw.multi_dotu_cuda_gpu(x.gpudata, x.shape, y.gpudata, x.dtype, s)
        else:
            s=self.multi_zdotu_cpu(s, x,y, nvec)
        self.gd.comm.sum(s)
        if self.timer is not None:
            self.timer.stop('multi_zdotu')
        return s

            
    # Multiscale: a x => x
    #    def multi_scale(self, a,x, nvec):
    #        if isinstance(a, (float, complex)):
    #            x *= a
    #        else:
    #            for i in range(nvec):
    #                x[i] *= a[i]
    # Multiscale: a x => x
    def multi_scale_cpu(self, a,x, nvec):
        for i in range(nvec):
            x[i] *= a[i]
            
    def multi_scale(self, a,x, nvec):
        if self.timer is not None:
            self.timer.start('multi_scale')
        if isinstance(a, (float, complex)):
            x *= a
        else:
            if isinstance(x,gpuarray.GPUArray):
                if debug_cuda:
                    #if 1:
                    x_cpu=x.get()
                    self.multi_scale_cpu(a,x_cpu, nvec)
                #a_gpu=gpuarray.to_gpu(a)            
                _gpaw.multi_scal_cuda_gpu(a, x.gpudata, x.shape, x.dtype,nvec)
                if debug_cuda:
                    #if 1:
                    if not  np.allclose(x_cpu,x.get(),
                                        debug_cuda_reltol,debug_cuda_abstol):
                        diff=abs(x_cpu-x.get())
                        error_i=np.unravel_index(np.argmax(diff - debug_cuda_reltol * abs(x_cpu)),diff.shape)
                        print "Debug cuda: multi_scal max rel error: ",error_i,y_cpu[error_i],y.get()[error_i],abs(x_cpu[error_i]-x.get()[error_i])
                        error_i=np.unravel_index(np.argmax(diff),diff.shape)
                        print "Debug cuda: multi_scal max abs error: ",error_i, y_cpu[error_i],y.get()[error_i],abs(x_cpu[error_i]-x.get()[error_i])
            else:
                self.multi_scale_cpu(a,x, nvec)
        if self.timer is not None:
            self.timer.stop('multi_scale')



# -------------------------------------------------------------------

import numpy as np

class BandPropertyMonitor:
    def __init__(self, wfs, name, interval=1):
        self.niter = 0
        self.interval = interval

        self.wfs = wfs

        self.name = name

    def __call__(self):
        self.update(self.wfs)
        self.niter += self.interval

    def update(self, wfs):
        #strictly serial XXX!
        data_un = []

        for u, kpt in enumerate(wfs.kpt_u):
            data_n = getattr(kpt, self.name)

            data_un.append(data_n)

        self.write(np.array(data_un))

    def write(self, data):
        pass

class BandPropertyWriter(BandPropertyMonitor):
    def __init__(self, filename, wfs, name, interval=1):
        BandPropertyMonitor.__init__(self, wfs, name, interval)
        self.fileobj = open(filename,'w')

    def write(self, data):
        self.fileobj.write(data.tostring())
        self.fileobj.flush()

    def __del__(self):
        self.fileobj.close()

# -------------------------------------------------------------------


class StaticOverlapMonitor:
    def __init__(self, wfs, wf_u, P_aui, interval=1):
        self.niter = 0
        self.interval = interval

        self.wfs = wfs

        self.wf_u = wf_u
        self.P_aui = P_aui

    def __call__(self):
        self.update(self.wfs)
        self.niter += self.interval

    def update(self, wfs, calculate_P_ani=False):

        #strictly serial XXX!
        Porb_un = []

        for u, kpt in enumerate(wfs.kpt_u):
            swf = self.wf_u[u].ravel()

            psit_n = kpt.psit_nG.reshape((len(kpt.f_n),-1))
            Porb_n = np.dot(psit_n.conj(), swf) * wfs.gd.dv

            P_ani = kpt.P_ani

            if calculate_P_ani:
                #wfs.pt.integrate(psit_nG, P_ani, kpt.q)
                raise NotImplementedError('In case you were wondering, TODO XXX')

            for a, P_ni in P_ani.items():
                sP_i = self.P_aui[a][u]
                for n in range(wfs.nbands):
                    for i in range(len(P_ni[0])):
                        for j in range(len(P_ni[0])):
                            Porb_n[n] += (P_ni[n][i].conj() *
                                       wfs.setups[a].dO_ii[i][j] *
                                       sP_i[j])

            Porb_un.append(Porb_n)

        self.write(np.array(Porb_un))

    def write(self, data):
        pass

class StaticOverlapWriter(StaticOverlapMonitor):
    def __init__(self, filename, wfs, overlap, interval=1):
        StaticOverlapMonitor.__init__(self, wfs, overlap, interval)
        self.fileobj = open(filename,'w')

    def write(self, data):
        self.fileobj.write(data.tostring())
        self.fileobj.flush()

    def __del__(self):
        self.fileobj.close()

# -------------------------------------------------------------------


class DynamicOverlapMonitor:
    def __init__(self, wfs, overlap, interval=1):
        self.niter = 0
        self.interval = interval

        self.setups = overlap.setups
        self.operator = overlap.operator
        self.wfs = wfs

    def __call__(self):
        self.update(self.wfs)
        self.niter += self.interval

    def update(self, wfs, calculate_P_ani=False):

        #strictly serial XXX!
        S_unn = []

        for kpt in wfs.kpt_u:
            psit_nG = kpt.psit_nG
            P_ani = kpt.P_ani

            if calculate_P_ani:
                #wfs.pt.integrate(psit_nG, P_ani, kpt.q)
                raise NotImplementedError('In case you were wondering, TODO XXX')

            # Construct the overlap matrix:
            S = lambda x: x
            dS_aii = dict([(a, self.setups[a].dO_ii) for a in P_ani])
            S_nn = self.operator.calculate_matrix_elements(psit_nG, P_ani,
                                                           S, dS_aii)
            S_unn.append(S_nn)

        self.write(np.array(S_unn))

    def write(self, data):
        pass

class DynamicOverlapWriter(DynamicOverlapMonitor):
    def __init__(self, filename, wfs, overlap, interval=1):
        DynamicOverlapMonitor.__init__(self, wfs, overlap, interval)
        self.fileobj = open(filename,'w')

    def write(self, data):
        self.fileobj.write(data.tostring())
        self.fileobj.flush()

    def __del__(self):
        self.fileobj.close()

