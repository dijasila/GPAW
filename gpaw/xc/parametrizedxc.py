from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc import LibXC
import numpy as np
import copy
import _gpaw
import re


class ParametrizedKernel(XCKernel):
    def __init__(self, name):
        self.name = name
        #self.functionals = ['LDA_X','LDA_K_TF', 'GGA_K_VW']
        self.kernels = []

        #Parse and validate name string
        coefficients = []
        xc_names = []
        #Little regexp magic, this matches to strings like "1.0*LDA_K_TF + 0.9*LDA_X"
        regex = '(?:\s*([-+]?[0-9]*\.?[0-9]+)\s*\*\s*([A-Za-z]+[A-Za-z0-9_]*))'
        #Iterate over matches
        matches = 0
        for c,xc in re.findall(regex,name):
            matches += 1
            coefficients.append(float(c))
            xc_names.append(xc)
        if matches == 0:
            raise NameError('ParametrizedKernel has bad arguments: "%s".' % pxc_kernel)
        else:
            #Assumption here is that the arguments contain only LDA and GGA functionals
            GGA = False
            for xc in xc_names:
                if xc.startswith('GGA'):
                    GGA = True
                    break
            self.type = 'GGA' if GGA else 'LDA'
            for kernel in xc_names:
                self.kernels.append(LibXC(kernel))
            
            self.coefs = coefficients

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        e_g[:] = 0.0
        e_g_tmp = np.empty_like(e_g)
        dedn_sg_tmp = np.zeros_like(dedn_sg)
        dedsigma_xg_tmp = dedsigma_xg
        #GGA
        if self.type == 'GGA':
            dedsigma_xg[:] = 0.0
            dedsigma_xg_tmp = np.empty_like(dedsigma_xg)
        else:
            dedsigma_xg_tmp = None
        for kernel, c in zip(self.kernels, self.coefs):
            kernel.calculate(e_g_tmp, n_sg, dedn_sg_tmp,
                            sigma_xg, dedsigma_xg_tmp,
                            tau_sg, dedtau_sg)
            e_g += c*e_g_tmp
            dedn_sg += c*dedn_sg_tmp
            dedn_sg_tmp[:] = 0.0
            e_g_tmp[:] = 0.0
            if self.type == 'GGA':
                dedsigma_xg += c*dedsigma_xg_tmp
                dedsigma_xg_tmp[:] = 0.0

