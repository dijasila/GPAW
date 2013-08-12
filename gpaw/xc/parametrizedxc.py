from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc import LibXC
import numpy as np
import copy
import _gpaw

class ParametrizedKernel(XCKernel):
	def __init__(self, name, type, coefs, kernels):
		self.name = name
		self.type = type
		#self.functionals = ['LDA_X','LDA_K_TF', 'GGA_K_VW']
		self.kernels = kernels
		self.coefs = coefs

	def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
		#e_g[:] = 0.0
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

