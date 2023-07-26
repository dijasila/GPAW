from itertools import product

import pytest
import numpy as np

from gpaw.response.integrators import (TetrahedronIntegrator, Integrand,
                                       HilbertTetrahedron)
from gpaw.response.frequencies import FrequencyGridDescriptor

from gpaw.response import ResponseContext


class MyIntegrand(Integrand):
    def matrix_element(self, x_c):
        return np.array([[1.]], complex)

    def eigenvalues(self, x_c):
        return np.array([(x_c**2).sum()**0.5], float)


@pytest.mark.response
def test_tetrahedron_integrator():
    cell_cv = np.eye(3)
    context = ResponseContext()
    integrator = TetrahedronIntegrator(cell_cv, context, nblocks=1)
    x_g = np.linspace(-1, 1, 30)
    x_gc = np.array([comb for comb in product(*([x_g] * 3))])

    domain = (x_gc,)
    out_wxx = np.zeros((1, 1, 1), complex)
    integrator.integrate(task=HilbertTetrahedron(integrator.blockcomm),
                         domain=domain,
                         integrand=MyIntegrand(),
                         wd=FrequencyGridDescriptor([-1.0]),
                         out_wxx=out_wxx)

    assert abs(out_wxx[0, 0, 0] - 4 * np.pi) < 1e-2
    # equal(out_wxx[0, 0, 0], 4 * np.pi, 1e-2,
    #       msg='Integrated area of unit sphere is not 4 * pi')
