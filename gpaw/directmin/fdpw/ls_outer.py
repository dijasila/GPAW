"""
Class for finding an optimal
step alpha length in the optimization problem:
x = x + alpha * p.
These classes are for wave-functions
represented on real space grid or by plane waves
"""


import numpy as np


class UnitStepLength:

    def __init__(self, evaluate_phi_and_der_phi, maxstep=0.25, **kwargs):
        """
        :param evaluate_phi_and_der_phi: function which calculate
        phi, der_phi and g for given x_k, p_s and alpha
        """

        self.evaluate_phi_and_der_phi = evaluate_phi_and_der_phi
        self.maxstep = maxstep

        # self.log = log

    def step_length_update(self, x_k, p_k, *args, **kwargs):

        wfs = kwargs['wfs']
        dot = 0.0
        for kpt in wfs.kpt_u:
            k = wfs.kd.nibzkpts * kpt.s + kpt.q
            for p in p_k[k]:
                dot += wfs.integrate(p, p, False)
        dot = dot.real
        dot = wfs.world.sum(dot)
        dot = np.sqrt(dot)

        if dot > self.maxstep:
            a_star = self.maxstep / dot
        else:
            a_star = 1.0

        phi_star, der_phi_star, g_star =\
            self.evaluate_phi_and_der_phi(
                x_k, p_k, a_star, *args)

        return a_star, phi_star, der_phi_star, g_star
