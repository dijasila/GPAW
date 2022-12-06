import numpy as np

from gpaw.response import timer


class DysonSolver:
    """Class for invertion of Dyson-like equations."""

    def __init__(self, context):
        self.context = context

    @timer('Invert Dyson-like equation')
    def invert_dyson(self, chiks_wGG, Khxc_GG):
        """Invert the frequency dependent Dyson equation in plane wave basis:

        chi_wGG' = chiks_wGG' + chiks_wGG1 Khxc_G1G2 chi_wG2G'
        """
        self.context.print('Inverting Dyson-like equation')
        chi_wGG = np.empty_like(chiks_wGG)
        for w, chiks_GG in enumerate(chiks_wGG):
            chi_GG = invert_dyson_single_frequency(chiks_GG, Khxc_GG)

            chi_wGG[w] = chi_GG

        return chi_wGG


def invert_dyson_single_frequency(chiks_GG, Khxc_GG):
    """Invert the single frequency Dyson equation in plane wave basis:

    chi_GG' = chiks_GG' + chiks_GG1 Khxc_G1G2 chi_G2G'
    """
    enhancement_GG = np.linalg.inv(np.eye(len(chiks_GG)) -
                                   np.dot(chiks_GG, Khxc_GG))
    chi_GG = np.dot(enhancement_GG, chiks_GG)

    return chi_GG
