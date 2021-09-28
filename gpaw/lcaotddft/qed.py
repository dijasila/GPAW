import numpy as np

from gpaw.mpi import world
from ase.units import Bohr

rr_quantization_plane = 0
polarization_cavity   = [0,0,0]

def create_qed(name, **kwargs):
    if  name == 'RRemission':
        return RRemission(**kwargs)
    else:
        raise ValueError('Unknown qed ansatz: %s' % name)

class RRemission(object):
    r"""
    # Radiation-reaction potential accoridng to Schaefer et al. [arXiv 2109.09839]
    # The potential accounts for the friction forces acting on the radiating system
    # of oscillating charges emitting into a single dimension. A more elegant 
    # formulation would use the current instead of the dipole. 
    # Please contact christian.schaefer.physics@gmail.com if any problems 
    # should appear or you would like to consider more complex emission.
    # Big thanks to Tuomas Rossi and Jakub Fojt for their help.

    Parameters
    ----------
    rr_quantization_plane: float
        value of :math:`rr_quantization_plane` in atomic units
    pol_cavity: array
        value of :math:`pol_cavity` dimensionless (directional)
    """

    def __init__(self, rr_quantization_plane_in, pol_cavity_in):
        global rr_quantization_plane 
        global polarization_cavity 
        rr_quantization_plane = rr_quantization_plane_in / Bohr**2
        polarization_cavity = pol_cavity_in
