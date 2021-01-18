from typing import Tuple

import numpy as np
import pytest

import gpaw.mpi as mpi
from gpaw import setup_paths
from gpaw.atom.configurations import parameters, tf_parameters
from gpaw.atom.generator import Generator
from gpaw.hints import Array1D


def equal(x, y, tolerance=0):
    assert x == pytest.approx(y, abs=tolerance)


def print_reference(data_i, name='ref_i', fmt='%.12le'):
    if mpi.world.rank == 0:
        print('%s = [' % name, end='')
        for i, val in enumerate(data_i):
            if i > 0:
                print('', end='\n')
                print(' ' * (len(name) + 4), end='')
            print(fmt % val, end='')
            print(',', end='')
        print('\b]')


def findpeak(x: Array1D, y: Array1D) -> Tuple[float, float]:
    """Find peak.

    >>> x = np.linspace(1, 5, 10)
    >>> y = 1 - (x - np.pi)**2
    >>> x0, y0 = findpeak(x, y)
    >>> f'x0={x0:.6f}, y0={y0:.6f}'
    'x0=3.141593, y0=1.000000'
    """
    i = y.argmax()
    a, b, c = np.polyfit(x[i - 1:i + 2] - x[i], y[i - 1:i + 2], 2)
    assert a < 0
    dx = -0.5 * b / a
    x0 = x[i] + dx
    return x0, a * dx**2 + b * dx + c


def gen(symbol, exx=False, name=None, yukawa_gamma=None,
        write_xml=False, **kwargs):
    setup = None
    if mpi.rank == 0:
        if 'scalarrel' not in kwargs:
            kwargs['scalarrel'] = True
        g = Generator(symbol, **kwargs)
        if 'orbital_free' in kwargs:
            setup = g.run(exx=exx, name=name, yukawa_gamma=yukawa_gamma,
                          use_restart_file=False,
                          write_xml=write_xml,
                          **tf_parameters.get(symbol, {'rcut': 0.9}))
        else:
            setup = g.run(exx=exx, name=name, yukawa_gamma=yukawa_gamma,
                          use_restart_file=False,
                          write_xml=write_xml,
                          **parameters[symbol])
    setup = mpi.broadcast(setup, 0)
    if setup_paths[0] != '.':
        setup_paths.insert(0, '.')
    return setup
