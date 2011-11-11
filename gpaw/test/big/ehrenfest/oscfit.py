#!/usr/bin/env python

import os
import numpy as np

from Scientific.Functions.LeastSquares import leastSquaresFit
from ase.io.trajectory import PickleTrajectory

name = 'na2_osc'

# Import relevant test and make sure it has the prerequisite parameters
mod = __import__(name, globals={})
for attr in ['d_bond', 'd_disp', 'timestep', 'period', 'ndiv', 'niter']:
    if not hasattr(mod, attr):
        raise KeyError('Module %s has no %s value' % (name, attr))

traj = PickleTrajectory(name + '_td.traj', 'r')
nframes = len(traj)

t = np.empty(nframes)
d = np.empty(nframes)
for i in range(nframes):
    frame = traj[i]
    t[i] = mod.timestep * mod.ndiv * i
    pos_av = frame.get_positions()
    d[i] = np.sum((pos_av[1]-pos_av[0])**2)**0.5

# Least-squares fit model during equilibration
def targetfunc(params, t):
    return params[0] * np.cos(params[1] * t) + params[2]

data = np.vstack((t,d)).T.copy()
params0 = (mod.d_disp, 2*np.pi/mod.period, mod.d_bond)
params,chisq = leastSquaresFit(targetfunc, params0, data, stopping_limit=1e-9)
print params,chisq

print 'T=%13.9f fs' % (2*np.pi/params[1]*1e-3)

# -------------------------------------------------------------------

import pylab as pl
ax = pl.axes()
ax.plot(t,d,'-b',t,targetfunc(params,t),'--k')
pl.show()

