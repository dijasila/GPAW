#!/usr/bin/env python

import os
import numpy as np

from scipy.optimize import leastsq
from ase.io.trajectory import PickleTrajectory

name = 'na2_osc'

# Import relevant test and make sure it has the prerequisite parameters
mod = __import__(name, {}, {})
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
def f(p, t):
    return p[0] * np.cos(p[1] * (t - p[2])) + p[3]

def Df(p, t):
    return np.array([
        np.cos(p[1] * t),
        -p[0] * np.sin(p[1] * (t - p[2])) * (t - p[2]),
        p[0] * np.sin(p[1] * (t - p[2])) * p[1],
        np.ones_like(t)])

data = np.vstack((t,d)).T.copy()
params0 = (mod.d_disp, 2*np.pi/mod.period, -mod.timestep*mod.ndiv, mod.d_bond)
params,cov,info,msg,status = leastsq(lambda params: f(params, t)-d, params0,
    full_output=True, Dfun=lambda params: Df(params,t), col_deriv=True)
print 'leastsq returned %d: %s' % (status, msg.replace('\n ',''))
print 'params0:', np.asarray(params0)
print 'params :', params
assert status in [1,2,3,4], (params, cov, info, msg, status)

print 'T=%13.9f fs, Tref=%13.9f fs, err=%5.2f %%' % (2*np.pi/params[1]*1e-3,
    mod.period*1e-3, 1e2*np.abs(2*np.pi/params[1]-mod.period)/mod.period)

# -------------------------------------------------------------------

import pylab as pl
ax = pl.axes()
ax.plot(t,d,'-b',t,f(params,t),'--k')
pl.show()

