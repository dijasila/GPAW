#!/usr/bin/env python

import numpy as np

try:
    # Matplotlib is not a dependency
    import matplotlib as mpl
    mpl.use('Agg')  # force the antigrain backend
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except (ImportError, RuntimeError):
    mpl = None

from scipy.optimize import leastsq
from ase.io.trajectory import PickleTrajectory

# Dimer oscillation model used for least-squares fit 
def f(p, t):
    return p[0] * np.cos(p[1] * (t - p[2])) + p[3]

# Jacobian of model with respect to its four parameters
def Df(p, t):
    return np.array([
        np.cos(p[1] * t),
        -p[0] * np.sin(p[1] * (t - p[2])) * (t - p[2]),
        p[0] * np.sin(p[1] * (t - p[2])) * p[1],
        np.ones_like(t)])


for name in ['h2_osc', 'n2_osc', 'na2_osc']:
    print '\nAnalysing %s\n%s' % (name, '-'*32)

    # Import relevant test and make sure it has the prerequisite parameters
    m = __import__(name, {}, {})
    for attr in ['d_bond', 'd_disp', 'timestep', 'period', 'ndiv', 'niter']:
        if not hasattr(m, attr):
            raise ImportError('Module %s has no %s value' % (name, attr))

    # Read dimer bond length time series from trajectory file
    traj = PickleTrajectory(name + '_td.traj', 'r')
    symbol = traj[0].get_name()
    nframes = len(traj)
    t = np.empty(nframes)
    d = np.empty(nframes)
    for i in range(nframes):
        pos_av = traj[i].get_positions()
        t[i] = m.timestep * m.ndiv * i
        d[i] = np.sum((pos_av[1] - pos_av[0])**2)**0.5
    print 'Read %d frames from trajectory...' % nframes
    #assert nframes * ndiv == niter, (nframes, ndiv, niter) #TODO uncomment
    traj.close()

    # Fit model to time series using imported parameters as an initial guess
    p0 = (m.d_disp, 2 * np.pi / m.period, -m.timestep * m.ndiv, m.d_bond)
    p, cov, info, msg, status = leastsq(lambda p: f(p, t) - d, p0, \
        Dfun=lambda p: Df(p,t), col_deriv=True, full_output=True)
    print 'leastsq returned %d: %s' % (status, msg.replace('\n ',''))
    print 'p0=', np.asarray(p0)
    print 'p =', p
    assert status in range(1,4+1), (p, cov, info, msg, status)

    tol = 0.1 #TODO use m.reltol
    err = np.abs(2 * np.pi / p[1] - m.period) / m.period
    print 'T=%13.9f fs, Tref=%13.9f fs, err=%5.2f %%, tol=%.1f %%' \
        % (2 * np.pi / p[1] * 1e-3, m.period * 1e-3, 1e2 * err, 1e2 * tol)

    if mpl:
        fig = Figure()
        ax = fig.add_axes([0.1, 0.1, 0.87, 0.83])
        raw = r',\;'.join([r'T=%.2f\mathrm{\,fs}',
                           r'T_\mathrm{ref}=%.2f\mathrm{\,fs}',
                           r'\eta=%.2f\,\%%'])
        mathmode = raw % (2 * np.pi / p[1] * 1e-3, m.period * 1e-3, 1e2 * err)
        ax.set_title(symbol + ' ($' + mathmode + '$)')
        ax.set_xlabel('Time [fs]')
        ax.set_ylabel('Dimer bond length [Ang]')
        ax.plot(t * 1e-3, d, '-b', t * 1e-3, f(p,t), '--k')
        ax.legend(('Ehrenfest data', r'$A\,\mathrm{cos}(\omega(t-t_0))+B$'))
        FigureCanvasAgg(fig).print_figure(name + '.png', dpi=90)

    if err > tol:
        print 'Relative error %f %% > tolerance %f %%' % (1e2 * err, 1e2 * tol)
        raise SystemExit(1)

