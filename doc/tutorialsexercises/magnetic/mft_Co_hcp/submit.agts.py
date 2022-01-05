from myqueue.workflow import run


def workflow():
    with run(script='converge_gs.py', cores=16, tmax='2h'):
        with run(script='high_sym_pts.py', cores=16, tmax='2h'):
            run(script='magnon_energy_plot.py')
        with run(script='high_sym_path.py', cores=16, tmax='4h'):
            run(script='magnon_dispersion_plot.py')