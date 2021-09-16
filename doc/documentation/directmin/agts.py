from myqueue.workflow import run


def workflow():
    run(script='h2o.py')
    r1 = run(script='run_g2_with_dm_ui.py', cores=8, tmax='2h')
    r2 = run(script='run_g2_with_scf.py', cores=8, tmax='2h')
    with r1, r2:
        run(script='plot_g2.py')
    r3 = run(script='wm_dmscf.py', cores=8, tmax='30m')
    with r3:
        run(script='plot_h2o.py')
