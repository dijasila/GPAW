from myqueue.workflow import run


def workflow():
    with run(script='gs.py', cores=8):
        ksd = run(script='ksd_init.py')
        with run(script='td.py', cores=8, tmax='30m'):
            with run(script='tdc.py', cores=8, tmax='30m'):
                run(script='td_replay.py', cores=8, tmax='30m')
                with run(script='spectrum.py') as spec:
                    run(script='spec_plot.py')
                with run(script='td_fdm_replay.py', cores=8):
                    with run(script='fdm_ind.py', cores=8):
                        run(script='ind_plot.py')
                    run(script='tcm_plot.py', deps=[ksd, spec])
        with run(script='td_pulse.py', cores=8, tmax='30m'):
            run(script='plot_pulse.py')
