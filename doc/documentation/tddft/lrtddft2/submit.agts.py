from myqueue.workflow import run


def workflow():
    with run(script='gs.py', cores=8, tmax='5m'):
        with run(script='unocc.py', cores=8, tmax='30m'):
            with run(script='lr2.py', cores=8, tmax='5m'):
                with run(script='lr2_restart.py', cores=8, tmax='5m'):
                    with run(script='lr2_analyze.py', cores=1, tmax='5m'):
                        run(script='plot_spec.py', tmax='1m')
                        run(script='check_consistency_with_docs.py', tmax='1m')
