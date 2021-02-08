def workflow():
    from myqueue.workflow import run
    with run(script='h2o.py', cores=8, tmax='15m'):
        with run(script='H2O_vib.py', cores=8, tmax='15m'):
            run(script='H2O_vib_2.py', cores=4, tmax='15m')
