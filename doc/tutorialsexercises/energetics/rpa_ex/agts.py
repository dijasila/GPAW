def workflow():
    from myqueue.workflow import run
    with run(script='si.pbe.py'):
        run(script='si_pbe_exx.py', cores=4, tmax='15m')
        run(script='test_pbe_output.py', cores=1, tmax='5m')
    with run(script='si.rpa_init_pbe.py'):
        run(script='si.rpa.py', cores=4, tmax='15m')
