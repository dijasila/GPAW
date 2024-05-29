def workflow():
    from myqueue.workflow import run
    with run(script='atom/si.atom.pbe+exx.py', cores=24, tmax='20m'):
        if 0:
            # Something wrong with this one (#1186):
            run(script='atom/test_pbe_isolated_output.py')
        with run(script='atom/si.atom.rpa_init_pbe.py', cores=24, tmax='15m'):
            run(script='atom/si.atom.rpa.py',
                cores=40, nodename='xeon40_768', tmax='15h')
    with run(script='si.pbe.py'):
        exx = run(script='si_pbe_exx.py', cores=4, tmax='15m')
        run(script='test_pbe_output.py', deps=[exx])
    with run(script='si.rpa_init_pbe.py'):
        run(script='si.rpa.py', cores=4, tmax='15m')
