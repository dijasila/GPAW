def workflow():
    from myqueue.workflow import run
    run(script='run_test_sjm.py', cores=4, tmax='15h')
