def workflow():
    from myqueue.workflow import run
    run(script='run_sjm_test.py', cores=4, tmax='5h')
