def workflow():
    from myqueue.workflow import run
    run(script='initial.py', cores=2, tmax='15m')
    with run(script='solution.py', cores=2, tmax='15m'):
        run(script='densitydiff.py')
