def workflow():
    from myqueue.workflow import run
    run(script='CO.py', cores=8, tmax='15m')
