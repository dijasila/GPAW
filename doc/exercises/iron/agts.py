def workflow():
    from myqueue.workflow import run
    runs = [run(script='ferro.py', cores=4, tmax='15m'),
            run(script='anti.py', cores=4, tmax='15m'),
            run(script='non.py', cores=2, tmax='15m')]
    run(script='PBE.py', folder='iron', deps=runs)
