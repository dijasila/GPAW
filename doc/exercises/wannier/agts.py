def workflow():
    from myqueue.workflow import run
    with run(script='si.py', cores=8, tmax='15m'):
        run(script='wannier-si.py')
    with run(script='benzene.py', cores=8, tmax='15m'):
        run(script='wannier-benzene.py')
