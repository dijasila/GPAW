def workflow():
    from myqueue.workflow import run
    with run(script='CO.py', cores=8, tmax='15m'):
        run(script='../dos/dos.py', args=['CO.gpw'])
