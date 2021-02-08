def workflow():
    from myqueue.workflow import run
    run(script='h2o.py', folder='water')
    run(script='CO.py', cores=8, tmax='15m', folder='wavefunctions')
    run(script='ag.py', folder='band_structure')
    run(script='test.py', folder='eels', deps='../band_structure/ag.py')
    run(script='test.py', folder='gw')
    run(script='con_pw.py', folder='stress')
    run(script='stress.py', folder='stress')
