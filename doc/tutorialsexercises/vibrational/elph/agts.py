from myqueue.workflow import run


def workflow():
    el = run(script='effective_potential.py', cores=8, tmax='80m')
    with el:
        sc = run(script='supercell.py', cores=8, tmax='5m')
    with el, sc:
        run(script='gmatrix.py')
