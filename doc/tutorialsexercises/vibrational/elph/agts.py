from myqueue.workflow import run


def workflow():
    el = run(script='effective_potential.py', cores=16, tmax='4h')
    scf = run(script='scf.py', cores=16, tmax='10m')
    with el:
        sc = run(script='supercell.py', cores=4, tmax='1h')
    with el, sc, scf:
        run(script='gmatrix.py')
