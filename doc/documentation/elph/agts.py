from myqueue.workflow import run


def workflow():
    el = run(script='elph.py', cores=8)
    ph = run(script='phonon.py', cores=8)
    with el, ph:
        run(script='construct_matrix.py')
