from myqueue.workflow import run


def workflow():
    with run(script='top.py', cores=8, tmax='15m'):
        run(script='pdos.py')
    with run(script='lcaodos_gs.py', cores=8, tmax='15m'):
        run(script='lcaodos_plt.py')
