from myqueue.job import Job


def workflow():
    return [
        Job('top.py@8x15m'),
        Job('pdos.py', deps=['top.py']),
        Job('lcaodos_gs.py@8x15m'),
        Job('lcaodos_plt.py', deps=['lcaodos_gs.py'])]
