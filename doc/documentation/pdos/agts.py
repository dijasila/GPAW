from myqueue.job import Job


def workflow():
    return [
        task('top.py@8:15m'),
        task('pdos.py', deps=['top.py']),
        task('lcaodos_gs.py@8:15m'),
        task('lcaodos_plt.py', deps=['lcaodos_gs.py'])]
