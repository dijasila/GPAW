from q2.job import Job


def workflow():
    return [
        Job('top.py@8x15s'),
        Job('pdos.py', deps=['top.py']),
        Job('lcaodos_gs.py@8x15s'),
        Job('lcaodos_plt.py', deps=['lcaodos_gs.py'])]
