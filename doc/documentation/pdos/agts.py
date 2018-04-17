from q2.job import Job


def workflow():
    return [
        Job('top.py@8x15s'),
        Job('pdos.py', deps=['top.py']),
        Job('lcaodos_gs.py@8x15s'),
        Job('lcaodos_plt.py', deps=['lcaodos_gs.py'])]

def agts(queue):
    top = queue.add('top.py', ncpus=8)
    queue.add('pdos.py', deps=top, creates='pdos.png')

    calc = queue.add('lcaodos_gs.py', ncpus=8)
    queue.add('lcaodos_plt.py', deps=calc, creates='lcaodos.png')
