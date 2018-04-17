from q2.job import Job


def workflow():
    return [
        Job('lcaotddft_basis.py@1x10s'),
        Job('lcaotddft_ag55.py@48x1m', deps=['lcaotddft_basis.py']),
        Job('lcaotddft_fig1.py', deps=['lcaotddft_ag55.py']),
        Job('lcaotddft.py@4x40s')]

def agts(queue):
    basis = queue.add('lcaotddft_basis.py', ncpus=1, walltime=10)
    ag55 = queue.add('lcaotddft_ag55.py', deps=[basis], ncpus=48, walltime=100)
    queue.add('lcaotddft_fig1.py', deps=[ag55], creates='fig1.png')
    queue.add('lcaotddft.py', ncpus=4, walltime=40)
