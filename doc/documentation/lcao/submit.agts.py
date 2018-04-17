from q2.job import Job


def workflow():
    return [
        Job('basisgeneration.py@1x10s'),
        Job('lcao_h2o.py@1x10s'),
        Job('lcao_opt.py@1x10s')]

def agts(queue):
    queue.add('basisgeneration.py', ncpus=1, walltime=10)
    queue.add('lcao_h2o.py', ncpus=1, walltime=10)
    queue.add('lcao_opt.py', ncpus=1, walltime=10)

