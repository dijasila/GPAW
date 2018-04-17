from q2.job import Job


def workflow():
    return [
        Job('basisgeneration.py@1x10s'),
        Job('lcao_h2o.py@1x10s'),
        Job('lcao_opt.py@1x10s')]
