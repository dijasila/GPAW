from myqueue.job import Job


def workflow():
    return [
        task('basisgeneration.py@1:10m'),
        task('lcao_h2o.py@1:10m'),
        task('lcao_opt.py@1:10m')]
