from myqueue.workflow import run


def workflow():
    return [
        task('CO.py@1:10s'),
        task('CO2cube.py@1:10s', deps='CO.py')]
