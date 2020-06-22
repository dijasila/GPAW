from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task('split_db.py'),
        task(nbrun, args=['machinelearning.master.ipynb'],
             tmax='8h', deps='split_db.py')]
