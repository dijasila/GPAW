from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task('xyz.py'),
        task(nbrun, args=['magnetism1.master.ipynb'],
             tmax='1h', deps='xyz.py'),
        task(nbrun, args=['magnetism2.master.ipynb'],
             tmax='2h', deps='xyz.py')]
