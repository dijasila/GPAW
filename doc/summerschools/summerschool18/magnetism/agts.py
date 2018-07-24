from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task(nbrun, args=['magnetism1.master.ipynb'], tmax='13h'),
        task(nbrun, args=['magnetism2.master.ipynb'], tmax='13h'),
        task(nbrun, args=['magnetism3.ipynb'], tmax='13h')]
