from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task(nbrun, args=['batteries1.master.ipynb'], tmax='1h'),
        task(nbrun, args=['batteries2.master.ipynb'], tmax='3h'),
        task(nbrun, args=['batteries3.master.ipynb'], tmax='1h', cores=8,
             deps=nbrun + '+batteries1.master.ipynb')]
