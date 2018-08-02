from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task(nbrun, args=['batteries1.master.ipynb'], tmax='13h'),
        task(nbrun, args=['batteries2.master.ipynb'], tmax='13h',
             deps=nbrun + '+batteries1.master.ipynb')
        task(nbrun, args=['batteries3.master.ipynb'], tmax='13h',
             deps=nbrun + '+batteries2.master.ipynb')]
