from myqueue.task import task


def create_tasks():
    return []  # disable for now
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task(nbrun, args=['pv1.master.ipynb'], tmax='13h'),
        task(nbrun, args=['pv2.master.ipynb'], tmax='13h'),
        task(nbrun, args=['pv3.master.ipynb'], tmax='13h')]
