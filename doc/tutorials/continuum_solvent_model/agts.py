from myqueue.task import task


def workflow():
    return [
        task('ethanol_in_water.py@4:10m'),
        task('check.py', deps=['ethanol_in_water.py'])]
