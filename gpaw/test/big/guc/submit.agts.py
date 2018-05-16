from myqueue.task import task


def workflow():
    return [task('graphene.py@8:15m')]
