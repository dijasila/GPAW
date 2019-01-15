from myqueue.task import task


def create_tasks():
    return [
        task('C2-test.py', cores=8),
        task('C2v-test.py', cores=8),
        task('C3v-test.py', cores=8),
        task('D2d-test.py', cores=8),
        task('D3h-test.py', cores=8),
        task('D5-test.py', cores=8),
        task('D5h-test.py', cores=8),
        task('I-test.py', cores=8),
        task('Ih-test.py', cores=8),
        task('Oh-test.py', cores=8),
        task('Td-test.py', cores=8),
        task('Th-test.py', cores=8) ]
