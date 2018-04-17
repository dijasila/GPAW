from q2.job import Job


def workflow():
    return [
        Job('adenine-thymine_complex_stack.py@4x2m')]

def agts(queue):
    queue.add('adenine-thymine_complex_stack.py', ncpus=4, walltime=2 * 60)
