from q2.job import Job


def workflow():
    return [
        Job('Na2TDDFT.py@2x1m'),
        Job('part2.py', deps=['Na2TDDFT.py']),
        Job('ground_state.py@8x15s'),
        Job('spectrum.py', deps=['ground_state.py'])]

def agts(queue):
    calc1 = queue.add('Na2TDDFT.py',
                      ncpus=2,
                      walltime=60)
    queue.add('part2.py', deps=calc1)
    gs = queue.add('ground_state.py', ncpus=8)
    queue.add('spectrum.py', deps=gs)
