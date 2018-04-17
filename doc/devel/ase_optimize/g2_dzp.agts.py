from q2.job import Job


def workflow():
    return [
        Job('g2_dzp.py 0@4x13m'),
        Job('g2_dzp.py 1@4x13m'),
        Job('g2_dzp.py 2@4x13m'),
        Job('g2_dzp.py 3@4x13m'),
        Job('g2_dzp.py 4@4x13m'),
        Job('g2_dzp.py 5@4x13m'),
        Job('g2_dzp.py 6@4x13m'),
        Job('g2_dzp.py 7@4x13m'),
        Job('g2_dzp.py 8@4x13m'),
        Job('g2_dzp.py 9@4x13m'),
        Job('g2_dzp_csv.py', deps=['g2_dzp.py 0', 'g2_dzp.py 1', 'g2_dzp.py 2', 'g2_dzp.py 3', 'g2_dzp.py 4', 'g2_dzp.py 5', 'g2_dzp.py 6', 'g2_dzp.py 7', 'g2_dzp.py 8', 'g2_dzp.py 9'])]

def agts(queue):
    jobs = [queue.add('g2_dzp.py ' + str(i), ncpus=4, walltime=800)
            for i in range(10)]
    queue.add('g2_dzp_csv.py', deps=jobs, creates='g2_dzp.csv')
