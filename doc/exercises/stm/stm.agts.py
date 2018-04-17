from q2.job import Job


def workflow():
    return [
        Job('HAl100.py'),
        Job('stm.agts.py', deps=['HAl100.py'])]

def agts(queue):
    job = queue.add('HAl100.py')
    queue.add('stm.agts.py', ncpus=1, deps=[job])

if __name__ == '__main__':
    import sys
    sys.argv = ['', 'HAl100.gpw']
    exec(open('stm.py').read())
