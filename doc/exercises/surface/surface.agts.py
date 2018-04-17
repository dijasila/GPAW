from q2.job import Job


def workflow():
    return [
        Job('surface.agts.py'),
        Job('work_function.py', deps=['surface.agts.py'])]

def agts(queue):
    al = queue.add('surface.agts.py')
    queue.add('work_function.py', ncpus=1, deps=[al])

if __name__ == '__main__':
    exec(open('Al100.py').read(), {'k': 6, 'N': 5})
