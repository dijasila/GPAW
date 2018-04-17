def workflow():
    from q2.job import Job
    return [
        Job('surface.agts.py'),
        Job('work_function.py', deps=['surface.agts.py'])]


if __name__ == '__main__':
    exec(open('Al100.py').read(), {'k': 6, 'N': 5})
