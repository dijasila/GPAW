def workflow():
    from myqueue.job import Job
    return [
        Job('HAl100.py'),
        Job('stm.agts.py', deps=['HAl100.py'])]


if __name__ == '__main__':
    import sys
    sys.argv = ['', 'HAl100.gpw']
    exec(open('stm.py').read())
