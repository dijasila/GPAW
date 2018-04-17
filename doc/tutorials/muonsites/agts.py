from q2.job import Job


def workflow():
    return [
        Job('mnsi.py'),
        Job('plot2d.py', deps=['mnsi.py'])]

def agts(queue):
    mnsi = queue.add('mnsi.py')
    queue.add('plot2d.py', deps=mnsi, creates='pot_contour.png')
