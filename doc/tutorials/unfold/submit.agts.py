from q2.job import Job


def workflow():
    return [
        Job('gs_3x3_defect.py@16x5s'),
        Job('unfold_3x3_defect.py@16x10s', deps=['gs_3x3_defect.py']),
        Job('plot_sf.py', deps=['unfold_3x3_defect.py'])]

def agts(queue):
    gs = queue.add('gs_3x3_defect.py', ncpus=16, walltime=5)
    unfolding = queue.add('unfold_3x3_defect.py', deps=gs, ncpus=16,
                          walltime=10)
    queue.add('plot_sf.py', deps=unfolding, creates='sf_3x3_defect.png')
