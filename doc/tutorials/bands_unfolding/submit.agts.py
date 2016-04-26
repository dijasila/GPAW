def agts(queue):
    gs = queue.add('gs_3x3_defect.py', ncpus=16, walltime=5)
    unfolding = queue.add('unfold_3x3_defect.py', deps=gs, ncpus=16,
                          walltime=10, creates='sf.png')

