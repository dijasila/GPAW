def agts(queue):
    queue.add('gaps.py')
    eos = queue.add('eos.py', ncpus=4, walltime=600)
    queue.add('plot_a.py', deps=eos, creates='a.png')
