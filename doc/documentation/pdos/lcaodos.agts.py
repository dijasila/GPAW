def agts(queue):
    calc = queue.add('lcaodos_gs.py', ncpus=8)
    queue.add('lcaodos_plt.py', deps=calc)
