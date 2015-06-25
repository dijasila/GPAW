def agts(queue):
    basis = queue.add('lcaotddft_basis.py', ncpus=1, walltime=1)
    ag55 = queue.add('lcaotddft_ag55.py', deps=[basis], ncpus=64, walltime=120)
    queue.add('lcaotddft_fig1.py', deps=[ag55], creates='fig1.png')



