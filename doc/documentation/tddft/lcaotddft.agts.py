def agts(queue):
    basis = queue.add('lcaotddft_basis.py', ncpus=1, walltime=1)
    queue.add('lcaotddft_ag55.py', deps=[basis], ncpus=64, walltime=1)


