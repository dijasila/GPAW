def agts(queue):
    queue.add('lcao_dos.py', ncpus=1, walltime=15,
              creates=['dos_GaAs.png'])
    
