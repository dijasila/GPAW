def agts(queue):
    queue.add('pbe_abinit_paw Au', ncpus=1,
              queueopts='-l nodes=1:ppn=16:xeon16', walltime=40*60)
