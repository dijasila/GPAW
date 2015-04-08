def agts(queue):
    queue.add('pbe_abinit_hgh Al', ncpus=1,
              queueopts='-l nodes=1:ppn=16:xeon16', walltime=15*60)
