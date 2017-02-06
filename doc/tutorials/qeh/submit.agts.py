def agts(queue):
    gs_MoS2 = queue.add('gs_MoS2.py', ncpus=16, walltime=25)
    gs_WSe2 = queue.add('gs_WSe2.py', ncpus=16, walltime=25)

    bb_MoS2 = queue.add('bb_MoS2.py', deps=gs_MoS2, ncpus=16,
                        walltime=1200)
    bb_WSe2 = queue.add('bb_WSe2.py', deps=gs_WSe2, ncpus=16,
                        walltime=1200)

    interp = queue.add('interpolate.py', deps=[bb_MoS2, bb_WSe2], ncpus=1,
                        walltime=10)

    queue.add('interlayer.py', deps=interp,
              creates='W_r.svg')
