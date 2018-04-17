from q2.job import Job


def workflow():
    return [
        Job('gs_MoS2.py@16x25s'),
        Job('gs_WSe2.py@16x25s'),
        Job('bb_MoS2.py@16x20m', deps=['gs_MoS2.py']),
        Job('bb_WSe2.py@16x20m', deps=['gs_WSe2.py']),
        Job('interpolate_bb.py', deps=['bb_MoS2.py', 'bb_WSe2.py']),
        Job('interlayer.py', deps=['interpolate_bb.py'])]

def agts(queue):
    gs_MoS2 = queue.add('gs_MoS2.py', ncpus=16, walltime=25)
    gs_WSe2 = queue.add('gs_WSe2.py', ncpus=16, walltime=25)

    bb_MoS2 = queue.add('bb_MoS2.py', deps=gs_MoS2, ncpus=16,
                        walltime=1200)
    bb_WSe2 = queue.add('bb_WSe2.py', deps=gs_WSe2, ncpus=16,
                        walltime=1200)

    interp = queue.add('interpolate_bb.py', deps=[bb_MoS2, bb_WSe2])

    queue.add('interlayer.py', deps=interp, creates='W_r.svg')
