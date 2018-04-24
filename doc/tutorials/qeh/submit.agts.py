from myqueue.job import Job


def workflow():
    return [
        Job('gs_MoS2.py@16x25m'),
        Job('gs_WSe2.py@16x25m'),
        Job('bb_MoS2.py@16x20h', deps=['gs_MoS2.py']),
        Job('bb_WSe2.py@16x20h', deps=['gs_WSe2.py']),
        Job('interpolate_bb.py', deps=['bb_MoS2.py', 'bb_WSe2.py']),
        Job('interlayer.py', deps=['interpolate_bb.py'])]
