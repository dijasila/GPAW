from q2.job import Job


def workflow():
    return [
        Job('gs_Si.py@4x20s'),
        Job('eps_Si.py@4x4m', deps=['gs_Si.py']),
        Job('plot_Si.py@1x10s', deps=['eps_Si.py']),
        Job('gs_MoS2.py@4x1m'),
        Job('pol_MoS2.py@64x33m', deps=['gs_MoS2.py']),
        Job('plot_MoS2.py@1x10s', deps=['pol_MoS2.py']),
        Job('get_2d_eps.py@1x8m', deps=['gs_MoS2.py']),
        Job('plot_2d_eps.py@1x10s', deps=['get_2d_eps.py']),
        Job('alpha_MoS2.py@1x10s', deps=['gs_MoS2.py'])]

def agts(queue):
    gs_si = queue.add('gs_Si.py', ncpus=4, walltime=20)
    bse_si = queue.add('eps_Si.py', ncpus=4, walltime=240, deps=gs_si)
    queue.add('plot_Si.py', ncpus=1, walltime=10, deps=bse_si,
              creates='bse_Si.png')

    gs_mos2 = queue.add('gs_MoS2.py', ncpus=4, walltime=100)
    bse_mos2 = queue.add('pol_MoS2.py', ncpus=64, walltime=2000, deps=gs_mos2)
    queue.add('plot_MoS2.py', ncpus=1, walltime=10, deps=bse_mos2,
              creates='bse_MoS2.png')

    eps = queue.add('get_2d_eps.py', ncpus=1, walltime=500, deps=gs_mos2)
    queue.add('plot_2d_eps.py', ncpus=1, walltime=10, deps=eps,
              creates='2d_eps.png')

    queue.add('alpha_MoS2.py', ncpus=1, walltime=10, deps=gs_mos2)
