from q2.job import Job


def workflow():
    return [
        Job('plot_freq.py'),
        Job('silicon_ABS_simpleversion.py'),
        Job('plot_silicon_ABS_simple.py', deps=['silicon_ABS_simpleversion.py']),
        Job('silicon_ABS.py@16x1m'),
        Job('plot_ABS.py', deps=['silicon_ABS.py']),
        Job('aluminum_EELS.py@8x1m'),
        Job('plot_aluminum_EELS_simple.py', deps=['aluminum_EELS.py']),
        Job('graphite_EELS.py@8x1m'),
        Job('plot_EELS.py', deps=['graphite_EELS.py']),
        Job('tas2_dielectric_function.py@8x15s'),
        Job('graphene_dielectric_function.py@8x15s')]

def agts(queue):
    queue.add('plot_freq.py', creates='nl_freq_grid.png')

    simple_si = queue.add('silicon_ABS_simpleversion.py')
    queue.add('plot_silicon_ABS_simple.py', creates='si_abs.png',
              deps=simple_si)

    si = queue.add('silicon_ABS.py', creates='mac_eps.csv',
                   ncpus=16, walltime=100)
    queue.add('plot_ABS.py', deps=si, creates='silicon_ABS.png')

    al = queue.add('aluminum_EELS.py', ncpus=8, walltime=100)
    queue.add('plot_aluminum_EELS_simple.py', deps=al,
              creates=['aluminum_EELS.png'])

    GR = queue.add('graphite_EELS.py', ncpus=8, walltime=100)
    queue.add('plot_EELS.py', deps=GR, creates='graphite_EELS.png')

    queue.add('tas2_dielectric_function.py', ncpus=8, creates='tas2_eps.png')

    queue.add('graphene_dielectric_function.py', ncpus=8,
              creates='graphene_eps.png')
