from q2.job import Job


def workflow():
    return [
        Job('h2_osc.py@8x2m'),
        Job('n2_osc.py@40x15m'),
        Job('na2_md.py@8x2m'),
        Job('na2_osc.py@8x40m')]


def agts(queue):
    h2_osc = queue.add('h2_osc.py', ncpus=8, walltime=2 * 60)
    n2_osc = queue.add('n2_osc.py', ncpus=5 * 8, walltime=15 * 60)
    na2_md = queue.add('na2_md.py', ncpus=8, walltime=2 * 60)
    na2_osc = queue.add('na2_osc.py', ncpus=8, walltime=40 * 60)
    if 0:  # Disabling. See https://listserv.fysik.dtu.dk/pipermail/gpaw-developers/2013-January/003402.html
        queue.add('oscfit.py', ncpus=1, walltime=5,
                  deps=[h2_osc, n2_osc, na2_md, na2_osc],
                  creates=['h2_osc.png', 'n2_osc.png',
                           'na2_md.png', 'na2_osc.png'])
