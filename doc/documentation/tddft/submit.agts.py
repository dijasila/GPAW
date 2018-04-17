from q2.job import Job


def workflow():
    return [
        Job('Be_gs_8bands.py@2x20s'),
        Job('Be_8bands_lrtddft.py@2x20s', deps=['Be_gs_8bands.py']),
        Job('Be_8bands_lrtddft_dE.py@2x20s', deps=['Be_gs_8bands.py']),
        Job('Na2_relax_excited.py@4x8m')]

def agts(queue):
    calc1 = queue.add('Be_gs_8bands.py',
                      ncpus=2,
                      walltime=20)
    queue.add('Be_8bands_lrtddft.py',
              ncpus=2,
              walltime=20,
              deps=calc1)
    queue.add('Be_8bands_lrtddft_dE.py',
              ncpus=2,
              walltime=20,
              deps=calc1)
    queue.add('Na2_relax_excited.py',
              ncpus=4,
              walltime=500)
