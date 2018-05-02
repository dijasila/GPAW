from myqueue.job import Job


def workflow():
    return [
        Job('Be_gs_8bands.py@2x20m'),
        Job('Be_8bands_lrtddft.py@2x20m', deps=['Be_gs_8bands.py']),
        Job('Be_8bands_lrtddft_dE.py@2x20m', deps=['Be_gs_8bands.py']),
        Job('Na2_relax_excited.py@4x8h')]
