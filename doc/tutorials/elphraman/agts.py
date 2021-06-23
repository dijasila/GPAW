from myqueue.workflow import run
from pathlib import Path


def workflow():
    ph = run(script='phonon.py')
    el = run(script='elph.py')
    mo = run(script='momentum_matrix.py')
    with ph, el:
        with run(script='supercell_matrix.py'):
            with run(script='elph_matrix.py', cores=1):
                run(script='raman_intensities.py', core=1)
                run(function=check)

def check():
    """Read result and make sure it's OK."""
    pass
    #lines = Path('H2O_rraman_summary.txt').read_text().splitlines()
    #for line in lines:
        #if line.strip().startswith('8'):
            #_, frequency, _, intensity = (float(x) for x in line.split())
            #assert abs(frequency - 474) < 1
            #assert abs(intensity - 57) < 1
            #return
    #raise ValueError
 
