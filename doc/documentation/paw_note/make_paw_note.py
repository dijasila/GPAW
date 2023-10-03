# creates: paw_note.pdf
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

from gpaw.atom.aeatom import AllElectronAtom

ae = AllElectronAtom('Pt')
ae.run()
ae.plot_wave_functions(show=False)
rcut = 2.5
lim = [0, 3.5, -2, 4]
plt.plot([rcut, rcut], lim[2:], 'k--', label='_nolegend_')
plt.axis(lim)
plt.text(0.6, 2, '[Pt] = [Xe]4f$^{14}$5d$^9$6s$^1$')
plt.savefig('Pt.png', dpi=80)

dir = os.environ.get('PDF_FILE_DIR')
if dir:
    shutil.copyfile(Path(dir) / 'paw_note.pdf', 'paw_note.pdf')
else:
    try:
        subprocess.run(
            'pdflatex -interaction=nonstopmode paw_note > /dev/null && '
            'bibtex paw_note > /dev/null && '
            'pdflatex -interaction=nonstopmode paw_note > /dev/null && '
            'pdflatex -interaction=nonstopmode paw_note > /dev/null',
            shell=True, check=True)
    except subprocess.CalledProcessError:
        subprocess.run('echo "No pdflatex" > paw_note.pdf', shell=True)

subprocess.run(['cp', 'paw_note.pdf', '..'])
