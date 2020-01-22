# creates: paw_note.pdf
import subprocess

import matplotlib.pyplot as plt

from gpaw.atom.all_electron import AllElectron

ae = AllElectron('Pt')
ae.run()

fig = plt.figure(figsize=(7, 4), dpi=80)
fig.subplots_adjust(left=0.05, bottom=0.11, right=0.85, top=0.95)
for n, l, u in zip(ae.n_j, ae.l_j, ae.u_j):
    plt.plot(ae.r, u, label='%i%s' % (n, 'spdf'[l]))

rcut = 2.5
lim = [0, 3.5, -2, 3]
plt.plot([rcut, rcut], lim[2:], 'k--', label='_nolegend_')
plt.axis(lim)

kwpad = {'borderpad': 0.05, 'labelspacing': 0.01}

plt.legend(loc=(1.02, 0.03), markerscale=1, **kwpad)
plt.xlabel(r'$r$ [Bohr]')
plt.text(rcut + 0.05, lim[2] + 0.05, '$r_c$', ha='left', va='bottom')
plt.text(0.6, 2, '[Pt] = [Xe]4f$^{14}$5d$^9$6s$^1$')
plt.savefig('Pt.png', dpi=80)

try:
    subprocess.run(
        'pdflatex -interaction=nonstopmode paw_note > /dev/null && '
        'bibtex paw_note > /dev/null && '
        'pdflatex -interaction=nonstopmode paw_note > /dev/null && '
        'pdflatex -interaction=nonstopmode paw_note > /dev/null && '
        'cp paw_note.pdf ..',
        shell=True, check=True)
except subprocess.CalledProcessError:
    subprocess.run('echo "No pdflatex" > paw_note.pdf; cp paw_note.pdf ..',
                   shell=True)
