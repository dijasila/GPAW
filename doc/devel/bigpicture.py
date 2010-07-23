# creates: bigpicture.pdf bigpicture.png
import os
from math import pi, cos, sin

import numpy as np

latex = r"""\documentclass[10pt,landscape]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\parindent=0pt
\pagestyle{empty}
%\usepackage{landscape}
\usepackage{pstricks-add,graphicx,hyperref}
\usepackage[margin=1cm]{geometry}
\newsavebox\PSTBox
%\special{papersize=420mm,297mm}
\begin{document}

\psset{framesep=2mm,arrowscale=1.75}

\begin{pspicture}(0,0)(27,18)
\psframe*[linecolor=green!20](21.5,13)(26,17.9)
\rput(22,17.5){ASE}
%\newrgbcolor{yellow7}{0.97 0.5 0.85}
"""

all = []
names = {}

class Box:
    def __init__(self, name, description=None, attributes=None,
                 color='black!20', width=None):
        self.position = None
        self.name = name
        if isinstance(description, str):
                description = [description]
        self.description = description
        self.attributes = attributes
        self.width = width
        self.color = color
        self.owns = []
        all.append(self)
        if name in names:
            names[name] += 1
            self.id = name + str(names[name])
        else:
            self.id = name
            names[name] = 1
            
    def set_position(self, position):
        self.position = np.asarray(position)

    def to_latex(self):
        if self.width:
            format = 'p{%fcm}' % self.width
        else:
            format = 'c'
        boxes = [
            '\\rput(%f,%f){' % tuple(self.position) +
            '\\rnode{%s}{' % self.id +
            '\\psshadowbox[fillcolor=%s,fillstyle=solid]{' %
            self.color + '\\begin{tabular}{%s}' % format]
        url = ''#'\\href{https://wiki.fysik.dtu.dk/gpaw/devel/devel.html}'
        table = [url + '{\small %s}' % self.name]
        if self.description:
            table.extend(['{\\tiny %s}' % txt for txt in self.description])
        if self.attributes:
            table.append('{\\tiny \\texttt{%s}}' %
                         ', '.join(self.attributes).replace('_', '\\_'))
        boxes.append('\\\\\n'.join(table))
        boxes += ['\\end{tabular}}}}']
        arrows = []
        for other, name, x in self.owns:
            arrows += ['\\ncline{->}{%s}{%s}' % (self.id, other.id)]
            if name:
                arrows += [
                    '\\rput(%f, %f){\\psframebox*[framesep=0.05]{\\tiny %s}}' %
                    (tuple(((1 - x) * self.position + x * other.position)) +
                     (name.replace('_', '\\_'),))]
                
        return boxes, arrows

    def has(self, other, name, angle=None, distance=None, x=0.55):
        self.owns.append((other, name, x))
        if angle is not None:
            angle *= pi / 180
            other.set_position(self.position +
                               [cos(angle) * distance,
                                sin(angle) * distance])

atoms = Box('Atoms', '', ['positions', 'numbers', 'cell', 'pbc'],
            color='white')
paw = Box('PAW', None, ['initialized'], 'green!75')
scf = Box('SCFLoop', None)
density = Box('Density', 
              [r'$\tilde{n}_\sigma = \sum_{\mathbf{k}n}' +
               r'|\tilde{\psi}_{\sigma\mathbf{k}n}|^2' +
               r'+\frac{1}{2}\sum_a \tilde{n}_c^a$',
               r'$\tilde{\rho}(\mathbf{r}) = ' +
               r'\sum_\sigma\tilde{n}_\sigma + \sum_{aL}Q_L^a \hat{g}_L^a$'],
              ['nspins', 'nt_sG', 'nt_sg', 'rhot_g', 'Q_aL', 'D_asp'])
mixer = Box('Mixer')#, color='blue!30')
hamiltonian = Box('Hamiltonian',
                  r"""$-\frac{1}{2}\nabla^2 +
 \tilde{v} +
 \sum_a \sum_{i_1i_2} |\tilde{p}_{i_1}^a \rangle 
 \Delta H_{i_1i_2} \langle \tilde{p}_{i_2}^a|$""",
                  ['nspins', 'vt_sG', 'vt_sg', 'vHt_g', 'dH_asp',
                   'Etot', 'Ekin', 'Exc', 'Epot', 'Ebar'])
wfs = Box('WaveFunctions',
          r"""$\tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r})$""",
          ['nspins', 'ibzk_qc', 'mynbands',
           'kpt_comm', 'band_comm'], width=2.5, color='magenta!60')
gd = Box('GridDescriptor', '(coarse grid)',
         ['cell_cv', 'N_c', 'pbc_c', 'dv', 'comm'], 'orange!90')
finegd = Box('GridDescriptor', '(fine grid)',
         ['cell_cv', 'N_c', 'pbc_c', 'dv', 'comm'], 'orange!90')
rgd = Box('RadialGridDescriptor', None, ['r_g, dr_g, rcut'], color='orange!90')
setups = Box('Setups', ['', '', '', ''], ['nvalence', 'nao', 'Eref'],
             width=4.2)
xccorrection = Box('XCCorrection')
nct = Box('LFC', r'$\tilde{n}_c^a(r)$', None, 'red!70')
vbar = Box('LFC', r'$\bar{v}^a(r)$', None, 'red!70')
ghat = Box('LFC', r'$\hat{g}_{\ell m}^a(\mathbf{r})$', None, 'red!70')
fd = Box('FDWaveFunctions',
           r"""$\tilde{\psi}_{\sigma\mathbf{k}n}(ih,jh,kh)$""",
           None, 'magenta!60')
pt = Box('LFC', r'$\tilde{p}_i^a(\mathbf{r})$', None, 'red!70')
lcao = Box('LCAOWaveFunctions',
           r"""$\tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r})=
\sum_{\mu\mathbf{R}} C_{\sigma\mathbf{k}n\mu}
\Phi_\mu(\mathbf{r} - \mathbf{R}) \exp(i\mathbf{k}\cdot\mathbf{R})$""",
           ['S_qMM', 'T_qMM', 'P_aqMi'], 'magenta!60')
atoms0 = Box('Atoms', '(copy)', ['positions', 'numbers', 'cell', 'pbc'],
             color='black!5')
parameters = Box('InputParameters', None, ['xc', 'nbands', '...'])
forces = Box('ForceCalculator')
occupations = Box(
    'OccupationNumbers',
    r'$\epsilon_{\sigma\mathbf{k}n} \rightarrow f_{\sigma\mathbf{k}n}$')
poisson = Box('PoissonSolver',
              r'$\nabla^2 \tilde{v}_H(\mathbf{r}) = -4\pi \tilde{\rho}(\mathbf{r})$')
eigensolver = Box('EigenSolver')
symmetry = Box('Symmetry')
restrictor = Box('Transformer', '(fine -> coarse)',
                 color='yellow!80')
interpolator = Box('Transformer', '(coarse -> fine)',
                   color='yellow!80')
xcfunc = Box('XCFunctional')
xc3dgrid = Box('XC3DGrid', color='brown!80')
xc1dgrid = Box('XCRadialGrid', color='brown!80')
kin = Box('FDOperator', r'$-\frac{1}{2}\nabla^2$')
hsoperator = Box('HSOperator',
                 r"$\langle \psi_n | A | \psi_{n'} \rangle,~" +
                 r"\sum_{n'}U_{nn'}|\tilde{\psi}_{n'}\rangle$")
                 
overlap = Box('Overlap')
basisfunctions = Box('BasisFunctions', r'$\Phi_\mu(\mathbf{r})$',
                     color='red!70')
tci = Box('TwoCenterIntegrals',
          r'$\langle\Phi_\mu|\Phi_\nu\rangle,'
          r'\langle\Phi_\mu|\hat{T}|\Phi_\nu\rangle,'
          r'\langle\tilde{p}^a_i|\Phi_\mu\rangle$')

atoms.set_position((23.5, 16))
atoms.has(paw, 'calculator', -160, 7.5)
paw.has(scf, 'scf', 160, 4, x=0.48)
paw.has(density, 'density', -150, 14, 0.23)
paw.has(hamiltonian, 'hamiltonian', 180, 10, 0.3)
paw.has(wfs, 'wfs', -65, 5.5, x=0.48)
paw.has(atoms0, 'atoms', 9, 7.5)
paw.has(parameters, 'input_parameters', 90, 4)
paw.has(forces, 'forces', 50, 4)
paw.has(occupations, 'occupations', 136, 4)
density.has(mixer, 'mixer', 130, 3.3)
density.has(gd, 'gd', x=0.33)
density.has(finegd, 'finegd', 76, 3.5)
density.has(setups, 'setups', 0, 7, 0.45)
density.has(nct, 'nct', -90, 3)
density.has(ghat, 'ghat', -130, 3.4)
density.has(interpolator, 'interpolator', -45, 4)
hamiltonian.has(restrictor, 'restrictor', 40, 4)
hamiltonian.has(xc3dgrid, 'xc', 160, 6, x=0.6)
hamiltonian.has(vbar, 'vbar', 80, 4)
hamiltonian.has(setups, 'setups', x=0.3)
hamiltonian.has(gd, 'gd', x=0.45)
hamiltonian.has(finegd, 'finegd')
hamiltonian.has(poisson, 'poissonsolver', 130, 4)
hamiltonian.has(xcfunc, 'xcfunc', x=0.6)
xc3dgrid.has(xcfunc, 'xcfunc', -90, 4.5)
wfs.has(gd, 'gd', 160, 4.8, x=0.48)
wfs.has(setups, 'setups', x=0.4)
wfs.has(lcao, 'INSTANCE', -55, 5.9)
wfs.has(fd, 'INSTANCE', -112, 5.0)
wfs.has(eigensolver, 'eigensolver', 30, 5, x=0.6)
wfs.has(symmetry, 'symmetry', 80, 3)
fd.has(pt, 'pt', -45, 3.6)
fd.has(kin, 'kin', -90, 3)
fd.has(overlap, 'overlap', -135, 3.5)
lcao.has(basisfunctions, 'basis_functions', -50, 3.5)
lcao.has(tci, 'tci', -90, 4.2)
overlap.has(setups, 'setups', x=0.4)
overlap.has(hsoperator, 'operator', -115, 2.5, x=0.41)

for i in range(3):
    setup = Box('Setup', None,
                ['Z', 'Nv','Nc', 'pt_j','nct', 'vbar','ghat_l', 'Delta_pl'],
                'blue!40', width=2.1)
    setup.set_position(setups.position +
                       (0.9 - i * 0.14, 0.3 - i * 0.14))
setup.has(xccorrection, 'xc_correction', -110, 3.7)
xccorrection.has(rgd, 'rgd', -105, 2.4, 0.4)
xccorrection.has(xc1dgrid, 'xc', -170, 10.03)
xc1dgrid.has(xcfunc, 'xcfunc')

kpts = [Box('KPoint', None, ['psit_nG', 'C_nM', 'eps_n', 'f_n', 'P_ani'],
            color='cyan!50') for i in range(3)]
wfs.has(kpts[1], 'kpt_u', 0, 5.4, 0.48)
kpts[0].set_position(kpts[1].position - 0.14)
kpts[2].set_position(kpts[1].position + 0.14)

allboxes = []
allarrows = []
for b in all:
   boxes, arrows = b.to_latex()
   allboxes.extend(boxes)
   allarrows.extend(arrows)
   
latex = [latex] + allboxes + allarrows + ['\\end{pspicture}\n\\end{document}']
open('bigpicture.tex', 'w').write('\n'.join(latex))

os.system('latex -halt-on-error bigpicture.tex > bigpicture.log')
os.system('dvipdf bigpicture.dvi')
os.system('cp bigpicture.pdf ../_build')
os.system('convert bigpicture.pdf -resize 50% bigpicture.png')
