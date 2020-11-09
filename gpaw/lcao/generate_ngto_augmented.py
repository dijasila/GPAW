import os
import re
import numpy as np
from gpaw.atom.basis import BasisMaker
from gpaw.atom.basis import QuasiGaussian
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.atom.configurations import parameters, parameters_extra
from gpaw.basis_data import BasisFunction
from gpaw.basis_data import parse_basis_name
from gpaw.mpi import world

# Module for generating basis sets that compose of usual basis sets
# augmented with Gaussian type orbital (GTO).
#
# GTOs are truncated and represented numerically.


def read_gbs(fname):
    """Read gbs file.

    This reads only the first element/atom from the file
    as separated with line beginning with '*'.
    """
    gto_k = []
    description = ''

    f = open(fname, 'r')
    line_i = f.readlines()
    f.close()

    i = 0
    Ni = len(line_i)
    while True:
        line = line_i[i].strip()
        if line == '' or line[0] == '*':
            pass
        elif line[0] == '!':
            description += '%s\n' % line[1:].strip()
        else:
            break
        i += 1
    description = description.strip()

    atom = line_i[i].strip().split()[0]
    i += 1
    while i < Ni:
        line = line_i[i]
        if line[0] == '*':
            break
        i += 1
        d = line.split()
        l = 'spdfghi'.index(d[0].lower())
        Nj = int(d[1])
        alpha_j = []
        coeff_j = []
        for _ in range(Nj):
            line = line_i[i]
            d = line.split()
            alpha = float(d[0].replace('D', 'E'))
            coeff = float(d[1].replace('D', 'E'))
            alpha_j.append(alpha)
            coeff_j.append(coeff)
            i += 1
        gto_k.append({'l': l, 'alpha_j': alpha_j, 'coeff_j': coeff_j})

    return atom, description, gto_k


def get_ngto(rgd, l, alpha, rcut):
    gaussian = QuasiGaussian(alpha, rcut)
    psi_g = gaussian(rgd.r_g) * rgd.r_g**l
    norm = np.sum(rgd.dr_g * (rgd.r_g * psi_g)**2)**.5
    psi_g /= norm
    return psi_g


def create_ngto(rgd, l, alpha, rmax, tol):
    # Get NGTO with the initial (large) rcut=rmax
    psiref_g = get_ngto(rgd, l, alpha, rmax)

    # Make rcut smaller

    # Guess initial rcut where we are close to the tolerance
    i = np.where(psiref_g > tol)[0][-1]
    rcut = rgd.r_g[i]
    psi_g = get_ngto(rgd, l, alpha, rcut)
    err = np.max(np.absolute(psi_g - psiref_g))

    # Increase/decrease rcut to find the smallest rcut
    # that yields error within the tolerance
    if err > tol:
        # Increase rcut -> decrease err
        for i in range(i, len(rgd.r_g)):
            rcut = rgd.r_g[i]
            psi_g = get_ngto(rgd, l, alpha, rcut)
            err = np.max(np.absolute(psi_g - psiref_g))
            if err < tol:
                break
    else:
        # Decrease rcut -> increase err
        for i in range(i, 0, -1):
            rcut = rgd.r_g[i]
            psi_g = get_ngto(rgd, l, alpha, rcut)
            err = np.max(np.absolute(psi_g - psiref_g))
            if err > tol:
                i += 1
                break

    # Construct NGTO with the found rcut
    rcut = rgd.r_g[i]
    psi_g = get_ngto(rgd, l, alpha, rcut)

    # Change norm (maybe unnecessary)
    psi_g = psi_g[:(i + 1)] * 0.5

    return psi_g


def add_ngto(basis, l, coeff_j, alpha_j, tol, label):
    rgd = basis.get_grid_descriptor()
    rmax = rgd.r_g[-1]

    # Create linear combination of NGTO's
    psi_g = np.zeros(rgd.r_g.shape)
    i_max = 0
    for coeff, alpha in zip(coeff_j, alpha_j):
        contrib = coeff * create_ngto(rgd, l, alpha, rmax, tol)
        i = contrib.size
        i_max = max(i, i_max)
        psi_g[0:i] += contrib

    psi_g = psi_g[0:i_max]
    rcut = rgd.r_g[i_max]

    # Create associated basis function
    bf = BasisFunction(None, l, rcut, psi_g, label)
    basis.bf_j.append(bf)


def do_nao_ngto_basis(atom, xc, naobasis, gbsfname, label, rmax=100.0,
                      tol=0.001):
    # Read Gaussians
    atomgbs, descriptiongbs, gto_k = read_gbs(gbsfname)
    assert atom == atomgbs

    # Generate nao basis
    zetacount, polarizationcount = parse_basis_name(naobasis)

    # Choose basis sets without semi-core states XXXXXX
    if atom == 'Ag':
        label = '11.%s' % label
        p = parameters_extra
    else:
        p = parameters

    bm = BasisMaker(atom, label, run=False, gtxt=None, xc=xc)
    bm.generator.run(write_xml=False, use_restart_file=False, **p[atom])
    basis = bm.generate(zetacount, polarizationcount, txt=None)

    # Increase basis function max radius
    assert isinstance(basis.rgd, EquidistantRadialGridDescriptor)
    h = basis.rgd.dr_g[0]
    assert basis.rgd.r_g[0] == 0.0
    N = int(rmax / h) + 1
    basis.rgd = EquidistantRadialGridDescriptor(h, N)

    # Add NGTOs

    description = []
    msg = 'Augmented with NGTOs'
    description.append(msg)
    description.append('=' * len(msg))
    description.append('')
    msg = 'GTOs from file %s' % os.path.basename(gbsfname)
    description.append(msg)
    description.append('-' * len(msg))
    description.append(descriptiongbs)
    description.append('')
    description.append('NGTO truncation tolerance: %f' % tol)
    description.append('Functions: NGTO(l,coeff*alpha + ...)')

    for gto in gto_k:
        l = gto['l']
        alpha_j = gto['alpha_j']
        coeff_j = gto['coeff_j']
        coeff_alpha_list = ['%+.3f*%.3f' % (c, a)
                            for c, a in zip(coeff_j, alpha_j)]
        coeff_alpha_label = ''.join(coeff_alpha_list[0:3])
        if len(coeff_alpha_list) > 3:
            coeff_alpha_label += '+...'
        ngtolabel = 'NGTO(%s,%s)' % ('spdfghi'[l], coeff_alpha_label)
        description.append('    ' + ngtolabel)
        add_ngto(basis, l, coeff_j, alpha_j, tol, ngtolabel)

    basis.generatordata += '\n\n' + '\n'.join(description)

    basis.write_xml()


def main():
    xc = 'PBE'

    # Process all gbs files
    fname_i = [fname for fname in sorted(os.listdir('.'))
               if fname.endswith('.gbs')]
    for i, fname in enumerate(fname_i):
        if i % world.size != world.rank:
            continue
        m = re.match(r'(?P<atom>\w+)-(?P<label>NAO-(?P<nao>\w+)\+' +
                     r'NGTO-N(?P<Nngto>\d+)).gbs', fname)
        if m is not None:
            if world.size > 1:
                print(world.rank, fname)
            else:
                print(fname)
            do_nao_ngto_basis(m.group('atom'), xc, m.group('nao'),
                              fname, m.group('label'))


if __name__ == '__main__':
    main()
