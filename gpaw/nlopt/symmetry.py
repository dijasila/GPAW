
# Import the required modules: General
import itertools

# Import the required modules: GPAW/ASE
from ase.spacegroup import Spacegroup, get_spacegroup
from ase.io import read

# Get the non-zero tensor elements


def get_tensor_elements(atoms):
    """
    # Get the non-zero tensor elements of a structure and their symmetries
    Input:
        atoms           ASE atoms object used for the phonon calculation
    Output:
        tensordict      A dictionary with symmetry of the elements
    """
    # Get the space group
    if isinstance(atoms, int):
        sg = Spacegroup(atoms)
    else:
        sg = get_spacegroup(atoms, symprec=1e-3)

    # Find the group point
    spno = sg.no
    diff = [x - spno for x in splist]
    if 0 in diff:
        spind = diff.index(0)
    else:
        for x in diff:
            if x > 0:
                spind = diff.index(x)
                break
    pgstr = gplist[spind]

    # invlist = ['-1', '2 / m', '2 / m 2 / m 2 / m', '4 / m',
    # '4 / m 2 / m 2 / m',
    #  '-3', '-3 2 / m', '6 / m', '6 / m 2 / m 2 / m', '2 / m -3',
    # '4 3 2', '4 / m -3 2 / m']
    # if pgstr in invlist:
    if spno == 189:
        pgstr += '_2'
    seltensor = dtensor[pgstr]

    # Return the tensor dictionary
    tensordict = make_relations(seltensor)
    return tensordict

# Get the unique tensor elements for each tensor


def get_unique(seltensor):
    """
    Get the unique tensor elements for each tensor
    Input:
        seltensor       Selected tensor
    Output:
        uqlist          Unique list of tensor elements
    """
    # Flaten the list of tensor elements
    flatten = itertools.chain.from_iterable

    # Make the tensor a flat list
    seltensor = list(flatten(seltensor))
    # Discard zero elements
    nzlist = [elem for elem in seltensor if elem != 0]
    # Make a unique list of required elements
    uqlist = [elem[1:] for elem in nzlist]
    uqlist = list(set(uqlist))

    # Return the unique list
    return uqlist

# Make a dictionary of all relations


def make_relations(seltensor):
    """
    Make a dictionary of all relations

    Input:
        seltensor       Selected tensor
    Output:
        tensordict      A dictionary with symmetry of the elements
    """
    # Get the unique tensor elements
    uqlist = get_unique(seltensor)
    tensordict = dict(zip(uqlist, uqlist))
    tensordict['zero'] = ''

    # Loop over all components
    for ii, alpha in enumerate(['x', 'y', 'z']):
        for jj, beta in enumerate(['xx', 'yy', 'zz', 'yz', 'xz', 'xy']):
            comp = alpha + beta
            if seltensor[ii][jj] != 0:
                if seltensor[ii][jj] == ('+' + comp):
                    if jj >= 3:
                        tensordict[comp] += '=' + alpha + beta[-1::-1]
                else:
                    tensval = seltensor[ii][jj]
                    signval = '' if tensval[0] == '+' else '-'
                    tensordict[tensval[1:]] += '=' + signval + comp
                    if jj >= 3:
                        tensordict[tensval[1:]] += '=' + \
                            signval + alpha + beta[-1::-1]

            else:
                tensordict['zero'] += comp + '='
                if jj >= 3:
                    tensordict['zero'] += alpha + beta[-1::-1] + '='

    # Remove the extra '='
    if tensordict['zero']:
        if tensordict['zero'][-1] == '=':
            tensordict['zero'] = tensordict['zero'][:-1]

    # Return the tensor dictionary
    return tensordict


# List of transition between space groups and point groups
splist = [1, 2,  # Triclinic
          5, 9, 15,  # Monoclinic
          24, 46, 74,  # Orthorhombic
          80, 82, 88, 98, 110, 122, 142,  # Tetragonal
          146, 148, 155, 161, 167,  # Trigonal
          173, 174, 176, 182, 186, 190, 194,  # Hexagonal
          199, 206, 214, 220, 230]  # Cubic
gplist = ['1', '-1',  # Triclinic
          '2', 'm', '2 / m',  # Monoclinic
          '2 2 2', 'm m 2', '2 / m 2 / m 2 / m',  # Orthorhombic
          '4', '-4', '4 / m', '4 2 2', '4 m m', '-4 2 m', '4 / m 2 / m 2 / m',
          '3', '-3', '3 2', '3 m', '-3 2 / m',  # Trigonal
          '6', '-6', '6 / m', '6 2 2', '6 m m', '-6 m 2', '6 / m 2 / m 2 / m',
          '2 3', '2 / m -3', '4 3 2', '-4 3 m', '4 / m -3 2 / m']  # Cubic

# Make a library of tensor with their symmetries
dtensor = dict()
# Triclinic class: 2
dtensor['1'] = [['+xxx', '+xyy', '+xzz', '+xyz', '+xxz', '+xxy'],
                ['+yxx', '+yyy', '+yzz', '+yyz', '+yxz', '+yxy'],
                ['+zxx', '+zyy', '+zzz', '+zyz', '+zxz', '+zxy']]
dtensor['-1'] = [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]
# Monoclinic class: 3
# dtensor['2'] = [[0, 0, 0, '+xyz', 0, '+xxy'],
#                 ['+yxx', '+yyy', '+yzz', 0, '+yxz', 0],
#                 [0, 0, 0, '+zyz', 0, '+zxy']]
dtensor['2'] = [[0, 0, 0, 0, '+xxz', '+xxy'],
                ['+yxx', '+yyy', '+yzz', '-xxz', 0, 0],
                ['+zxx', '-xxy', '+zzz', '+zyz', 0, 0]]
dtensor['m'] = [['+xxx', '+xyy', '+xzz', 0, '+xxz', 0],
                [0, 0, 0, '+yyz', 0, '+yxy'],
                ['+zxx', '+zyy', '+zzz', 0, '+zxz', 0]]  # for AB2 6 group
# dtensor['m'] = [[0, 0, 0, 0, '+xxz', '+xxy'], ['+yxx', '+yyy', '+yzz',
# '+yyz', 0, 0], ['+zxx', '+zyy', '+zzz', '+zyz', 0, 0]] # x to y change
# for AB 6 group
dtensor['2 / m'] = [[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
# Orthorhombic class: 3
dtensor['2 2 2'] = [[0, 0, 0, '+xyz', 0, 0],
                    [0, 0, 0, 0, '+yxz', 0],
                    [0, 0, 0, 0, 0, '+zxy']]
# dtensor['m m 2'] = [[0, 0, 0, 0, '+xxz', 0],
#                     [0, 0, 0, '+yyz', 0, 0],
#                     ['+zxx', '+zyy', '+zzz', 0, 0, 0]]
dtensor['m m 2'] = [['+xxx', '+xyy', '+xzz', 0, 0, 0],
                    [0, 0, 0, 0, 0, '+yxy'],
                    [0, 0, 0, 0, '+zxz', 0]]  # x to z change
dtensor['2 / m 2 / m 2 / m'] = [[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]
# Tetragonal class: 7
dtensor['4'] = [[0, 0, 0, '+xyz', '+xxz', 0],
                [0, 0, 0, '+xxz', '-xyz', 0],
                ['+zxx', '+zxx', '+zzz', 0, 0, 0]]
dtensor['-4'] = [[0, 0, 0, '+xyz', '+xxz', 0],
                 [0, 0, 0, '-xxz', '+xyz', 0],
                 ['+zxx', '-zxx', 0, 0, 0, '+zxy']]
dtensor['4 / m'] = [[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
dtensor['4 2 2'] = [[0, 0, 0, '+xyz', 0, 0],
                    [0, 0, 0, 0, '-xyz', 0],
                    [0, 0, 0, 0, 0, 0]]
dtensor['4 m m'] = [[0, 0, 0, 0, '+xxz', 0],
                    [0, 0, 0, '+xxz', 0, 0],
                    ['+zxx', '+zxx', '+zzz', 0, 0, 0]]
# dtensor['-4 2 m'] = [[0, 0, 0, '+xyz', 0, 0],
#                      [0, 0, 0, 0, '+xyz', 0],
#                      [0, 0, 0, 0, 0, '+zxy']]
dtensor['-4 2 m'] = [[0, 0, 0, 0, '+xxz', 0],
                     [0, 0, 0, '-xxz', 0, 0],
                     ['+zxx', '-zxx', 0, 0, 0, 0]]  # Rotate 45 degrees
dtensor['4 / m 2 / m 2 / m'] = [[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]
# Trigonal class: 5
dtensor['3'] = [['+xxx', '-xxx', 0, '+xyz', '+xxz', '-yyy'],
                ['-yyy', '+yyy', 0, '+xxz', '-xyz', '-xxx'],
                ['+zxx', '+zxx', '+zzz', 0, 0, 0]]
dtensor['-3'] = [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]
dtensor['3 2'] = [['+xxx', '-xxx', 0, '+xyz', 0, 0],
                  [0, 0, 0, 0, '-xyz', '-xxx'],
                  [0, 0, 0, 0, 0, 0]]
dtensor['3 m'] = [[0, 0, 0, 0, '+xxz', '-yyy'],
                  ['-yyy', '+yyy', 0, '+xxz', 0, 0],
                  ['+zxx', '+zxx', '+zzz', 0, 0, 0]]
dtensor['-3 2 / m'] = [[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]]
# Hexagonal class: 7
dtensor['6'] = [[0, 0, 0, '+xyz', '+xxz', 0],
                [0, 0, 0, '+xxz', '-xyz', 0],
                ['+zxx', '+zxx', '+zzz', 0, 0, 0]]
dtensor['-6'] = [['+xxx', '-xxx', 0, 0, 0, '-yyy'],
                 ['-yyy', '+yyy', 0, 0, 0, '-xxx'],
                 [0, 0, 0, 0, 0, 0]]
dtensor['6 / m'] = [[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
dtensor['6 2 2'] = [[0, 0, 0, '+xyz', 0, 0],
                    [0, 0, 0, 0, '-xyz', 0],
                    [0, 0, 0, 0, 0, 0]]
dtensor['6 m m'] = [[0, 0, 0, 0, '+xxz', 0],
                    [0, 0, 0, '+xxz', 0, 0],
                    ['+zxx', '+zxx', '+zzz', 0, 0, 0]]
dtensor['-6 m 2'] = [[0, 0, 0, 0, 0, '-yyy'],
                     ['-yyy', '+yyy', 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]]
dtensor['-6 m 2_2'] = [['+xxx', '-xxx', 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, '-xxx'],
                       [0, 0, 0, 0, 0, 0]]
dtensor['6 / m 2 / m 2 / m'] = [[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]
# Cubic class: 5
dtensor['2 3'] = [[0, 0, 0, '+xyz', 0, 0],
                  [0, 0, 0, 0, '+xyz', 0],
                  [0, 0, 0, 0, 0, '+xyz']]
dtensor['2 / m -3'] = [[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]]
dtensor['4 3 2'] = [[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
dtensor['-4 3 m'] = [[0, 0, 0, '+xyz', 0, 0],
                     [0, 0, 0, 0, '+xyz', 0],
                     [0, 0, 0, 0, 0, '+xyz']]
dtensor['4 / m -3 2 / m'] = [[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]]

if __name__ == '__main__':
    # Read the data
    atoms_name = 'structure.json'
    atoms = read(atoms_name)

    tensordict = get_tensor_elements(atoms)
    # print(tensordict)

    # for ii in range(1, 230):
    #     tensordict = get_tensor_elements(ii)
    #     print(tensordict)
