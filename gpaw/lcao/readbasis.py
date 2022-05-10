from collections import defaultdict

def lines(f):
    lines = f.readlines()
    for line in lines:
        yield line

def strip_comment(line):
    return line.split('#')[0].strip()

def group_by_l(basis):
    print('group_by_l', basis)
    basis_l = defaultdict(list)
    for l, contraction in basis:
        basis_l[l].append(contraction)

    # Default dict ensures an empty array even in case of a missing channel
    return [ basis_l[l] for l in range(max(basis_l)+1) ] 

class BasisSetLibrary:
    def __init__(self, library):
        self.library = library

    def __repr__(self):
        s = ''
        for name in self.library:
            s += name +'\n'
        return s

    def resolve(self, basisname):
        basis = self.library[basisname]
        print('Resolve start', basis)        
        resolved = []
        for l, r in basis:
            if isinstance(l, str):
                ref_basis = self.resolve('%s-%s' % (l, r)) 
                resolved.extend( ref_basis )
            else:
                resolved.append( (l, r) )
        return resolved

    def get(self, basisname):
        basis = self.resolve(basisname)
        if len(basis) == 0:
            raise ValueError('Basis not found %s' % basisname)
        return group_by_l(basis)

class LibraryBuilder:
    def __init__(self, debug = True):
        self.library = defaultdict(list)
        self.current_bases = []
        self.debug = debug

        self.coeff = []
        self.current_channel = None
        self.contractions_remaining = 0
        self.current_references = []
        self.current_basis = []

    def has_names(self):
        return len(self.current_bases)>0

    def get_library(self):
        return BasisSetLibrary(self.library)

    def end_input(self):
        print(self.current_basis, self.current_bases, self.current_references,'xxx')
        for basename in self.current_bases:
            self.library[basename].extend(self.current_basis)
            self.library[basename].extend(self.current_references)

        self.current_bases = []
        self.current_references = []
        self.current_basis = []

    def basis_name(self, element, name):
        self.current_bases.append( '%s-%s' % (element, name.strip()) )
        if self.debug:
            print(element, name)

    def contraction(self,  contractions, channel):
        self.current_channel = channel
        self.contractions_remaining = contractions

        if self.debug:
            print('basis', contractions, channel)

    def coefficients(self, exponent, coefficient):
        self.contractions_remaining -= 1
        if self.contractions_remaining <0:
            raise RuntimeError('Unexpected coefficient input in basis set file')
        self.coeff.append( (exponent, coefficient) )
        if self.contractions_remaining == 0:
            self.current_basis.append( (self.current_channel, self.coeff) )
            self.coeff = []

        if self.debug:
            print('coeff', exponent, coefficient)

    def add_partial(self, element, name):
        if self.debug:
            print('->', element, name)
        self.current_references.append( (element, name) )

def basis_input(lines, builder):
    bases = defaultdict(list) 
    mode = 'basisnames'
    while 1:
        command = strip_comment(next(lines))
        if command == '':
            continue
        elif command == '$ecp':
            break
        elif command == '*':
            if mode == 'basisnames':
                if builder.has_names():
                    mode = 'basisinput'
            elif mode == 'basisinput':
                mode = 'basisnames'
                builder.end_input()
            else:
                raise RuntimeError('Internal error. Unknown mode: %s' % command)
        else:
            if mode == 'basisnames':
                try:
                    element, name = command.split(' ', 1)
                    builder.basis_name(element, name)
                except ValueError:
                    raise RuntimeError('Unknown command %s' % command)
            else:
                splits = command.split()
                if splits[0] == '->':
                    assert len(splits)==3
                    builder.add_partial(splits[1], splits[2])
                    continue
                try:
                    contractions, channel = splits
                    contractions = int(contractions)
                    builder.contraction(contractions, 'spdfghijk'.find(channel))
                    for n in range(contractions):
                        exponent, coefficient = map(float, strip_comment(next(lines)).replace('D','e').split())
                        builder.coefficients(exponent, coefficient)
                except ValueError:
                    raise RuntimeError('Unknown command %s' % command)
               
def build_library_from_file(f, builder):
    gen = lines(f)
    try:
        while 1:
            line = strip_comment(next(gen))
            if line == '':
                continue
            elif line == '$end':
                break
            elif line == '$ecp':
                print('Stopping library reading, because ecp input not implemented.')
                break
            elif line == '$basis':
                basis_input(gen, builder)
                break
            else:
                raise RuntimeError('Unexpected syntax: %s' % line)
    except StopIteration:
        raise RuntimeError('Unexpected end of file. Missing ''$end''?')

    return builder.get_library()

def build_library(filename, format = None):
    assert format is not None
    assert format == 'turbomole'
    builder = LibraryBuilder()
    with open(filename,'r') as f:
        return build_library_from_file(f, builder)

if __name__ == '__main__':
    build_library('/home/niflheim/kuisma/bases/h', format='turbomole')
