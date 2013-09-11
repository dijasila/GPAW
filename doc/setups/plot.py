import pickle
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
symbols = chemical_symbols[1:87]
D = pickle.load(open('data.pckl'))
fd1 = open('bulk.csv', 'w')
fd1.write(', `FE`, `\Delta FE`, `\Delta CE^{FCC}`, `\Delta CE^{RS}`, ' +
          '`a^{FCC}`, `\Delta a^{FCC}`, `a^{RS}`, `\Delta a^{RS}`\n')
fd2 = open('conv.csv', 'w')
fd2.write(', `FE`, `\Delta FE`, `\Delta CE^{FCC}`, `\Delta CE^{RS}`, ' +
          '`a^{FCC}`, `\Delta a^{FCC}`, `a^{RS}`, `\Delta a^{RS}`\n')
for s in symbols:
    for t in ['std', 'sc']:
        d = D.get((s, t))
        if d is None:
            continue
        locals().update(d)
        fd1.write('%s\ :sub:`%d`' % (s, e))
        for x in [fe0, dfenr, dcenrF, dcenrR,
                  anrF0, anrF - anrF0, anrR0, anrR - anrR0]:
            fd1.write(', %.3f' % x)
        fd1.write('\n')


#    for x in ('arF0 anrF0 anrF arF arFL arR0 anrR0 anrR arR arRL ' +
#              'fe dfenr dferL ceF dcenrF dcerFL ceR dcenrR dcerFL ' +
#              'de dde egg es').split():

from table import table
fig = table(dict((key, d['dfenr']) for key, d in D.items()),
            r'$|\Delta FE|$ [eV]')
fig.savefig('dfe.png')
fig = table(dict((key, d['dcenrF']) for key, d in D.items()),
            r'$|\Delta CE^{FCC}|$ [eV]')
fig.savefig('dce.png')
