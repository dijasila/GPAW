import pickle
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
symbols = chemical_symbols[1:87]
D = pickle.load(open('data.pckl'))
fd1 = open('bulk.csv', 'w')
fd1.write(', `F`, `\Delta F`, `\Delta C^{FCC}`, `\Delta C^{RS}`?, ' +
          '`a^{FCC}`, `\Delta a^{FCC}`, `a^{RS}?`, `\Delta a^{RS}`?\n')
fd2 = open('relconv.csv', 'w')
fd2.write(', ' + ', '.join('%d' % e for e in range(300, 551, 50)) + '\n')
fd3 = open('absconv.csv', 'w')
fd3.write(', ' + ', '.join('%d' % e for e in range(300, 601, 50)) + '\n')
fd4 = open('lcao.csv', 'w')
fd4.write(', `\Delta F`, `\Delta C^{FCC}`, `\Delta C^{RS}`, ' +
          '`\Delta a^{FCC}`, `\Delta a^{RS}`, `\Delta a^{FCC}`, `\Delta a^{RS}`\n')
fd5 = open('egg.csv', 'w')
for s in symbols:
    for t in ['std', 'sc']:
        d = D.get((s, t))
        if d is None:
            continue
        locals().update(d)
        fd1.write('%s\ :sub:`%d`' % (s, e))
        fd2.write('%s\ :sub:`%d`' % (s, e))
        fd3.write('%s\ :sub:`%d`' % (s, e))
        fd4.write('%s\ :sub:`%d`' % (s, e))
        fd5.write('%s\ :sub:`%d`' % (s, e))
        for x in [fe0, dfenr, dcenrF, dcenrR,
                  anrF0, anrF - anrF0, anrR0, anrR - anrR0]:
            fd1.write(', %.3f' % x)
        for x in dde[:-1]:
            fd2.write(', %.3f' % x)
        for x in de:
            fd3.write(', %.3f' % x)
        for x in [dferL, dcerFL, dcerRL,
                  arFL - arF, arRL - arR, arF - arF0, arR - arR0]:
            fd4.write(', %.3f' % x)
        for x in egg:
            fd5.write(', %.3f' % x)
        fd1.write('\n')
        fd2.write('\n')
        fd3.write('\n')
        fd4.write('\n')
        fd5.write('\n')


#    for x in ('arF0 anrF0 anrF arF arFL arR0 anrR0 anrR arR arRL ' +
#              'fe dfenr dferL ceF dcenrF dcerFL ceR dcenrR dcerFL ' +
#              'de dde egg es').split():

from table import table
fig = table(dict((key, d['dfenr']) for key, d in D.items()),
            r'$|\Delta F|$ [eV]')
fig.savefig('dfe.png')
fig = table(dict((key, d['dcenrF']) for key, d in D.items()),
            r'$|\Delta C^{FCC}|$ [eV]')
fig.savefig('dce.png')
fig = table(dict((key, d['anrF'] - d['anrF0']) for key, d in D.items()),
            r'$|\Delta a^{FCC}|$ [Ang]')
fig.savefig('a.png')

fig = table(dict((key, d['dferL']) for key, d in D.items()),
            r'$|\Delta F|$ [eV]')
fig.savefig('dfelcao.png')
fig = table(dict((key, abs(d['dcerFL']) + abs(d['dcerRL']))
                 for key, d in D.items()),
            r'$|\Delta C^{FCC}| + |\Delta C^{RS}|$ [eV]')
fig.savefig('dcelcao.png')
fig = table(dict((key, abs(d['arFL'] - d['arF']) +
                  abs(d['arRL'] - d['arR']))
                 for key, d in D.items()),
            r'$|\Delta a^{FCC}| + |\Delta a^{RS}|$ [Ang]')
fig.savefig('alcao.png')

fig = table(dict((key, d['dde'][0]) for key, d in D.items()),
            r'$|\Delta E(350 eV) - \Delta E(600 eV)|$ [eV]')
fig.savefig('conv.png')
fig = table(dict((key, d['egg'][0]) for key, d in D.items()),
            r'$|\Delta E(0.2 Ang)|$ [meV]')
fig.savefig('egg.png')
