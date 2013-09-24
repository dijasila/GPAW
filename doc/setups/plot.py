import pickle
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
symbols = chemical_symbols[1:87]
D = pickle.load(open('data.pckl'))
fd1 = open('bulk.csv', 'w')
fd2 = open('relconv.csv', 'w')
fd3 = open('absconv.csv', 'w')
fd4 = open('lcao.csv', 'w')
fd5 = open('egg.csv', 'w')
fd6 = open('misc.csv', 'w')
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
        fd6.write('%s\ :sub:`%d`' % (s, e))
        for x in [fe0, dfenr, dcenrR,
                  anrF0, anrF - anrF0, anrR0, anrR - anrR0]:
            fd1.write(', %.3f' % x)
        for x in dde[:-1]:
            fd2.write(', %.3f' % x)
        for x in de:
            fd3.write(', %.3f' % x)
        for x in [dferL, dcerRL,
                  arFL - arF, arRL - arR]:
            fd4.write(', %.3f' % x)
        for x in egg:
            fd5.write(', %.3f' % x)
        for x in [es[0], es[1], arF - arF0, arR - arR0]:
            fd6.write(', %.3f' % x)
        fd1.write('\n')
        fd2.write('\n')
        fd3.write('\n')
        fd4.write('\n')
        fd5.write('\n')
        fd6.write('\n')


from table import table
fig = table(dict((key, d['dfenr']) for key, d in D.items()),
            r'$|\Delta F|$ [eV]')
fig.savefig('dfe.png')
print dict((key, d['dcenrR']) for key, d in D.items())
fig = table(dict((key, d['dcenrR']) for key, d in D.items()),
            r'$|\Delta C^{RS}|$ [eV]')
fig.savefig('dce.png')

fig = table(dict((key, abs(d['arF'] - d['arF0']))
                 for key, d in D.items()),
            r'$|\Delta a^{FCC}|$ [Ang]')
fig.savefig('aF.png')
fig = table(dict((key, abs(d['arR'] - d['arR0']))
                 for key, d in D.items()),
            r'$|\Delta a^{RS}|$ [Ang]')
fig.savefig('aR.png')
fig = table(dict((key, abs(d['anrF'] - d['anrF0']))
                 for key, d in D.items()),
            r'$|\Delta a^{FCC}|$ [Ang]')
fig.savefig('anrF.png')
fig = table(dict((key, abs(d['anrR'] - d['anrR0']))
                 for key, d in D.items()),
            r'$|\Delta a^{RS}|$ [Ang]')
fig.savefig('anrR.png')

fig = table(dict((key, d['dferL']) for key, d in D.items()),
            r'$|\Delta F|$ [eV]')
fig.savefig('dfelcao.png')

fig = table(dict((key, d['dcerRL'])
                 for key, d in D.items()),
            r'$|\Delta C^{RS}|$ [eV]')
fig.savefig('dcelcao.png')

fig = table(dict((key, d['arFL'] - d['arF'])
                 for key, d in D.items()),
            r'$|\Delta a^{FCC}|$ [Ang]')
fig.savefig('aFlcao.png')

fig = table(dict((key, d['arRL'] - d['arR'])
                 for key, d in D.items()),
            r'$|\Delta a^{RS}|$ [Ang]')
fig.savefig('aRlcao.png')

fig = table(dict((key, d['dde'][0]) for key, d in D.items()),
            r'$|\Delta E(350 eV) - \Delta E(600 eV)|$ [eV]')
fig.savefig('conv.png')
fig = table(dict((key, d['egg'][0]) for key, d in D.items()),
            r'$|\Delta E(0.2 Ang)|$ [meV]')
fig.savefig('egg.png')
