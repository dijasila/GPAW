#!/usr/bin/python

import os
import glob
import pylab
import datetime

date1 = datetime.date(2005, 10, 19)
date2 = datetime.date(2008,  7,  7)
date1 = datetime.date(2008, 1, 19)
date2 = datetime.date(2008,  7,  7)
#date2 = datetime.date(2005,  10, 21)
delta = datetime.timedelta(days=100)
dates = pylab.drange(date1, date2, delta)

def count(dir, pattern):
    if not os.path.isdir(dir):
        return 0
    p = os.popen('wc -l `find %s -name %s` | tail -1' % (dir, pattern), 'r')
    return int(p.read().split()[0])
dates = [date1, date2]
stat = open('stat2.dat', 'w')
dates = [datetime.date(y, m, 1) for y in range(2005,2008) for m in range(1,13)]
dates = [datetime.date(2008, m, d) for m in range(1,8) for d in range(1,30,5)]
for date in dates:
    print date
    # Checkout of relevant gpaw folders
    svn = ('svn export --revision {%s} '
           'https://svn.fysik.dtu.dk/projects/gpaw/trunk' % date.isoformat())
    e = os.system(svn + ' temp-gpaw > /dev/null')
    if e != 0:
        os.system('rm -rf temp-gpaw')
        continue
    # Run PyLint:
    os.system('rm -rf temp-gpaw/gpaw/gui')

    libxc = count('temp-gpaw/c/libxc', '\\*.[ch]')
    ch = count('temp-gpaw/c', '\\*.[ch]') - libxc
    py = count('temp-gpaw/gridpaw', '\\*.py')
    py += count('temp-gpaw/gpaw', '\\*.py')
    test = count('temp-gpaw/test', '\\*.py')

    # dump collected data to file, and clean up
    print date, libxc, ch, py, test
    print >> stat, pylab.date2num(date), libxc, ch, py, test
    os.system('rm -rf temp-gpaw')
