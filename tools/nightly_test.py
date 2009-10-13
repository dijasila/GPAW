#!/usr/bin/python

import os
import sys
import time
import glob
import trace
import tempfile


def send_email(subject, filename='/dev/null'):
    assert os.system('mail -s "%s" gridpaw-developer@lists.berlios.de < %s' %
    #assert os.system('mail -s "%s" jensj@fysik.dtu.dk < %s' %
                     (subject, filename)) == 0

def send_jj_email(subject, filename='/dev/null'):
    assert os.system('mail -s "%s" jensj@fysik.dtu.dk < %s' %
                     (subject, filename)) == 0

def fail(msg, filename='/dev/null'):
    send_email(msg, filename)
    raise SystemExit

tmpdir = tempfile.mkdtemp(prefix='gpaw-')
os.chdir(tmpdir)

day = time.localtime()[6]

# Checkout a fresh version and install:
if os.system('svn checkout ' +
             'https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw') != 0:
    fail('Checkout of gpaw failed!')

if day % 2:
    d = {}
    execfile('gpaw/gpaw/version.py', d)
    asesvnrevision = d['ase_required_svnrevision']
else:
    asesvnrevision = 'HEAD'

if os.system('svn checkout ' +
             'https://svn.fysik.dtu.dk/projects/ase/trunk ase -r %s' %
             asesvnrevision) != 0:
    fail('Checkout of ASE failed!')
try: 
    # subprocess was introduced with python 2.4
    from subprocess import Popen, PIPE
    cmd = Popen('svnversion ase',
                shell=True, stdout=PIPE, stderr=PIPE, close_fds=True).stdout
except ImportError:
    cmd = popen3('svnversion ase')[1] # assert that we are in gpaw project
aserevision = int(cmd.readline())
cmd.close()

os.chdir('gpaw')

try: 
    # subprocess was introduced with python 2.4
    from subprocess import Popen, PIPE
    cmd = Popen('svnversion', 
                shell=True, stdout=PIPE, stderr=PIPE, close_fds=True).stdout
except ImportError:
    cmd = popen3('svnversion')[1] # assert that we are in gpaw project
gpawrevision = int(cmd.readline().strip('M\n'))
cmd.close()

if os.system('python setup.py install --home=%s ' % tmpdir +
             '2>&1 | grep -v "c/libxc/src"') != 0:
    fail('Installation failed!')

os.system('mv ../ase/ase ../lib/python')

os.system('wget --no-check-certificate --quiet ' +
          'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')

os.system('tar xvzf gpaw-setups-latest.tar.gz')

setups = tmpdir + '/gpaw/' + glob.glob('gpaw-setups-[0-9]*')[0]
sys.path.insert(0, '%s/lib/python' % tmpdir)

if day % 4 < 2:
    sys.argv.append('--debug')

from gpaw import setup_paths
setup_paths.insert(0, setups)

# Run test-suite:
from gpaw.test import TestRunner, tests
os.mkdir('gpaw-test')
os.chdir('gpaw-test')
out = open('test.out', 'w')
failed = TestRunner(tests, stream=out).run()
out.close()
if failed:
    # Send mail:
    n = len(failed)
    if n == 1:
        subject = 'One failed test: ' + failed[0][:-1]
    else:
        subject = ('%d failed tests: %s, %s' %
                   (n, failed[0][:-1], failed[1][:-1]))
        if n > 2:
            subject += ', ...'
    fail(subject, 'test.out')

open('/home/camp/jensj/gpawrevision.ok', 'w').write('%d %d\n' %
                                                    (aserevision,
                                                     gpawrevision))

if 0:
    # PyLint:
    os.chdir('../../lib/python')
    os.system('rm -rf gpaw/gui')
    if os.system(export + 'pylint -f html gpaw; ' +
                 'cp pylint_global.html %s' % dir) != 0:
        fail('PyLint failed!')

def count(dir, pattern):
    p = os.popen('wc -l `find %s -name %s` | tail -1' % (dir, pattern), 'r')
    return int(p.read().split()[0])

os.chdir('..')
libxc = count('c/libxc', '\\*.[ch]')
ch = count('c', '\\*.[ch]') - libxc
test = count('gpaw/test', '\\*.py')
py = count('gpaw', '\\*.py') - test

import pylab
# Update the stat.dat file:
dir = '/scratch/jensj/nightly-test/'
f = open(dir + 'stat.dat', 'a')
print >> f, pylab.epoch2num(time.time()), libxc, ch, py, test
f.close()

# Construct the stat.png file:
lines = open(dir + 'stat.dat').readlines()
date, libxc, c, code, test = zip(*[[float(x) for x in line.split()]
                                   for line in lines[1:]])
date = pylab.array(date)
code = pylab.array(code)
test = pylab.array(test)
c = pylab.array(c)

def polygon(x, y1, y2, *args, **kwargs):
    x = pylab.concatenate((x, x[::-1]))
    y = pylab.concatenate((y1, y2[::-1]))
    pylab.fill(x, y, *args, **kwargs)

fig = pylab.figure()
ax = fig.add_subplot(111)
polygon(date, code + test, code + test + c,
        facecolor='r', label='C-code')
polygon(date, code, code + test,
        facecolor='y', label='Tests')
polygon(date, [0] * len(date), code,
        facecolor='g', label='Python-code')
polygon(date, [0] * len(date), [0] * len(date),
        facecolor='b', label='Fortran-code')

months = pylab.MonthLocator()
months3 = pylab.MonthLocator(interval=3)
month_year_fmt = pylab.DateFormatter("%b '%y")

ax.xaxis.set_major_locator(months3)
ax.xaxis.set_minor_locator(months)
ax.xaxis.set_major_formatter(month_year_fmt)
labels = ax.get_xticklabels()
pylab.setp(labels, rotation=30)
pylab.axis('tight')
pylab.legend(loc='upper left')
pylab.title('Number of lines')
pylab.savefig(dir + 'stat.png')

"""
# Coverage test:
os.chdir('test')
if day == 6:  # only Sunday
    if os.system(export +
                 'rm %s/*.cover; ' % dir +
                 'python %s/trace.py' % os.path.dirname(trace.__file__) +
                 ' --count --coverdir coverage --missing' +
                 ' --ignore-dir /usr:%s test.py %s' % (home, args)) != 0:
        fail('Coverage failed!')
    
    filenames = glob.glob('coverage/gpaw.*.cover')
else:
    filenames = glob.glob(dir + '*.cover')
    
names = []
for filename in filenames:
    missing = 0
    for line in open(filename):
        if line.startswith('>>>>>>'):
            missing += 1
    if missing > 0:
        if filename.startswith('coverage/gpaw.'):
            name = filename[14:-6]
            if os.system('cp %s %s/%s.cover' %
                         (filename, dir, name)) != 0:
                fail('????')
        else:
            name = filename[28:-6]
        names.append((-missing, name))

"""
os.system('cd; rm -r ' + tmpdir)

