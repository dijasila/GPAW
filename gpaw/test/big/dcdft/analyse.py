import os

import sys

import urllib
import urllib2

import tarfile
import zipfile

import csv

from ase.test import NotAvailable

from ase import units

from ase.test.tasks.dcdft import DeltaCodesDFTTask as Task

if len(sys.argv) == 1:
    tag = None
else:
    tag = sys.argv[1]

task = Task(
    tag=tag,
    use_lock_files=True,
    )

task.read()
task.analyse()

csvwriter1 = csv.writer(open('%s_raw.csv' % tag, 'wb'))
# header
h = ['#element', 'V0', 'B0', 'B1']
csvwriter1.writerow(h)
csvwriter2 = csv.writer(open('%s.csv' % tag, 'wb'))
h2 = h + ['%' + h[1], '%' + h[2], '%' + h[3]]
csvwriter2.writerow(h2)

rows = []
rowserr = []
for n in task.collection.names:
    row = [n]
    if n in task.data.keys():
        ref = task.collection.ref[n]
        try:
            v = round(task.data[n]['dcdft volume'], 3)
            b0 = round(task.data[n]['dcdft B0'] / (units.kJ * 1e-24), 3)
            b1 = round(task.data[n]['dcdft B1'], 3)
            row.extend([v, b0, b1])
        except KeyError: # completely failed to find eos minimum
                row.extend(['N/A', 'N/A', 'N/A'])
    else:
        # element not calculated
        row.extend(['N/A', 'N/A', 'N/A'])
    if 'N/A' not in row:
        v0, b00, b10 = ref
        ve = round((v - v0) / v0 * 100, 1)
        b0e = round((b0 - b00) / b00 * 100, 1)
        b1e = round((b1 - b10) / b10 * 100, 1)
        rows.append(row)
        #print row + ref + [ve, b0e, b1e]
        csvwriter1.writerow(row)
        csvwriter2.writerow(row + [ve, b0e, b1e])

if 1:
    # download and create the project databases
    src = 'http://molmod.ugent.be/sites/default/files/Delta_v1-0.zip'
    name = os.path.basename(src)
    dir = 'Delta'
    if not os.path.exists(dir): os.makedirs(dir)
    os.chdir(dir)
    try:
        resp = urllib2.urlopen(src)
        urllib.urlretrieve(src, filename=name)
        z = zipfile.ZipFile(name)
        try:  # new in 2.6
            z.extractall()
        except AttributeError:
            # http://stackoverflow.com/questions/7806563/how-to-unzip-a-zip-file-with-python-2-4
            for f in z.namelist():
                fd = open(f, "w")
                fd.write(z.read(f))
                fd.close()
        urllib.urlretrieve('http://dcwww.camd.dtu.dk/~dulak/calcDelta.py',
                           filename='calcDelta.py')
        # AttributeError if unzip not found
    except (urllib2.HTTPError, AttributeError):
        raise NotAvailable('Retrieval of zip failed')
    os.chdir('..')

    # calculate Delta
    f = open('%s.txt' % tag, 'wb')
    csvwriter3 = csv.writer(f, delimiter='\t')
    for r in rows:
        csvwriter3.writerow(r)
    f.close()
    cmd = 'python ' + os.path.join(dir, 'calcDelta.py')
    cmd += ' ' + '%s.txt ' % tag + os.path.join(dir, 'WIEN2k.txt') + ' --full --stdout'
    cmd += ' > ' + '%s_Delta.txt' % tag
    os.system(cmd)
