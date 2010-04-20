#!/usr/bin/env python
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as pl
import tempfile

"""Database structure:
dict(testname: [(rev, runtime, info), (rev, runtime, info), ...])
    rev: SVN revision
    runtime: Run time in seconds. Negative for crashed jobs!
    info: A string describing the outcome
"""

class DatabaseHandler:
    """Database class for keeping timings and info for long tests"""
    def __init__(self, filename):
        self.filename = filename
        self.data = dict()

    def read(self):
        if os.path.isfile(self.filename):
            self.data = pickle.load(file(self.filename))
        else:
            print 'File does not exist, starting from scratch'

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
        if os.path.isfile(filename):
            os.rename(filename, filename + '.old')
        pickle.dump(self.data, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)

    def add_data(self, name, rev, runtime, info):
        if not self.data.has_key(name):
            self.data[name] = []
        self.data[name].append((rev, runtime, info))

    def get_data(self, name):
        """Return rev, time_array"""
        revs, runtimes = [], []
        if self.data.has_key(name):
            for datapoint in self.data[name]:
                revs.append(datapoint[0])
                runtimes.append(datapoint[1])

        return np.asarray(revs), np.asarray(runtimes)

    def update(self, queue, rev):
        """Add all new data to database"""
        for job in queue.jobs:
            absname = job.absname

            tstart = job.tstart
            if tstart is None:
                tstart = np.nan
            tstop = job.tstop
            if tstop is None:
                tstop = np.nan

            info = job.status

            self.add_data(absname, rev, tstop - tstart, info)

class TestAnalyzer:
    def __init__(self, name, revs, runtimes):
        self.name = name
        self.revs = revs
        self.runtimes = runtimes
        self.better = []
        self.worse = []
        self.relchange = None
        self.abschange = None

    def analyze(self, reltol=0.1, abstol=5.0):
        """Analyze timings

        When looking at a point, attention is needed if it deviates more than
        10\% from the median of previous points. If such a point occurs the
        analysis is restarted.
        """
        self.better = []
        self.worse = []
        abschange = 0.0
        relchange = 0.0
        status = 0
        current_first = 0   # Point to start analysis from
        for i in range(1, len(self.runtimes)):
            tmpruntimes = self.runtimes[current_first:i]
            median = np.median(tmpruntimes[np.isfinite(tmpruntimes)])
            if np.isnan(median):
                current_first = i
            elif np.isfinite(self.runtimes[i]):
                abschange = self.runtimes[i] - median
                relchange = abschange / median
                if relchange < -reltol and abschange < -abstol:
                    # Improvement
                    current_first = i
                    self.better.append(i)
                    status = -1
                elif relchange > reltol and abschange > abstol:
                    # Regression 
                    current_first = i
                    self.worse.append(i)
                    status = 1
                else:
                    status = 0

        self.status = status
        self.abschange = abschange
        self.relchange = relchange * 100

    def plot(self, outputdir=None):
        if outputdir is None:
            return
        fig = pl.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.revs, self.runtimes, 'ko-')
        ax.plot(self.revs[self.better],
                self.runtimes[self.better],
                'go', markersize=8)
        ax.plot(self.revs[self.worse],
                self.runtimes[self.worse],
                'ro', markersize=8)
        ax.set_title(self.name)
        if not outputdir.endswith('/'):
            outputdir += '/'
        figname = self.name.replace('/','_')
        fig.savefig(outputdir + figname + '.png')

class MailGenerator:
    def __init__(self):
        self.better = []
        self.worse = []

    def add_test(self, name, abschange, relchange):
        if abschange < 0.0:
            self.add_better(name, abschange, relchange)
        else:
            self.add_worse(name, abschange, relchange)

    def add_better(self, name, abschange, relchange):
        self.better.append((name, abschange, relchange))

    def add_worse(self, name, abschange, relchange):
        self.worse.append((name, abschange, relchange))

    def generate_mail(self):
        mail = 'Results from weekly tests:\n\n'
        if len(self.better):
            mail += 'The following tests improved:\n'
            for test in self.better:
                mail += '%-40s %7.2f s (%7.2f%%)\n' % test
        else:
            mail += 'No tests improved!\n'
        mail += '\n'
        if len(self.worse):
            mail += 'The following tests regressed:\n'
            for test in self.worse:
                mail += '%-40s +%6.2f s (+%6.2f%%)\n' % test
        else:
            mail += 'No tests regressed!\n'

        return mail

    def send_mail(self, address):
        fullpath = tempfile.mktemp()
        f = open(fullpath, 'w')
        f.write(self.generate_mail())
        f.close()
        os.system('mail -s "Results from weekly tests" %s < %s' % \
                  (address, fullpath))

#def csv2database(infile, outfile):
#    """Use this file once to import the old data from csv"""
#    csvdata = np.recfromcsv(infile)
#    db = DatabaseHandler(outfile)
#    for test in csvdata:
#        name = test[0]
#        for i in range(1, len(test) - 1):
#            runtime = float(test[i])
#            info = ''
#            db.add_data(name, 0, runtime, info)
#    db.write()

def analyse(queue, dbpath, outputdir=None, rev=None, mailto=None):
    """Analyse runtimes from testsuite

    Parameters:
        queue: AGTSQueue
            Que to analuze
        dbpath: str
            Path to file storing previous results
        outputdir: str|None
            If str, figures will be put in this dir
        rev: int|None
            GPAW revision. If None time.time() is used
        mailto: str|None
            Mailaddres to send results to. If None, results will be printed to
            stdout.
    """
    if rev is None:
        import time
        rev = time.time()
    db = DatabaseHandler(dbpath)
    db.read()
    db.update(queue, rev)
    db.write()
    mg = MailGenerator()
    for job in queue.jobs:
        name = job.absname
        revs, runtimes = db.get_data(name)
        ta = TestAnalyzer(name, revs, runtimes)
        ta.analyze(abstol=0)
        if ta.status:
            mg.add_test(name, ta.abschange, ta.relchange)
        ta.plot(outputdir)

    if mailto is not None:
        mg.send_mail(mailto)
    else:
        print mg.generate_mail()
