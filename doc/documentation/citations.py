# creates: citations.png citations.csv

import datetime

import matplotlib
matplotlib.use('Agg')
import pylab as plt


months = [datetime.date(2000, m, 1).strftime('%B')[:3].upper()
          for m in range(1, 13)]


def f(filename):
    papers = []
    lines = open(filename).readlines()
    n = 0
    while n < len(lines):
        line = lines[n]
        tag = line[:2]
        if tag == 'TI':
            ntitle = n
        elif tag == 'SO':
            title = ' '.join(lines[i][3:-1] for i in range(ntitle, n))
        elif tag == 'DI':
            doi = line[3:-1]
        elif tag == 'PD':
            w = line.split()[1:]
            if len(w) == 1:
                date = datetime.date(int(w[0]), 6, 15)
            elif len(w) == 2:
                date = datetime.date(int(w[1]), 1 + months.index(w[0]), 15)
            else:
                date = datetime.date(int(w[2]), 1 + months.index(w[0]),
                                     int(w[1]))
            papers.append((date, doi, title))
        n += 1

    papers.sort()
    return papers

plt.figure(figsize=(10, 5))
total = {}
for bib in ['gpaw1', 'tddft', 'gpaw2', 'response']:
    papers = f(bib + '.txt')
    plt.plot([paper[0] for paper in papers], range(1, len(papers) + 1),
             '-o', label=bib)
    for date, doi, title in papers:
        total[doi] = (date, title)
    x = dict([(p[1], 0) for p in papers])
    print(bib, len(papers), len(x), len(total))

allpapers = [(paper[0], doi, paper[1]) for doi, paper in total.items()]
allpapers.sort()
plt.plot([paper[0] for paper in allpapers], range(1, len(allpapers) + 1),
             '-o', label='total')

fd = open('citations.csv', 'w')
n = len(allpapers)
for date, doi, title in allpapers[::-1]:
    fd.write('%d,"`%s <http://dx.doi.org/%s>`__"\n' % (n, title, doi))
    n -= 1
fd.close()

plt.xlabel('date')
plt.ylabel('number of citations')
plt.legend(loc='upper left')
plt.savefig('citations.png')
