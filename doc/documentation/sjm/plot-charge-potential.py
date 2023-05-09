# web-page: charge-potential.png

import numpy as np
from matplotlib import pyplot
from scipy.stats import linregress

fig, ax = pyplot.subplots()
fig.subplots_adjust(top=0.99, right=0.99)

potentials, charges = [], []
with open('potential.txt', 'r') as f:
    lines = f.read().splitlines()
for line in lines:
    potential, charge = line.split(',')
    potentials += [float(potential)]
    charges += [float(charge)]

line = linregress(charges, potentials)
x_line = [min(charges), max(charges)]
y_line = [line.slope * _ + line.intercept for _ in x_line]

ax.plot(x_line, y_line, '-', color='C1')
ax.plot(charges, potentials, 'o', color='C0')
ax.text(np.mean(x_line) + 0.0 * np.ptp(x_line),
        np.mean(y_line) + 0.1 * np.ptp(y_line),
        '$R^2 = {:.4f}$'.format(line.rvalue**2))

ax.set_xlabel('excess electrons')
ax.set_ylabel('potential, V')
fig.savefig('charge-potential.png')
