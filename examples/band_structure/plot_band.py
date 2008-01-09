from pylab import *

file = open('Na_bands.dat', 'r')
lines = file.readlines()
xs = []
ys = []
for line in lines:
    cols=line.split()
    xs.append(eval(cols[0]))
    ys.append(eval(cols[1]))

plot(xs, ys, '.m')
#axis([None,None,-5,8])
xlabel('Kpoint', fontsize=22)
ylabel('Eigenvalue', fontsize=22)
show()
