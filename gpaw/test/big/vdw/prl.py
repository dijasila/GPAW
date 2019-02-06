import pickle
import pylab as plt
import numpy as np
fromm ase.units import kcal, mol

#VDWFunctional().make_prl_plot()

d = np.linspace(3.0, 5.5, 11)
S = ['Ar', 'Kr']
S = ['Kr']
for symbol in S:
    d, e, de = pickle.load(open(symbol + '.pckl', 'rb'))
    #plt.plot(d, e * 1000)
    plt.plot(d, (e + de) * 1000, label=symbol)
for symbol in S:
    e, de, e0, de0 = pickle.load(open(symbol + '.new.pckl', 'rb'))
    #plt.plot(d, (e-2*e0) * 1000)
    plt.plot(d, (e + de-2*e0-2*de0) * 1000, label=symbol+'.new')
for symbol in S:
    e, de, e0, de0 = pickle.load(open(symbol + '.fft.pckl', 'rb'))
    plt.plot(d, (e-2*e0) * 1000)
    plt.plot(d, (e + de-2*e0-2*de0) * 1000, label=symbol+'.fft')
plt.legend()
plt.ylim(-25, 50)
plt.show()



d, e, de = pickle.load(open('benzene.pckl', 'rb'))
plt.plot(d, e / (kcal/mol))
plt.plot(d, (e + de) / (kcal/mol))
e, de, e0, de0 = pickle.load(open('benzene.fft.pckl', 'rb'))
plt.plot(d, (e-2*e0) / (kcal/mol))
plt.plot(d, (e + de - 2 * e0 - 2 * de0) / (kcal/mol))
plt.xlim(3, 4.9)
plt.ylim(-4, 5)
plt.show()

