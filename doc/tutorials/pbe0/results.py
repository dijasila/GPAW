from __future__ import division
import matplotlib.pyplot as plt
import ase.db
from ase.utils.eos import EquationOfState


def lattice_constant(volumes, energies):
    eos = EquationOfState(volumes, energies)
    v, e, B = eos.fit()
    a = (v * 4)**(1 / 3)
    return a
    
    
con = ase.db.connect('si.db')
results = []
K = list(range(2, 9))
A = []
A0 = []
for k in K:
    rows = list(con.select(k=k))
    V = [row.volume for row in rows]
    E = [row.energy for row in rows]
    E0 = [row.epbe0 for row in rows]
    A.append(lattice_constant(V, E))
    A0.append(lattice_constant(V, E0))


LDA = dict(
    a=5.4037,
    B=95.1,
    eGX=0.52)
PBE = dict(
    a=5.469,
    B=87.8,
    eGG=2.56,
    eGX=0.71,
    eGL=1.54,
    eI=0.47,  # indirect
    ea=4.556)
PBE0 = dict(
    a=5.433,
    B=99.0,
    eGG=3.96,
    eGX=1.93,
    eGL=2.87,
    eI=1.74,
    ea=4.555)

plt.plot(K, A, label='PBE')
plt.plot(K, A0, label='PBE0')
plt.xlabel('number of k-points')
plt.ylabel('lattice constant [Ang]')
plt.savefig('a.png')
