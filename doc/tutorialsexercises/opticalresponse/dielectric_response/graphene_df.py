from gpaw.response.df import DielectricFunction
from ase.units import Hartree, Bohr
from ase.utils import seterr
import matplotlib.pyplot as plt
import numpy as np

a = 2.5
c = 3.22
df = DielectricFunction('gsresponse.gpw',
                        eta=25e-3,
                        rate='eta',
                        frequencies={'type': 'nonlinear',
                                     'domega0': 0.01},
                        integrationmode='tetrahedron integration')
df1tetra, df2tetra = df.get_dielectric_function(q_c=[0, 0, 0], filename='df_tetra.csv')

df = DielectricFunction('gsresponse.gpw',
                        frequencies={'type': 'nonlinear',
                                     'domega0': 0.01},
                        eta=25e-3,
                        rate='eta')
df1, df2 = df.get_dielectric_function(q_c=[0, 0, 0], filename='df_point.csv')
omega_w = df.get_frequencies()

from gpaw.test import findpeak
w1, I1 = findpeak(df.wd.omega_w, -(1. / df2).imag)
print(w1, I1)

plt.figure(figsize=(6, 6))
plt.plot(omega_w, df2.imag * 2, label='Point sampling')
plt.plot(omega_w, df2tetra.imag * 2, label='Tetrahedron')
# Analytical result for graphene
sigmainter = 1 / 4.  # The surface conductivity of graphene
with seterr(divide='ignore', invalid='ignore'):
    dfanalytic = 1 + (4 * np.pi * 1j / (omega_w / Hartree) *
                      sigmainter / (c / Bohr))

plt.plot(omega_w, dfanalytic.imag, label='Analytic')

plt.xlabel('Frequency (eV)')
plt.ylabel('$\\mathrm{Im}\\varepsilon$')
plt.xlim(0, 6)
plt.ylim(0, 50)
plt.legend()
plt.tight_layout()
plt.savefig('graphene_eps.png', dpi=600)



