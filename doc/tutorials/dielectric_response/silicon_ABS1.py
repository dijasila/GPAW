import numpy as np
from gpaw import GPAW
from gpaw.response.df2 import DielectricFunction

w = np.linspace(0, 24., 481)    # 0-24 eV with 0.05 eV spacing

df = DielectricFunction(calc='si.gpw',
                        frequencies=w,
                        eta=0.1,           # Broadening parameter 
                        ecut=150,          # Energy cutoff for planewaves
                        txt='df_2.out')    # Output text

df.get_polarizability(filename='si_abs.csv') 
