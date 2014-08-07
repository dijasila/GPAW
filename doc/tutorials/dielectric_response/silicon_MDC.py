import numpy as np
from gpaw.reponse.df2 import DielectricFunction

w = np.linspace(0, 24, 481)

# getting macroscopic constant
df = DielectricFunction(calc='si.gpw', omega_w=w, eta=0.0001,
                        ecut=150, txt='df_1.out')

df.get_macroscopic_dielectric_constant()
