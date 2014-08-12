import numpy as np
#from gpaw.response.df import DielectricFunction as DF
from gpaw.response.df0 import DF
from ase.parallel import paropen

w_grid = np.linspace(5., 15., 1001)

df = DF(calc='LiF_fulldiag.gpw',
       xc='RPA',
optical_limit=True,
q=np.array([0.0001, 0., 0.]),         
w=w_grid,
eta=0.1,              # Broadening parameter 
nbands=60,       # Number of bands to consider in the summation for building chi
ecut=30,              # Energy cutoff for planewaves
txt='LiF_RPA_out.txt')      # Output text

df1_w, df2_w = df.get_dielectric_function()         # It returns the the dielectric function without and with local field effects)

# Txt files for saving the spectrum
Im_df_RPA_data = paropen('Im_df_RPA_data.txt', 'w') 
for iw in range(0,len(w_grid)-1):
   print >> Im_df_RPA_data, w_grid[iw], df2_w[iw].imag    # 1st column frequency grid, 2nd column Dielectric Function
