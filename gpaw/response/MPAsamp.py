#-------------------------------------------------------------------
# Pair sampling of a positive unidimentional (frequency) domain as a
# function of a scale parameter and the exponent of the distribution
# 
#                               to be used in the MPA interpolation*
#
# *DA. Leon et al, PRB 104, 115157 (2021)
#
# Notes:
#
#   1) Homogeneous (homo)
#   2) Linear Partition Pair Sampling (lPPS)
#   3) Quadratic Partition Pair Sampling (qPPS)
#   4) Cubic Partition Pair Sampling (cPPS)
#   5) Alpha Partition Pair Sampling (aPPS)
#
#   The samplings do not depend on the sampled function 
#-------------------------------------------------------------------

import numpy as np
from cmath import *

def mpa_frequency_sampling(npol, w0, d, ps='2l', alpha=1): #, w_grid
    # integer,      intent(in)  :: npol          # number of desired frequency pairs/poles
    # complex(SP),  intent(in)  :: w0(2)         # segment [w1,w2]
    # real(SP),     intent(in)  :: d(2)          # shifts
    # character(2), intent(in)  :: ps            # flavour of sampling: 1line, 2lines
    # real(2),      intent(in)  :: alpha         # type of grid: 0->homo, 1->lPPS
    # complex(SP),  intent(out) :: w_grid(2*np)  # grid of complex frequencies
      
    # Auxiliary variables
    # integer     :: i
    # integer(SP) :: lp,r,c,p
    # real(SP)    :: aux
    # complex(SP) :: ws
    # real(SP), parameter :: log2=0.693147180560_SP
  
    if(npol==1):
      w_grid = np.array(w0, dtype=complex)  #(/w1+epsilon(1._SP), w2/)    
    else:
      #match ps:
        #case '1l': 
          #action  
        #case '2l': 
        #case _:
          #action-default 
      if(ps=='1l'): 
        if(alpha==0):
          w_grid = np.linspace(w0[0],w0[1], 2*npol)
      elif(ps=='2l'): 
        if(alpha==0):
          w_grid = np.concatenate((np.linspace(complex(np.real(w0[0]),d[1]),complex(np.real(w0[1]),d[1]),npol), 
                                  np.linspace(w0[0],w0[1],npol)))
          w_grid[0] = complex(np.real(w0[0]),d[0])        
        else:
          #alpha=1
          ws = w0[1]-w0[0]
          w_grid = np.ones(2*npol, dtype=complex)      
          w_grid[0] = complex(np.real(w0[0]),d[0])
          w_grid[npol-1] = complex(np.real(w0[1]),d[1])
          w_grid[npol] = w0[0]
          w_grid[2*npol-1] = w0[1]      
          lp = int(np.log(npol-1)/np.log(2))
          r = int((npol-1)%(2**lp))
          print(r)
          if(r>0):
            for i in range(1,2*r):
              w_grid[npol+i]=w0[0]+ws*( i/2.**(lp+1) )**alpha
              w_grid[i]=complex(np.real(w_grid[npol+i]),d[1]) 
            for i in range(2*r,npol):
              w_grid[npol+i] = w0[0]+ws*( (i-r)/2.**(lp) )**alpha
              w_grid[i] = complex(np.real(w_grid[npol+i]),d[1])
          else:
            w_grid[npol+1] = w0[0]+ws/( 2.**(lp+1) )**alpha
            w_grid[1] = complex(np.real(w_grid[npol+1]),d[1]) 
            for i in range(2*r+2,npol-1):
              w_grid[npol+i] = w0[0]+ws*( (i-1-r)/2.**(lp) )**alpha
              w_grid[i] = complex(np.real(w_grid[npol+i]),d[1])  

    return w_grid
  
  
#------------- tests ------------- 
"""
print("npol=1, ps='1l', w1=0.1j, w2=1j, alpha=1:")
w_grid = mpa_frequency_sampling(1, [complex(0,0.1), complex(0,1)], [0.1,0.1], ps='1l', alpha=1)
print(w_grid.real)
print(w_grid.imag,'\n')

print("npol=1, ps='2l', w1=0.1j, w2=1j, alpha=0:")
w_grid = mpa_frequency_sampling(1, [complex(0,0.1), complex(0,1)], [0.1,0.1], ps='2l', alpha=0)
print(w_grid.real)
print(w_grid.imag,'\n')

print("npol=2, ps='1l', w1=0+1j, w2=2+1j, alpha=0:")
w_grid = mpa_frequency_sampling(2, [complex(0,1), complex(2,1)], [0.01,0.1], ps='1l', alpha=0)
print(w_grid.real)
print(w_grid.imag,'\n')

print("npol=2, ps='2l', w1=0+1j, w2=2+1j, alpha=1:")
w_grid = mpa_frequency_sampling(2, [complex(0,1), complex(2,1)], [0.01,0.1])
print(w_grid.real)
print(w_grid.imag,'\n')

print("npol=3, ps='2l', w1=0+1j, w2=2+1j, alpha=1:")
w_grid = mpa_frequency_sampling(3, [complex(0,1), complex(2,1)], [0.01,0.1])
print(w_grid.real)
print(w_grid.imag,'\n')

print("npol=4, ps='2l', w1=0+1j, w2=2+1j, alpha=1:")
w_grid = mpa_frequency_sampling(4, [complex(0,1), complex(2,1)], [0.01,0.1])
print(w_grid.real)
print(w_grid.imag,'\n')

print("npol=5, ps='2l', w1=0+1j, w2=2+1j, alpha=1:")
w_grid = mpa_frequency_sampling(5, [complex(0,1), complex(2,1)], [0.01,0.1])
print(w_grid.real)
print(w_grid.imag,'\n')

print("npol=6, ps='2l', w1=0+1j, w2=2+1j, alpha=1:")
w_grid = mpa_frequency_sampling(6, [complex(0,1), complex(2,1)], [0.01,0.1])
print(w_grid.real)
print(w_grid.imag,'\n')
"""
