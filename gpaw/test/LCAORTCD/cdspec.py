import numpy as np
from ase.units import Hartree, alpha, Bohr
from sys import argv
from gpaw.tddft.demo import read_dipole_moment_file

kick_i, time_t, dm_tv=read_dipole_moment_file(argv[1])

print(kick_i)

str_c=kick_i[0]['strength_v']
omega_w=np.arange(0,20,0.01)
dt=time_t[1]-time_t[0]
delta=(0.12/Hartree)**2*0.5

f=open(argv[1]+'.spec','w')

for omega in omega_w:

     betas=[]
     ecds=[]
     for k in range(3):

         beta=1/(2*omega/Hartree*np.linalg.norm(str_c))*np.sum(np.cos(omega/Hartree*time_t)*dm_tv[:,k]*np.exp(-delta*time_t**2))*dt
         betas.append(beta)
         ecd=-np.pi*(1/np.linalg.norm(str_c))*np.sum(np.cos(omega/Hartree*time_t)*dm_tv[:,k]*np.exp(-delta*time_t**2))*dt
         ecds.append(ecd)
     
     betas=np.array(betas)

     ecds=np.array(ecds)

     R=np.sum(3*omega/Hartree/(np.pi)*alpha*betas*Bohr**3*str_c/np.linalg.norm(str_c))

     ER=np.sum(ecds*str_c/np.linalg.norm(str_c))


     print('%20.15f %20.15f %20.15f %20.15f %20.15f'% (omega, ER, betas[0], betas[1], betas[2]), file=f)

print(np.exp(-delta*time_t[-1]**2))
   
f.close()
