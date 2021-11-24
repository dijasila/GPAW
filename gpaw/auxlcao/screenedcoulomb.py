# This file is testing spherical integration of screened coulomb

import numpy as np
from gpaw.sphere.lebedev import weight_n, Y_nL, R_nv
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from scipy.special import erfc
import matplotlib.pyplot as plt

gs = range(1, 300, 50)
omega = 0.11
N = 1200
h = 0.01
# Plot below is used to pick non-divergent radial direction to turn the other grid to
# Average of points 23, 31, 16 is chosen as direction
Rdir_v = np.mean(R_nv[[23,31,16],:],axis=0)
Q_vv, _ = np.linalg.qr(np.array([ Rdir_v]).T,'complete')
R2_nv = R_nv @ Q_vv
if 0:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(R_nv[:,0], R_nv[:,1], R_nv[:,2])
    for n, R_v in enumerate(R_nv):
        ax.text(R_v[0], R_v[1], R_v[2], str(n))
    ax.scatter(Rdir_v[0], Rdir_v[1], Rdir_v[2],'o')
    plt.show()
    
rgd = EquidistantRadialGridDescriptor(h, N)

V_gg = np.zeros( (N, N) )

def solve(r, L=0, L2 =0):
    v_g = np.zeros((N,))
    for n2, R2_v in enumerate(R2_nv):
        for g, r2 in enumerate(rgd.r_g):
            D_n = np.sum((R_nv*r - r2*R2_v[None, :])**2, axis=1)**0.5
            D2_n = np.where(D_n < h, h, D_n)
            V_n = erfc(D_n*omega) / D2_n
            #V_n = 1.0 / D2_n
            v_g[g] += weight_n[n2] * Y_nL[n2, L2] * np.sum(weight_n * Y_nL[:, L] * V_n)
    return 4*np.pi*v_g

l = 1
L = 1

for g in gs:
    print(g)
    V_gg[g,:] = solve(rgd.r_g[g], L=L, L2=L)

#eigs, vectors = np.linalg.eig(V_gg)
#print(eigs)
#print(vectors)


def Phi0(Xi, xi):
    return -1/(2*np.pi**0.5*xi*Xi) * ( ( np.exp(-(xi+Xi)**2) - np.exp(-(xi-Xi)**2 ) \
        -np.pi**0.5*((xi-Xi)*erfc(Xi-xi)+(Xi+xi)*erfc(xi+Xi))))

def Phi1(Xi, xi):
    Phi = -1/(2*np.pi**0.5*xi**2*Xi**2)
    Phi *= 1/2*(( np.exp(-(xi+Xi)**2) - np.exp(-(xi-Xi)**2 ) ) \
        * (2*xi**2 + 2*xi*Xi-(1-2*Xi**2)) - 4*xi*Xi*np.exp(-(xi+Xi)**2) ) -np.pi**0.5*((xi**3-Xi**3)* \
           erfc(Xi-xi)+(xi**3+Xi**3)*erfc(xi+Xi))
    #Phi *= ( 1/2*(( np.exp(-xi**2-2*xi*Xi-Xi**2) - np.exp(-xi**2+2*xi*Xi-Xi**2) \
    #    * (2*xi**2 + 2*xi*Xi-(1-2*Xi**2)) - 4*xi*Xi*np.exp(-(xi+Xi)**2) ) -np.pi**0.5*((xi**3-Xi**3)* \
    #       erfc(Xi-xi)+(xi**3+Xi**3)*erfc(xi+Xi))))
    return Phi / (4*np.pi)**0.5

def F0(R,r,mu):
    return mu*Phi0(mu*R, mu*r)

def F1(R,r,mu):
    return mu*Phi1(mu*R, mu*r)

F = [ F0, F1 ]
Vord_gg = np.zeros( (N, N) )
for g in gs:
    r1 = rgd.r_g[g]
    for g2, r2 in enumerate(rgd.r_g):
        rmin = min(r1,r2)
        rmax = max(r1,r2)
        #Vord_gg[g,g2] = rmin**l / rmax**(l+1)
        Vord_gg[g,g2] = F[l](rmax, rmin, omega)

for g in gs:
    plt.plot(rgd.r_g, V_gg[g,:],'r')
    plt.plot(rgd.r_g, Vord_gg[g,:],'k--')
#plt.ylim([0,1])
plt.savefig('coulomb.png')
plt.show()
