
from gpaw import GPAW, LCAO
from ase import Atoms
from scipy.linalg import eigh
from gpaw.lfc import LFC
from gpaw.utilities.tools import tri2full
from gpaw.basis_data import Basis, BasisFunction
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.special import binom as binomial

from scipy.special import factorial2


def fact2(n):
    return factorial2(n, exact=True)

def product_center_1D(alphaa, xa, alphab, xb):
    return (alphaa*xa+alphab*xb)/(alphaa+alphab)

def dist2(x1,y1,z1,x2,y2,z2):
    return (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)

def binomial_prefactor(s, ia, ib, xpa, xpb):
    sum = 0.
    for t in range(s+1):
        if (s-ia <= t) and (t<=ib):
            sum += binomial(ia,s-t)*binomial(ib,t)*np.power(xpa,ia-s+t)*np.power(xpb,ib-t);
    return sum


def overlap_1D(l1, l2, PAx, PBx, gamma):
    sum = 0.
    for i in range( 1 + int(np.floor(0.5*(l1+l2))) ):
        sum += binomial_prefactor(2*i,l1,l2,PAx,PBx)*fact2(2*i-1)/np.power(2*gamma,i)
    return sum

def overlap(alpha1, l1, m1, n1,
            xa, ya, za,
            alpha2, l2, m2, n2,
            xb, yb, zb):

    rab2 = dist2(xa,ya,za,xb,yb,zb);
    gamma = alpha1+alpha2
    xp = product_center_1D(alpha1,xa,alpha2,xb)
    yp = product_center_1D(alpha1,ya,alpha2,yb)
    zp = product_center_1D(alpha1,za,alpha2,zb)

    pre = np.power(np.pi/gamma,1.5)*np.exp(-alpha1*alpha2*rab2/gamma)

    wx = overlap_1D(l1,l2,xp-xa,xp-xb,gamma)
    wy = overlap_1D(m1,m2,yp-ya,yp-yb,gamma)
    wz = overlap_1D(n1,n2,zp-za,zp-zb,gamma)
    return pre*wx*wy*wz

print(overlap(1,0,0,0,0,0,0,1,0,0,0,0,0,0))

# >>> np.random.normal(size=(10,3))
disp_iv = np.array([[-0.7523356 , -1.50438196,  0.68988765],
                    [ 0.52734302, -0.44564955,  0.36458947],
                    [ 0.75467884,  1.08860359,  0.77792354],
                    [-0.02264378, -0.5212034 , -0.37139909],
                    [ 1.16511362, -0.05547106, -1.45820962],
                    [-0.41619208, -0.57016003,  0.44654634],
                    [-0.39531854,  0.96045002,  0.60285589],
                    [-0.87933943, -1.56150387, -1.50859305],
                    [ 0.73025088, -0.22871216,  0.12699363],
                    [-0.32902203,  0.07441659,  0.16721629]])*2
disp_iv = disp_iv[0:1]
def test_displacement(disp_v):
    for downsampling in [8,1]:
        os.environ['GPAW_RECIPROCAL_DOWNSAMPLING'] = str(downsampling)
        error = test_displacement2(disp_v)
        if error>1 and downsampling == 1:
            raise ValueError('Not doing downsampling is supposed to fix the f-f problem, but it did not.')

def test_displacement2(disp_v):
    h = 0.001
    N = 8001
    L = N*h
    rgd = EquidistantRadialGridDescriptor(h, N)
    basis = Basis('C','my', readxml=False, rgd=rgd)
    phit_g = 0.3*np.exp(-1*rgd.r_g**2)
    basis.append(BasisFunction(n=0, l=0, rc=h*L, phit_g=phit_g))
    phit_g = np.exp(-1*rgd.r_g**2)
    basis.append(BasisFunction(n=3, l=3, rc=h*L, phit_g=phit_g))

    atoms = Atoms('H2',
                   positions=[(0, 0, 0),
                              disp_v ])
    atoms.center(vacuum=7)

    calc = GPAW(mode=LCAO(), h=0.15, basis={'C':basis}, setups={'C':'ae'}, txt=None)

    atoms.set_calculator(calc)
    calc.initialize(atoms)

    calc.set_positions(atoms)

    #H_MM = calc.wfs.eigensolver.calculate_hamiltonian_matrix(calc.hamiltonian, calc.wfs, calc.wfs.kpt_u[0])
    S_MM = calc.wfs.kpt_u[0].S_MM

    gd = calc.density.gd

    one_g = gd.zeros()
    one_g[:] = 1.0
    Sref_MM = calc.wfs.basis_functions.calculate_potential_matrices(one_g)[0]
    tri2full(Sref_MM)


    error = np.linalg.norm(S_MM-Sref_MM)
    print('D_v = ',disp_v, '|D| =', np.linalg.norm(disp_v), ' error to grid', np.linalg.norm(S_MM-Sref_MM))
    print(S_MM[0,0],'S')
    print(Sref_MM[0,0],'Sref')
    print(Sref_MM[0,0]-S_MM[0,0],'diff')
    return error

for disp_v in disp_iv:
    test_displacement(disp_v)

