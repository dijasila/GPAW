import pickle
import matplotlib.pyplot as plt
from ase.units import Hartree

plt.figure(1)
plt.figure(figsize=(6.5, 4.5))
plt.figure(2)
plt.figure(figsize=(6.5, 4.5))

for k in [3,5,7,9]:

    e=[]
    HF_gap=[]
    GW_gap=[]

    for ecut in [50,100,150,200]:

        data=pickle.load(open('Si_GW_k%s_ecut%s.pckl' % (k, ecut)))
        e_skn=data['e_skn']
        vxc_skn=data['vxc_skn']
        exx_skn=data['exx_skn']
        QP_skn=data['QP_skn']

        HF_skn = e_skn - vxc_skn + exx_skn
        HFgap = (HF_skn[0,0,2] - HF_skn[0,0,1])*Hartree
        QPgap = (QP_skn[0,0,2] - QP_skn[0,0,1])*Hartree

        HF_gap.append(HFgap)
        GW_gap.append(QPgap)
        e.append(ecut)

    plt.figure(1)
    plt.plot(e, HF_gap, 'o-', label='(%sx%sx%s) k-points' % (k, k, k))
    plt.figure(2)
    plt.plot(e, GW_gap, 'o-', label='(%sx%sx%s) k-points' % (k, k, k))

plt.figure(1)
plt.xlabel('$E_{\text{cut}}$ (eV)')
plt.ylabel('Direct band gap (eV)')
plt.xlim([0., 250.])
plt.ylim([7.5, 10.])
plt.title('non-selfconsistent Hartree-Fock')
plt.legend(loc='upper right')
plt.savefig('Si_EXX.png')

plt.figure(2)
plt.xlabel('$E_{\text{cut}}$ (eV)')
plt.ylabel('Direct band gap (eV)')
plt.xlim([0., 250.])
plt.ylim([1.5, 4.])
plt.title('G$_0$W$_0$@LDA')
plt.legend(loc='upper right')
plt.savefig('Si_GW.png')
