import pickle
import matplotlib.pyplot as plt
from ase.units import Hartree

plt.figure(1)
plt.figure(figsize=(6.5, 4.5))
plt.figure(2)
plt.figure(figsize=(6.5, 4.5))

for dw in [0.01,0.02,0.05,0.1,0.2,0.5]:

    e=[]
    GW_gap=[]

    for wlin in [25.,50.,75.,100.]:

        data=pickle.load(open('Si_GW_wlin%s_dw%s.pckl' % (wlin, dw)))
        QP_skn=data['QP_skn']
        QPgap = (QP_skn[0,0,2] - QP_skn[0,0,1])*Hartree

        GW_gap.append(QPgap)
        e.append(dw)

    plt.plot(e, GW_gap, 'o-', label='$\omega_{\text{lin}} = $ %s' %wlin

plt.xlabel('$\Delta \omega$ (eV)')
plt.ylabel('Direct band gap (eV)')
plt.xlim([0., 250.])
plt.ylim([1.5, 4.])
plt.title('G$_0$W$_0$@LDA')
plt.legend(loc='upper right')
plt.savefig('Si_w.png')
