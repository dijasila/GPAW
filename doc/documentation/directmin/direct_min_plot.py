import matplotlib.pyplot as plt
import glob
f = plt.figure(figsize=(12, 4), dpi=240)
plt.subplot(121)
plt.title('Convegrence: ' +  r'res$^2$ $< 10^{-8} $eV$^2$')
# scf
scf_r = open('scf_ex.txt','r')
scf_r = scf_r.read().split('\n')
x = []
y = []
for s in scf_r:
    if s == '':
        continue
    x.append(s.split('\t')[0])
    y.append(int(s.split('\t')[1]))

plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'b^-', label='SCF', fillstyle='none')

# direct_min
prec = open('prec_3_ex.txt', 'r')
prec = prec.read().split('\n')
x = []
y = []
for s in prec:
    if s == '':
        continue
    x.append(s.split('\t')[0])
    y.append(int(s.split('\t')[1]))

plt.plot(range(len(x)), y, 'ro-', label='Direct Min', fillstyle='none')

plt.legend()
plt.ylabel('Number of iterations (energy and gradients calls)')
# plt.show()
# f.savefig("conv.png", bbox_inches='tight')


# direct_min
plt.subplot(122)
plt.title('Convegrence: ' +  r'res$^2$ $< 10^{-8} $eV$^2$')
prec = open('prec_3_dc.txt', 'r')
prec = prec.read().split('\n')
x = []
y = []
for s in prec:
    if s == '':
        continue
    x.append(s.split('\t')[0])
    y.append(int(s.split('\t')[1]))

plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'ro-', label='Direct Min', fillstyle='none')
plt.legend()
# plt.ylabel('Number of iterations (energy and gradients calls)')
plt.text(-6.0, 46.5, '(a)')
plt.text(-0.7, 46.5, '(b)')
f.savefig("scf_vs_dm.png", bbox_inches='tight')

