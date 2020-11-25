# creates: g2.png

import matplotlib.pyplot as plt


def find_didnotconverge(path):
    # print('\nAnalizing {}'.format(path))
    file = open(path, 'r')
    file = file.read().split('\n')
    x = []
    y = []
    didnotconverge = []
    for s in file:
        if s == '':
            continue
        x.append(s.split('\t')[0])
        try:
            y.append(float(s.split('\t')[-2]))
        except ValueError:
            # print(x[-1] + ' did not convegre in {}'.format(path))
            didnotconverge.append(x[-1])
            continue
    # if len(didnotconverge) == 0:
    #     print('All convegred')
    return didnotconverge


f = plt.figure(figsize=(12, 4), dpi=240)
plt.subplot(121)
plt.title('Convegrence: ' + r'res$^2$ $< 1\cdot10^{-10} $eV$^2$')
# scf
scf_r = open('g2_scfdata', 'r')
scf_r = scf_r.read().split('\n')
x = []
y = []
i = 0
i_max = 10
for s in scf_r:
    if s == '':
        continue
    x.append(s.split('\t')[0])
    y.append(int(s.split('\t')[-4]))
    i += 1
    if i == i_max:
        break

plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'b^-', label='SCF', fillstyle='none')

# direct_min
prec = open('g2_dmdata', 'r')
prec = prec.read().split('\n')
x = []
y = []
i = 0
i_max = 10
for s in prec:
    if s == '':
        continue
    x.append(s.split('\t')[0])
    y.append(int(s.split('\t')[-4]))
    i += 1
    if i == i_max:
        break

plt.plot(range(len(x)), y, 'ro-', label='Direct Min',
         fillstyle='none')
plt.legend()
plt.ylabel('Number of iterations (energy and gradients calls)')
# direct_min
didnotconverge = find_didnotconverge('g2_scfdata')
plt.subplot(122)
plt.title('Convegrence: ' + r'res$^2$ $< 1\cdot 10^{-10} $eV$^2$')
prec = open('g2_dmdata', 'r')
prec = prec.read().split('\n')
x = []
y = []
i = 0
i_max = 10
for s in prec:
    if s == '':
        continue
    a = s.split('\t')[0]
    if a in didnotconverge:
        x.append(s.split('\t')[0])
        y.append(int(s.split('\t')[-4]))
        i += 1
        if i == i_max:
            break

plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'ro-', label='Direct Min',
         fillstyle='none')
plt.legend()
# plt.ylabel('Number of iterations (energy and gradients calls)')
plt.text(-6.3, 75.5, '(a)')
plt.text(-0.7, 75.5, '(b)')
# f.savefig("conv.eps", bbox_inches='tight')
f.savefig("g2.png", bbox_inches='tight')
