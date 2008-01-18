from gpaw import Calculator

# Calculate band structure along Gamma-X i.e. from 0 to 0.5
nkpt = 50
kpts = [(k / float(2 * nkpt), 0, 0) for k in range(nkpt)]
calc = Calculator('8Si.gpw',
                   kpts=kpts) #, fixdensity=True, nbands=8*3,
#                   eigensolver='cg')
calc.get_potential_energy()
 
# Write the results to a file e.g. for plotting with gnuplot
f = open('Si8_bands.txt', 'w')
for k, kpt_c in enumerate(calc.GetIBZKPoints()):
    for eig in calc.get_eigenvalues(kpt=k):
        print >> f, kpt_c[0], eig - calc.get_fermi_level()
