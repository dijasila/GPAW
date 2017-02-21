from gpaw import restart
import matplotlib.pyplot as plt
import numpy as np

mnsi,calc = restart('mnsi.gpw', txt='mnsi.txt')
v = calc.get_electrostatic_potential() 
grid=v.shape
cel=mnsi.cell
x=np.linspace(0,cel[0][0],grid[0],endpoint=False)
y=np.linspace(0,cel[1][1],grid[1],endpoint=False)

f = plt.figure()
ax = f.add_subplot(111)
cax=ax.contour(x,y,v[:,:,int(grid[2]/2.)],100)
cbar = f.colorbar(cax)
ax.set_xlabel('x (Angstrom)'); ax.set_ylabel('y (Angstrom)'); ax.set_title('Pseudo-electrostatic Potential')
f.savefig('pot_contour.png')