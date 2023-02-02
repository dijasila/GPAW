# web-page: band.png

import ase.io
from ase.neb import NEBTools

images = ase.io.read('neb.traj', index='-7:')
nebtools = NEBTools(images)
fig = nebtools.plot_band()
fig.savefig('band.png')
