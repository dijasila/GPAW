from gpaw import GPAW,PW, FermiDirac
from ase.io import read
from math import sqrt, pi
import numpy as np
from os import getcwd
from os import listdir
from os.path import isfile, join
from gpaw.mpi import world


cwd = getcwd()

atoms_name = [f for f in listdir(cwd) if isfile(join(cwd, f)) and f.split('.')[-1] == 'xyz'][0]
atoms = read(atoms_name)

atoms.center(vacuum = 15, axis = 2)

dens_gs = 25
dens_fd = 12

sc = (1,1,1)

vx, vy, vz = atoms.get_cell()
lx, ly, lz = sqrt(np.sum(vx**2)), sqrt(np.sum(vy**2)), sqrt(np.sum(vz**2))

kx_gs = dens_gs*2.0*pi/lx
ky_gs = dens_gs*2.0*pi/ly

kx_gs = int(kx_gs) + (1-int(kx_gs)%2)
ky_gs = int(ky_gs) + (1-int(ky_gs)%2)

kx_fd = dens_fd*2.0*pi/lx
ky_fd = dens_fd*2.0*pi/ly

kx_fd = int(kx_fd) + (1-int(kx_fd)%2)
ky_fd = int(ky_fd) + (1-int(ky_fd)%2)

params_gs = dict(
        mode='lcao',
        symmetry={"point_group":False},
        nbands = "nao",
        convergence={"forces":1.e-5, "bands":"all"},
        basis='dzp',
        h = 0.1,
        parallel = {'domain': 1, "band":1},
        occupations=FermiDirac(width=0.05),
        kpts={'size': (kx_gs,ky_gs,1), 'gamma': True},
        xc='PBE')

params_fd = dict(
        mode='lcao',
        symmetry={"point_group":False},
        nbands = "nao",
        convergence={"forces":1.e-5, "bands":"all"},
        basis='dzp',
        parallel = {'domain': 1},
        occupations=FermiDirac(width=0.05),
        kpts={'size': (kx_fd,ky_fd,1), 'gamma': True},
        xc='PBE')
