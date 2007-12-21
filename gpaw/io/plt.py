"""IO routines for gOpenMol binary plt format"""

from struct import calcsize,pack,unpack
import numpy as npy
from ASE.Units import Convert
from gpaw.utilities import check_unit_cell

import gpaw.mpi as mpi
from gpaw.mpi import MASTER

def read_plt(filename):
    """Read plt files
    returns the cell(3x3 matrix), the grid and the origin,
    use it like:

    cell, grid, origin = read_plt('file.plt')
    """
    f = open(filename)
    fmt='ii'
    # check endian encoding
    byteswap = False
    three, tos = unpack(fmt,f.read(calcsize(fmt)))
    if three != 3: byteswap = True

    fmt='iii'
    if byteswap: fmt='>iii'
    nz, ny, nx = unpack(fmt,f.read(calcsize(fmt)))

    fmt='ff'
    if byteswap: fmt='>ff'
    z0, ze = unpack(fmt,f.read(calcsize(fmt)))
    y0, ye = unpack(fmt,f.read(calcsize(fmt)))
    x0, xe = unpack(fmt,f.read(calcsize(fmt)))
    dz = (ze-z0)/(nz-1)
    dy = (ye-y0)/(ny-1)
    dx = (xe-x0)/(nx-1)
    cell = npy.zeros((3,3))
    cell[0,0] = (nx+1)*dx 
    cell[1,1] = (ny+1)*dy 
    cell[2,2] = (nz+1)*dz 
    
    fmt='f'
    if byteswap: fmt='>f'
    size = nx*ny*nz * calcsize(fmt)
    arr = npy.fromstring(f.read(size),float32)
    if byteswap: arr = arr.byteswap()
    f.close()

    return cell, npy.transpose(npy.resize(arr,(nz,ny,nx))), (x0-dx,y0-dy,z0-dz)

def write_collected_plt(gd,
                        grid,
                        filename,
                        origin=(0.0,0.0,0.0), # ASE uses (0,0,0) as origin
                        typ=4):
    collected_grid = gd.collect(grid)
    if mpi.rank == MASTER:
        write_plt(gd, collected_grid, filename, origin, typ)

def write_plt(cell,
              grid,
              filename,
              origin=(0.0,0.0,0.0), # ASE uses (0,0,0) as origin
              typ=4):
    """All parameters are Input.

    cell unit cell object as given from ListOfAtoms.GetUnitCell() or grid decriptor

    grid the grid to write

    typ  type of surface (integer)

    The cell is assumed to be in Angstroms and the grid in atomc units (Bohr)
    """
    a0_A = Convert(1, 'Bohr', 'Ang') 
    if hasattr(cell, '_new_array'): # this is a GridDescriptor
        xe, ye, ze = cell.h_c * cell.N_c * a0_A # get Angstroms
    elif len(cell.shape) == 2:
        # Check that the cell is orthorhombic
        check_unit_cell(cell)
        xe, ye, ze = npy.diagonal(cell)
    else:
        xe, ye, ze = cell * a0_A # get Angstroms

    # Check, that the grid is 3D
    if len(grid.shape) != 3:
        raise RuntimeError("grid must be 3D")
    nx, ny, nz = grid.shape
    dx, dy, dz = [ xe/(nx+1), ye/(ny+1), ze/(nz+1) ]
    
    f = open(filename, 'w')
    f.write(pack('ii', 3, typ))
    f.write(pack('iii', nz, ny, nx))

    x0, y0, z0 = npy.array(origin) + npy.array([dx,dy,dz])
    xe = x0 +(nx-1)*dx
    ye = y0 +(ny-1)*dy
    ze = z0 +(nz-1)*dz
    f.write(pack('ff', z0, ze ))
    f.write(pack('ff', y0, ye ))
    f.write(pack('ff', x0, xe ))

    # we need a float array
    # Future: numpy has no 'typecode'
    if hasattr(grid,'typecode') and grid.dtype.char == 'f':
        fgrid = npy.transpose(grid)
    else:
        fgrid = npy.array(npy.transpose(grid).tolist(),'f')
        #    npy.asarray does not work here !
        #    fgrid = npy.asarray(npy.transpose(grid), float32)
    f.write(fgrid.tostring())

    f.close()

def wf2plt(paw,i,spin=0,fname=None):
    """Write a specific wavefunction as plt file"""
    kpt = paw.kpt_u[spin]
    wf = kpt.psit_nG[i]
    gd = kpt.gd

    if fname is None:
        fname = 'wf'+str(i)+'_'+str(spin)+'.plt'
    write_plt(gd, wf, fname)
    
def pot2plt(paw, spin=0, fname=None):
    """Write the potential as plt file"""
    kpt = paw.kpt_u[spin]
    gd = kpt.gd
    vt = paw.hamiltonian.vt_sG[spin]

    if fname is None:
        fname = 'vt_Gs'+str(spin)+'.plt'
    write_collected_plt(gd, vt, fname)

  
