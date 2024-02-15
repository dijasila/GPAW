from ase.geometry.minkowski_reduction import reduction_full
from gpaw.core import UGDesc
from gpaw.fd_operators import Gradient
from ase.geometry import cell_to_cellpar


def test_weird_cell_from_peder():
    ug = UGDesc.from_cell_and_grid_spacing(
        cell=[8.7, 5.6, 7.7, 28, 120, 965], grid_spacing=0.2)
    grad = Gradient(ug._gd, v=0)
    print(grad)
    cell, _ = reduction_full(ug.cell)
    # size_b = U_bc @ ug.size
    # print(size_b)
    print(cell_to_cellpar(ug.cell))
    print(cell_to_cellpar(cell))
    ug2 = UGDesc.from_cell_and_grid_spacing(cell=cell, grid_spacing=0.2)
    grad = Gradient(ug2._gd, v=0)
    print(grad)
