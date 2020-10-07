from typing import Tuple, Sequence


class ArrayND(Sequence):
    """Poor mans type-hints for np.ndarray."""

    T: 'ArrayND'
    size: int
    shape: Tuple[int, ...]
    ndim: int

    def sum(self):
        ...

    def dot(self, other):
        ...

    def __getitem__(self, n):
        ...

    def __setitem__(self, n, v):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...

    def __mul__(self, other) -> 'ArrayND':
        ...

    def __pow__(self, n: int) -> 'ArrayND':
        ...


Array1D = ArrayND
Array2D = ArrayND
Array3D = ArrayND
Array4D = ArrayND
