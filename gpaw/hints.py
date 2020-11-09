from typing import Tuple, Sequence, Union


class ArrayND(Sequence):
    """Poor mans type-hints for np.ndarray."""

    T: 'ArrayND'
    size: int
    shape: Tuple[int, ...]
    ndim: int

    def sum(self, axis: Union[int, Sequence[int]] = None):
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

    def __truediv__(self, other) -> 'ArrayND':
        ...

    def __rtruediv__(self, other) -> 'ArrayND':
        ...

    def __rmul__(self, other) -> 'ArrayND':
        ...

    def __add__(self, other) -> 'ArrayND':
        ...

    def __sub__(self, other) -> 'ArrayND':
        ...

    def __rsub__(self, other) -> 'ArrayND':
        ...

    def __pow__(self, n: int) -> 'ArrayND':
        ...

    def __neg__(self) -> 'ArrayND':
        ...

    def copy(self) -> 'ArrayND':
        ...

    def conj(self) -> 'ArrayND':
        ...

    def prod(self):
        ...

    def __lt__(self, other) -> 'ArrayND':
        ...


Array1D = ArrayND
Array2D = ArrayND
Array3D = ArrayND
Array4D = ArrayND
