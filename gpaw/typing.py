from typing import Any, Union, Sequence
import numpy as np

try:
    # New in Python-3.11
    from typing_extension import Self
except ImportError:
    Self = Any  # type: ignore

try:
    # New in Python-3.8
    from typing import Literal
except ImportError:
    class _Literal:
        def __getitem__(self, index):
            return Any
    Literal = _Literal()  # type: ignore

try:
    # Needs numpy-1.20:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = Any  # type: ignore
    DTypeLike = Any  # type: ignore

ArrayLike1D = ArrayLike
ArrayLike2D = ArrayLike

ArrayND = np.ndarray
Array1D = ArrayND
Array2D = ArrayND
Array3D = ArrayND
Array4D = ArrayND

# Used for sequences of three numbers:
Vector = Union[Sequence[float], Array1D]
IntVector = Union[Sequence[int], Array1D]
