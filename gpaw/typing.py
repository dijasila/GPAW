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
    from numpy.typing import ArrayLike, DTypeLike, NDArray
    RealNDArray = NDArray[np.float64]
    ComplexNDArray = NDArray[np.complex128]
except ImportError:
    ArrayLike = Any  # type: ignore
    DTypeLike = Any  # type: ignore
    RealNDArray = np.ndarray[Any, np.dtype[np.float64]]  # type: ignore
    ComplexNDArray = np.ndarray[Any, np.dtype[np.complex128]]  # type: ignore

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
