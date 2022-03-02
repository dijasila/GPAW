from typing import Any, TYPE_CHECKING, Union, Sequence
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike
else:
    ArrayLike = Any  # type: ignore
    DTypeLike = Any  # type: ignore

ArrayLike1D = ArrayLike
ArrayLike2D = ArrayLike

ArrayND = np.ndarray
Array1D = ArrayND
Array2D = ArrayND
Array3D = ArrayND
Array4D = ArrayND

Vector = Union[Sequence[float], Array1D]
