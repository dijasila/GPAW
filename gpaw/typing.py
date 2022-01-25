from typing import Any, TYPE_CHECKING, Union, Tuple
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any  # type: ignore

ArrayLike1D = ArrayLike
ArrayLike2D = ArrayLike

ArrayND = np.ndarray
Array1D = ArrayND
Array2D = ArrayND
Array3D = ArrayND
Array4D = ArrayND

Vector = Union[Tuple[float, float, float], Array1D]
