from typing import Any, Union, Sequence
import numpy as np

if [int(x) for x in np.__version__.split('.')] >= [1, 20]:
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
