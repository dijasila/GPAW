from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Generator, TypeVar

from typing_extensions import Self

if TYPE_CHECKING:
    from gpaw.new.ibzwfs import IBZWaveFunctions, PWFDWaveFunctions

from gpaw.core.arrays import DistributedArrays
from gpaw.new import zips

_TArray_co = TypeVar("_TArray_co", bound=DistributedArrays, covariant=True)


class ArrayCollection(Generic[_TArray_co]):
    def __init__(self, data_u: list[_TArray_co]):
        self.array_u: list[_TArray_co] = data_u

    def __iter__(self) -> Generator[_TArray_co, None, None]:
        for array in self.array_u:
            yield array

    def __neg__(self) -> "ArrayCollection[_TArray_co]":
        new_array_u: list[_TArray_co] = [
            array.new(-array.data) for array in self
        ]
        return ArrayCollection(new_array_u)

    def __sub__(self, other: Self) -> "ArrayCollection[_TArray_co]":

        new_array_u: list[_TArray_co] = []
        for a, b in zips(self, other):
            new_array_u.append(a.new(a.data - b.data))

        return ArrayCollection(new_array_u)

    def __add__(self, other: Self) -> "ArrayCollection[_TArray_co]":

        new_array_u: list[_TArray_co] = []
        for a, b in zips(self, other):
            new_array_u.append(a.new(a.data + b.data))

        return ArrayCollection(new_array_u)

    def multiply_by_number_in_place(self, number: complex) -> None:
        for array in self:
            array.data *= number

    def multiply_by_number(
        self, number: complex
    ) -> "ArrayCollection[_TArray_co]":

        new_array_u: list[_TArray_co] = [
            array.new(number * array.data) for array in self
        ]

        return ArrayCollection(new_array_u)

    def make_copy(self) -> "ArrayCollection[_TArray_co]":
        new_array_u: list[_TArray_co] = [array.copy() for array in self]

        return ArrayCollection(new_array_u)

    def dot(self, other: Self) -> float:
        """dot product between self and complex conjugate other
        Notes
        _____
        this function can be speed up by a factor 2 by considering only
        unique products between arrays.
        """
        result = 0
        for a, b in zips(self, other):
            tmp_nn = a.integrate(b)
            xp = a.xp
            if not isinstance(tmp_nn, complex):
                xp.fill_diagonal(tmp_nn, tmp_nn.diagonal() * 2)
                result += tmp_nn.sum().real
            else:
                result += 2 * result.real
        return result

    @staticmethod
    def copy_from_ibzwfs(
        ibzwfs: IBZWaveFunctions[PWFDWaveFunctions],
    ) -> "ArrayCollection[_TArray_co]":
        new_array_u: list[_TArray_co] = []
        for wfs in ibzwfs:
            new_array_u.append(wfs.psit_nX.copy())

        return ArrayCollection(new_array_u)

    def empty(self) -> "ArrayCollection[_TArray_co]":
        new_array_u: list[_TArray_co] = [
            array.desc.empty(array.dims, array.comm, array.xp)
            for array in self
        ]
        return ArrayCollection(new_array_u)

    def zeros(self) -> "ArrayCollection[_TArray_co]":
        new_array_u: list[_TArray_co] = [
            array.new(array.xp.zeros_like(array.data)) for array in self
        ]
        return ArrayCollection(new_array_u)
