from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Generic, TypeVar
from typing_extensions import Self

if TYPE_CHECKING:
    from gpaw.new.ibzwfs import IBZWaveFunctions, PWFDWaveFunctions

from gpaw.core.arrays import DistributedArrays
from gpaw.new import zips

_TArray_co = TypeVar("_TArray_co", bound=DistributedArrays, covariant=True)


class ArrayCollection(Generic[_TArray_co]):
    def __init__(self, data_qs: list[list[_TArray_co]]):
        self.array_qs: list[list[_TArray_co]] = data_qs

    def __iter__(self) -> Generator[_TArray_co, None, None]:
        for array_s in self.array_qs:
            yield from array_s

    def __neg__(self) -> "ArrayCollection[_TArray_co]":
        new_array_qs: list[list[_TArray_co]] = []
        for array_s in self.array_qs:
            new_array_qs.append([])
            for array in array_s:
                new_array_qs[-1].append(array.new(-array.data))

        return ArrayCollection(new_array_qs)

    def __sub__(self, other: Self) -> "ArrayCollection[_TArray_co]":

        new_array_qs: list[list[_TArray_co]] = []
        for a_s, b_s in zips(self.array_qs, other.array_qs):
            new_array_qs.append([])
            for a, b in zips(a_s, b_s):
                new_array_qs[-1].append(a.new(a.data - b.data))

        return ArrayCollection(new_array_qs)

    def __add__(self, other: Self) -> "ArrayCollection[_TArray_co]":

        new_array_qs: list[list[_TArray_co]] = []
        for a_s, b_s in zips(self.array_qs, other.array_qs):
            new_array_qs.append([])
            for a, b in zips(a_s, b_s):
                new_array_qs[-1].append(a.new(a.data + b.data))

        return ArrayCollection(new_array_qs)

    def multiply_by_number_in_place(self, number: complex) -> None:
        for array in self:
            array.data *= number

    def multiply_by_number(
        self, number: complex
    ) -> "ArrayCollection[_TArray_co]":
        new_array_qs: list[list[_TArray_co]] = []
        for array_s in self.array_qs:
            new_array_qs.append([])
            for array in array_s:
                new_array_qs[-1].append(array.new(number * array.data))

        return ArrayCollection(new_array_qs)

    def make_copy(self) -> "ArrayCollection[_TArray_co]":
        new_array_qs: list[list[_TArray_co]] = []
        for array_s in self.array_qs:
            new_array_qs.append([])
            for array in array_s:
                new_array_qs[-1].append(array.copy())

        return ArrayCollection(new_array_qs)

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
                result += sum(tmp_nn).real
            else:
                result += 2 * result.real
        return result

    @staticmethod
    def copy_from_ibzwfs(
        ibzwfs: IBZWaveFunctions[PWFDWaveFunctions],
    ) -> "ArrayCollection[_TArray_co]":
        new_array_qs: list[list[_TArray_co]] = []
        for wfs_s in ibzwfs.wfs_qs:
            new_array_qs.append([])
            for wfs in wfs_s:
                new_array_qs[-1].append(wfs.psit_nX.copy())

        return ArrayCollection(new_array_qs)

    def empty(self) -> "ArrayCollection[_TArray_co]":
        new_array_qs: list[list[_TArray_co]] = []
        for array_s in self.array_qs:
            new_array_qs.append([])
            for array in array_s:
                new_array_qs[-1].append(
                    array.desc.empty(array.dims, array.comm, array.xp)
                )

        return ArrayCollection(new_array_qs)
