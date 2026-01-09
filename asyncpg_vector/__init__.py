"""
PostgreSQL vector support for asyncpg.

Registers data types `vector`, `halfvec` and `sparsevec` from the PostgreSQL extension `vector` to the asynchronous
PostgreSQL client `asyncpg`, and marshals vector data to and from PostgreSQL database tables.

:see: https://github.com/hunyadi/asyncpg_vector
"""

import base64
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from struct import pack, unpack
from typing import ClassVar

import asyncpg

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

__version__ = "0.1.0"
__author__ = "Levente Hunyadi"
__copyright__ = "Copyright 2025, Levente Hunyadi"
__license__ = "MIT"
__maintainer__ = "Levente Hunyadi"
__status__ = "Production"


class BasicVector(ABC):
    "Base class for PostgreSQL vector types."

    @abstractmethod
    def size(self) -> int:
        "Number of vector dimensions."

        ...

    @abstractmethod
    def to_float_list(self) -> list[float]:
        "Converts the vector to a list of double-precision floating-point values."

        ...

    @classmethod
    @abstractmethod
    def from_float_list(cls, vec: Sequence[float]) -> Self:
        "Creates a vector from a list of double-precision floating-point numbers."

        ...

    @classmethod
    def from_float_base64(cls, base64_encoded: str) -> Self:
        """
        Creates a vector from a base64-encoded string storing a list of double-precision floating-point numbers.

        This allows lossless transition of a floating-point value between APIs.
        """

        decoded_bytes = base64.b64decode(base64_encoded)
        float_tuple = unpack(f"{len(decoded_bytes) // 4}f", decoded_bytes)
        return cls.from_float_list(float_tuple)

    @abstractmethod
    def to_database_binary(self) -> bytes:
        "Writes the data into the PostgreSQL data transfer representation."

        ...

    @classmethod
    @abstractmethod
    def from_database_binary(cls, data: bytes) -> Self:
        "Reads the data from the PostgreSQL data transfer representation."

        ...

    @classmethod
    def _to_database_binary(cls, value: Self | list[float] | None) -> bytes | None:
        "Writes a value or instance into the PostgreSQL data transfer representation."

        if value is None:
            return value  # asyncpg uses `None` for representing SQL `NULL`

        match value:
            case BasicVector():
                obj = value
            case list():
                if value:
                    list_item = value[0]
                    if not isinstance(list_item, float):
                        raise ValueError(f"unsupported list item type: {type(list_item).__name__}")

                    obj = cls.from_float_list(value)
                else:
                    obj = cls()
            case _:
                raise ValueError(f"unsupported type: {type(value).__name__}")

        return obj.to_database_binary()

    @classmethod
    def _from_database_binary(cls, value: bytes | None) -> Self | None:
        "Creates an instance from the PostgreSQL data transfer representation."

        if value is None:
            return value

        return cls.from_database_binary(value)


class DenseVector(BasicVector):
    "Base class for PostgreSQL dense vector types."

    __slots__ = ("_data",)

    _data: bytes

    def __init__(self, data: bytes | None = None) -> None:
        if data is not None:
            self._data = data
        else:
            self._data = bytes()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self._data == value._data

    def __str__(self) -> str:
        return f"{type(self).__name__}(size={self.size()})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    @override
    def size(self) -> int:
        return len(self._data) // self.bytes_per_item()

    @classmethod
    @abstractmethod
    def bytes_per_item(cls) -> int:
        "Number of bytes per item in the vector."

        ...

    @override
    def to_database_binary(self) -> bytes:
        return pack(">HH", self.size(), 0) + self._data

    @override
    @classmethod
    def from_database_binary(cls, data: bytes) -> Self:
        size, _unused = unpack(">HH", data[0:4])
        if len(data) != 4 + cls.bytes_per_item() * size:
            raise ValueError(f"expected size: {cls.bytes_per_item()} * {size}; got {len(data) - 4} bytes")
        return cls(data[4:])


class HalfVector(DenseVector):
    "Implements the PostgreSQL extension type `halfvec`."

    type_name: ClassVar[str] = "halfvec"
    cosine_similarity: ClassVar[str] = "halfvec_cosine_ops"

    @override
    @classmethod
    def bytes_per_item(cls) -> int:
        return 2

    @override
    def to_float_list(self) -> list[float]:
        return list(unpack(f">{self.size()}e", self._data))

    @override
    @classmethod
    def from_float_list(cls, vec: Sequence[float]) -> Self:
        return cls(pack(f">{len(vec)}e", *vec))


class Vector(DenseVector):
    "Implements the PostgreSQL extension type `vector`."

    type_name: ClassVar[str] = "vector"
    cosine_similarity: ClassVar[str] = "vector_cosine_ops"

    @override
    @classmethod
    def bytes_per_item(cls) -> int:
        return 4

    @override
    def to_float_list(self) -> list[float]:
        return list(unpack(f">{self.size()}f", self._data))

    @override
    @classmethod
    def from_float_list(cls, vec: Sequence[float]) -> Self:
        return cls(pack(f">{len(vec)}f", *vec))

    @override
    @classmethod
    def from_float_base64(cls, base64_encoded: str) -> Self:
        return cls(base64.b64decode(base64_encoded))


class SparseVector(BasicVector):
    "Implements the PostgreSQL extension type `sparsevec`."

    __slots__ = ("_size", "_indices", "_values")

    _size: int
    _indices: bytes
    _values: bytes

    type_name: ClassVar[str] = "sparsevec"
    cosine_similarity: ClassVar[str] = "sparsevec_cosine_ops"

    def __init__(self, size: int | None = None, indices: bytes | None = None, values: bytes | None = None) -> None:
        if size is not None and indices is not None and values is not None:
            self._size = size
            self._indices = indices
            self._values = values
            if len(self._indices) != len(self._values):
                raise ValueError(f"expected: `indices` (of len {len(self._indices)}) and `values` (of len {len(self._values)}) to match in size")
        elif size is None and indices is None and values is None:
            self._size = 0
            self._indices = bytes()
            self._values = bytes()
        else:
            raise ValueError("expected: either all of `size`, `indices` and `values`, or neither")

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self._size == value._size and self._indices == value._indices and self._values == value._values

    def __str__(self) -> str:
        return f"{type(self).__name__}(size={self.size()}, nnz={self.nnz()})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size()}, indices={self._indices!r}, values={self._values!r})"

    @override
    def size(self) -> int:
        return self._size

    def nnz(self) -> int:
        return len(self._indices) // 4

    @override
    def to_float_list(self) -> list[float]:
        count = self.nnz()
        indices = unpack(f">{count}i", self._indices)
        values = unpack(f">{count}f", self._values)
        items = [0.0 for _ in range(self._size)]
        for index, value in zip(indices, values, strict=True):
            items[index] = value
        return items

    @override
    @classmethod
    def from_float_list(cls, vec: Sequence[float]) -> Self:
        indices = [index for index, value in enumerate(vec) if value != 0]
        values = [vec[index] for index in indices]
        count = len(indices)
        return cls(len(vec), pack(f">{count}i", *indices), pack(f">{count}f", *values))

    @override
    def to_database_binary(self) -> bytes:
        return pack(">iii", self._size, self.nnz(), 0) + self._indices + self._values

    @override
    @classmethod
    def from_database_binary(cls, data: bytes) -> Self:
        size, count, _ = unpack(">iii", data[0:12])
        return cls(size, data[12 : 12 + 4 * count], data[12 + 4 * count :])


async def register_vector(conn: asyncpg.Connection, schema: str = "public") -> None:
    "Registers `vector` extension types with Python module `asyncpg`."

    await conn.set_type_codec("vector", schema=schema, encoder=Vector._to_database_binary, decoder=Vector._from_database_binary, format="binary")  # pyright: ignore[reportPrivateUsage]
    await conn.set_type_codec("halfvec", schema=schema, encoder=HalfVector._to_database_binary, decoder=HalfVector._from_database_binary, format="binary")  # pyright: ignore[reportPrivateUsage]
    await conn.set_type_codec("sparsevec", schema=schema, encoder=SparseVector._to_database_binary, decoder=SparseVector._from_database_binary, format="binary")  # pyright: ignore[reportPrivateUsage]
