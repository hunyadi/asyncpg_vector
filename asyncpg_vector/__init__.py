"""
PostgreSQL vector support for asyncpg.

Registers data types `vector` and `halfvec` from the PostgreSQL extension `vector` to the asynchronous PostgreSQL
client `asyncpg`, and marshals vector data to and from PostgreSQL database tables.

:see: https://github.com/hunyadi/asyncpg_vector
"""

import base64
import struct
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from struct import pack
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

    __slots__ = ("_data",)

    _data: bytes

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self._data == value._data

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def size(self) -> int:
        "Number of vector dimensions."

        return len(self._data) // self.bytes_per_item()

    @classmethod
    @abstractmethod
    def bytes_per_item(cls) -> int:
        "Number of bytes per item in the vector."

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
        float_tuple = struct.unpack(f"{len(decoded_bytes) // 4}f", decoded_bytes)
        return cls.from_float_list(float_tuple)

    def to_database_binary(self) -> bytes:
        "Writes the data into the PostgreSQL data transfer representation."

        return pack(">HH", self.size(), 0) + self._data

    @classmethod
    def from_database_binary(cls, data: bytes) -> Self:
        "Reads the data from the PostgreSQL data transfer representation."

        size, _unused = struct.unpack(">HH", data[0:4])
        if len(data) != 4 + cls.bytes_per_item() * size:
            raise ValueError(f"expected size: {cls.bytes_per_item()} * {size}; got {len(data) - 4} bytes")
        return cls(data[4:])

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
                    obj = cls(bytes())
            case _:
                raise ValueError(f"unsupported type: {type(value).__name__}")

        return obj.to_database_binary()

    @classmethod
    def _from_database_binary(cls, value: bytes | None) -> Self | None:
        "Creates an instance from the PostgreSQL data transfer representation."

        if value is None:
            return value

        return cls.from_database_binary(value)


class HalfVector(BasicVector):
    "Implements the PostgreSQL extension type `halfvec`."

    type_name: ClassVar[str] = "halfvec"
    cosine_similarity: ClassVar[str] = "halfvec_cosine_ops"

    @override
    @classmethod
    def bytes_per_item(cls) -> int:
        return 2

    @override
    def to_float_list(self) -> list[float]:
        return list(struct.unpack(f">{self.size()}e", self._data))

    @override
    @classmethod
    def from_float_list(cls, vec: Sequence[float]) -> Self:
        return cls(struct.pack(f">{len(vec)}e", *vec))


class Vector(BasicVector):
    "Implements the PostgreSQL extension type `vector`."

    type_name: ClassVar[str] = "vector"
    cosine_similarity: ClassVar[str] = "vector_cosine_ops"

    @override
    @classmethod
    def bytes_per_item(cls) -> int:
        return 4

    @override
    def to_float_list(self) -> list[float]:
        return list(struct.unpack(f">{self.size()}f", self._data))

    @override
    @classmethod
    def from_float_list(cls, vec: Sequence[float]) -> Self:
        return cls(struct.pack(f">{len(vec)}f", *vec))

    @override
    @classmethod
    def from_float_base64(cls, base64_encoded: str) -> Self:
        return cls(base64.b64decode(base64_encoded))


async def register_vector(conn: asyncpg.Connection, schema: str = "public") -> None:
    "Registers `vector` extension types with Python module `asyncpg`."

    await conn.set_type_codec("vector", schema=schema, encoder=Vector._to_database_binary, decoder=Vector._from_database_binary, format="binary")  # pyright: ignore[reportPrivateUsage]
    await conn.set_type_codec("halfvec", schema=schema, encoder=HalfVector._to_database_binary, decoder=HalfVector._from_database_binary, format="binary")  # pyright: ignore[reportPrivateUsage]
