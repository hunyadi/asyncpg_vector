import unittest
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from random import random
from struct import pack, unpack

import asyncpg

from asyncpg_vector import HalfVector, SparseVector, Vector, register_vector


@asynccontextmanager
async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    conn = await asyncpg.connect(host="localhost", port=5432, user="postgres", password="postgres")
    try:
        yield conn
    finally:
        await conn.close()


def to_float16(values: Sequence[float]) -> list[float]:
    "Truncates a list of floating-point values to 16-bit width."

    return list(unpack(f"{len(values)}e", pack(f"{len(values)}e", *values)))


def to_float32(values: Sequence[float]) -> list[float]:
    "Truncates a list of floating-point values to 32-bit width."

    return list(unpack(f"{len(values)}f", pack(f"{len(values)}f", *values)))


def random_dense() -> list[float]:
    "Generates a dense random vector."

    return [random() for _ in range(1536)]


def random_sparse() -> list[float]:
    "Generates a sparse random vector with many zeros."

    return [(random() if random() > 0.9 else 0.0) for _ in range(1536)]


class TestVector(unittest.IsolatedAsyncioTestCase):
    async def test_types(self) -> None:
        f32d_vector = to_float32(random_dense())
        f16d_vector = to_float16(random_dense())
        f32s_vector = to_float32(random_sparse())

        # empty vectors
        self.assertEqual(Vector().size(), 0)
        self.assertEqual(HalfVector().size(), 0)
        self.assertEqual(SparseVector().size(), 0)

        # str() yields compact representation
        self.assertLess(len(str(Vector.from_float_list(f32d_vector))), 64)
        self.assertLess(len(str(HalfVector.from_float_list(f16d_vector))), 64)
        self.assertLess(len(str(SparseVector.from_float_list(f32s_vector))), 64)

        # repr() yields full representation
        self.assertGreater(len(repr(Vector.from_float_list(f32d_vector))), 1000)
        self.assertGreater(len(repr(HalfVector.from_float_list(f16d_vector))), 1000)
        self.assertGreater(len(repr(SparseVector.from_float_list(f32s_vector))), 1000)

        # round trip for list[float]
        self.assertEqual(Vector.from_float_list(f32d_vector).to_float_list(), f32d_vector)
        self.assertEqual(HalfVector.from_float_list(f16d_vector).to_float_list(), f16d_vector)
        self.assertEqual(SparseVector.from_float_list(f32s_vector).to_float_list(), f32s_vector)

    async def test_connection(self) -> None:
        create_sql = """
        --sql
        CREATE EXTENSION IF NOT EXISTS vector;

        --sql
        CREATE TEMPORARY TABLE vector_types(
            id bigint GENERATED ALWAYS AS IDENTITY,
            embedding vector(1536) NOT NULL,
            half_embedding halfvec(1536) NOT NULL,
            sparse_embedding sparsevec(1536) NOT NULL,
            CONSTRAINT pk_vector_types PRIMARY KEY (id)
        );
        """

        insert_sql = """
        --sql
        INSERT INTO vector_types (embedding, half_embedding, sparse_embedding)
        VALUES ($1, $2, $3);
        """

        select_sql = """
        --sql
        SELECT embedding, half_embedding, sparse_embedding
        FROM vector_types
        ORDER BY id;
        """

        async with get_connection() as conn:
            await conn.execute(create_sql)
            await register_vector(conn)

            records = [
                (
                    Vector.from_float_list(to_float32(random_dense())),
                    HalfVector.from_float_list(to_float16(random_dense())),
                    SparseVector.from_float_list(to_float32(random_sparse())),
                )
                for _ in range(1)
            ]
            await conn.executemany(insert_sql, records)
            rows = await conn.fetch(select_sql)
            for row, record in zip(rows, records, strict=True):
                self.assertEqual(tuple(row), record)


if __name__ == "__main__":
    unittest.main()
