# PostgreSQL vector support for asyncpg

Adds PostgreSQL [vector](https://github.com/pgvector/pgvector) support for Python.

Registers data types `vector` and `halfvec` from the PostgreSQL extension `vector` to the asynchronous PostgreSQL client `asyncpg`, and marshals vector data to and from PostgreSQL database tables.

Internally, the data is packed into a Python `bytes` object, with single-precision float vectors stored on 4 bytes per item (for class `Vector`) and half-precision float vectors stored on 2 bytes per item (for class `HalfVector`). Data is (un)packed using `struct` from the standard library as necessary.

This module provides functionality similar to [pgvector-python](https://github.com/pgvector/pgvector-python) but imports minimum dependencies (e.g. no dependency on `numpy`).

## Setup

#### Install the package

```sh
pip install asyncpg_vector
```

#### Initialize

Register vector types with your database connection or connection pool:

**Connection**:

```python
from asyncpg_vector import register_vector

async def main() -> None:
    ...

    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)
```

**Pool**:

```python
from asyncpg_vector import register_vector

async def init_connection(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)

async def main() -> None:
    ...

    pool = await asyncpg.create_pool(..., init=init_connection)
```

#### Perform similarity search

First, create a table and an index:

```python
async def create(conn: asyncpg.Connection) -> None:
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS items
        (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY,
            content text NOT NULL,
            embedding halfvec(1536) NOT NULL,
            CONSTRAINT pk_items PRIMARY KEY (id)
        );

        CREATE INDEX IF NOT EXISTS embedding_index ON items
        USING hnsw (embedding halfvec_cosine_ops);
    """)
```

Next, find documents in a knowledge base that match a search phrase using vector similarity with approximate nearest neighbor semantics:

```python
from asyncpg_vector import HalfVector

async def search(conn: asyncpg.Connection, phrase: str) -> list[str]:
    ...

    embedding_response = await ai_client.embeddings.create(
        input=phrase,
        model="text-embedding-3-small",
        encoding_format="base64"
    )
    embedding = HalfVector.from_float_base64(embedding_response.data[0].embedding)
    query = """
        SELECT
            id,
            content,
            embedding <=> $1 AS distance
        FROM items
        ORDER BY distance
        LIMIT 5
    """
    rows = await conn.fetch(query, embedding)

    ...
```
