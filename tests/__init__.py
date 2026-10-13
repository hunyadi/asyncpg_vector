"""
PostgreSQL vector support for asyncpg.

Registers data types `vector`, `halfvec` and `sparsevec` from the PostgreSQL extension `vector` to the asynchronous
PostgreSQL client `asyncpg`, and marshals vector data to and from PostgreSQL database tables.

Copyright 2025-2026, Levente Hunyadi

:see: https://github.com/hunyadi/asyncpg_vector
"""
