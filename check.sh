#!/usr/bin/env sh
#
# PostgreSQL vector support for asyncpg.
#
# Copyright 2025-2026, Levente Hunyadi
# https://github.com/hunyadi/asyncpg_vector

set -e
PYTHON_EXECUTABLE=${PYTHON:-python3}

$PYTHON_EXECUTABLE -m ruff check
$PYTHON_EXECUTABLE -m ruff format --check
$PYTHON_EXECUTABLE -m mypy asyncpg_vector
$PYTHON_EXECUTABLE -m mypy tests

$PYTHON_EXECUTABLE -m unittest discover
