# Packages

This directory contains shared packages used across the monorepo.

## Purpose

- Reusable code that multiple apps depend on
- Shared types and interfaces
- Common utilities and helpers
- Core abstractions

## Creating a New Package

1. Create a new directory in `packages/`
2. Add a `pyproject.toml` with:
   - `[project]` metadata
   - `[tool.uv] package = true` (for installable packages)
   - `dependencies` as needed
3. Add a `src/<package_name>/` directory with your code
4. Add an `__init__.py` to make it a Python package

## Example Package Structure

```
packages/
└── my-package/
    ├── pyproject.toml
    ├── README.md
    └── src/
        └── my_package/
            ├── __init__.py
            └── module.py
```

## Available Packages

(TODO: Add packages as they are created)
