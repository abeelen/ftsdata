# Minimum supported versions

`ftsdata` follows [SPEC 0 — Minimum Supported Dependencies](https://scientific-python.org/specs/spec-0000/)
for Python and core dependency support, with one explicit exception: **Python
3.8 remains supported** for the CONCERTO environment used by `kidsdata`.

## Policy

- Python versions are supported for three years after their initial release.
- Core dependencies are supported for two years after their initial release.
- Python 3.8 is an explicit exception to the Python support window.
- Minimum versions are raised on minor or major releases, never on patch
  releases.

## Current floors

As of October 2026:

| Python | NumPy | SciPy | Astropy |
|--------|-------|-------|---------|
| 3.8 (exception) | 1.24.4 | 1.10.1 | 5.2.2 |
| 3.9–3.10 | 2.0.2 | 1.13.1 | 6.0.1 |
| 3.11+ (SPEC 0) | 2.3.0 | 1.15.0 | 7.0.0 |

The 3.9–3.10 row keeps the package installable between the Python 3.8 exception
and the current SPEC 0 versions.  Python 3.9–3.12 remain installable through the
dependency markers, but the CI matrix covers the endpoints: **3.8, 3.11, 3.13
and 3.14**.

Astropy is not part of the SPEC 0 core-package list; its floors are chosen to
match the NumPy and Python versions above.

## Updating the floors

When a Python version or core dependency leaves its SPEC 0 support window,
update the following in the same change:

- `requires-python` and the dependency markers in `pyproject.toml`;
- the table above;
- the support-policy section in `README.md`;
- the Python matrix in `.gitlab-ci.yml`.
