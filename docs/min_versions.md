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
dependency markers, but the CI matrix covers the endpoints: **3.8 (pinned legacy
job), 3.11, 3.13 and 3.14**.

Astropy is not part of the SPEC 0 core-package list; its floors are chosen to
match the NumPy and Python versions above.

## Legacy stack removal target

The Python 3.8 exception — and therefore the NumPy 1.24, SciPy 1.10 and Astropy
5.2 floors — is targeted for removal in **`ftsdata` 0.2.0**, or as soon as the
CONCERTO environment used by `kidsdata` moves to Python 3.11+ and Astropy 7+,
whichever comes first.

Until then:

- `ci/constraints-py38.txt` pins the oldest supported stack;
- the `test-python38-legacy` CI job runs the test-suite against those pins;
- compatibility workarounds live in `ftsdata/_compat.py` and are marked for
  removal with the legacy stack.

Removing the exception means updating the Python/NumPy/SciPy/Astropy floors in
`pyproject.toml`, dropping the legacy CI job and constraints file, deleting
`ftsdata/_compat.py`, and updating this page and the support policy in
`README.rst`.

## Updating the floors

When a Python version or core dependency leaves its SPEC 0 support window,
update the following in the same change:

- `requires-python` and the dependency markers in `pyproject.toml`;
- the table above;
- the support-policy section in `README.rst`;
- the Python matrix and legacy constraints in `.gitlab-ci.yml` /
  `ci/constraints-py38.txt`.
