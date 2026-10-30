# Releasing

`ftsdata` follows the [OpenAstronomy packaging
guide](https://packaging-guide.openastronomy.org/en/latest/releasing.html):
the version is derived from git tags by
[`setuptools_scm`](https://setuptools-scm.readthedocs.io/), and the release is
built, tested and uploaded by CI.

## Version numbers

There is no version string to edit. `setuptools_scm` runs `git describe` and
turns the distance from the last tag into a PEP 440 version:

| Repository state | Version |
|---|---|
| exactly on tag `v0.1.0` | `0.1.0` |
| 22 commits after `v0.1.0` | `0.1.1.dev22+g<hash>` |
| uncommitted changes | `0.1.1.dev22+g<hash>.d20261030` |

Release tags must look like `vX.Y.Z`. `ftsdata/version.py` is generated at
build time and must not be edited or committed.

Because a dirty working tree changes the version, releases are always built by
CI from the tagged commit.

## Release checklist

1. Make sure `main` is green and the changelog (if any) is up to date.
2. Create and push an annotated tag:

   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

   If the GitHub mirror is used for the publish workflow, push the tag there
   too:

   ```bash
   git push <github-remote> v0.1.0
   ```

3. Let CI build, test and publish (see below).
4. Check <https://pypi.org/project/ftsdata/> and install from PyPI to confirm.
5. Update the installation section of `README.rst` once the first release is
   available.

## GitHub Actions (default)

`.github/workflows/publish.yml` calls the OpenAstronomy reusable workflow
`publish_pure_python.yml@v1`. It runs on pushes to `main`, on `v*` tags and on
manual dispatch; only `v*` tags are uploaded, so pushes to `main` and manual
runs are safe dry runs of the full build-and-test path.

The workflow:

- builds the sdist and the pure-Python wheel,
- installs the sdist in a clean virtual environment with the `all,test`
  extras,
- runs `pytest --pyargs ftsdata`,
- uploads `dist/*` to PyPI when the ref is a `v*` tag.

One-off setup: create a PyPI API token and store it as the `pypi_token`
repository (or organisation) secret in GitHub. The reusable workflow uses
token authentication, not PyPI Trusted Publishing.

## GitLab CI (optional)

`.gitlab-ci.yml` contains a `publish-pypi` job that mirrors the GitHub
workflow. It is **manual** and only exists in pipelines for `v*` tags, so
nothing is published without an explicit action:

1. push the tag,
2. wait for the tag pipeline's test jobs to pass,
3. press *play* on `publish-pypi`.

The job builds the distributions from the tag, runs `twine check`, then
uploads them. It expects a masked and protected `PYPI_API_TOKEN` CI/CD
variable (`TWINE_USERNAME` is set to `__token__` in the job). Protect the `v*`
tag pattern as well, so only maintainers can create release tags.

Remove the job if releases are managed through GitHub Actions only.

## Manual upload

The CI jobs can always be reproduced locally from a clean checkout of the tag:

```bash
python -m pip install --upgrade build twine
rm -rf dist/
python -m build
python -m twine check dist/*
```

Dry run against TestPyPI:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...        # TestPyPI token
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

python -m pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple ftsdata==0.1.0
```

Publish to PyPI:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...        # PyPI token
python -m twine upload dist/*
```

A version can never be re-uploaded to PyPI: if a release is wrong, bump the
version and tag again.

For a one-off build with an explicit version (for example to inspect the
metadata without tagging):

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 python -m build
```
