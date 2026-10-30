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

### One release tag per commit

Never put two release tags on the same commit, for example `v0.1.0rc1` and
`v0.1.0`. `git describe` cannot tell which tag is meant, so `setuptools_scm`
may report either version: a `v0.1.0` tag can end up building `0.1.0rc1`.

This is not hypothetical. On 2026-10-30 the first `v0.1.0` tag was placed on
the same commit as `v0.1.0rc1`, the GitHub workflow rebuilt `0.1.0rc1`, and
PyPI rejected the duplicate files with `HTTPError: 400 Bad Request`.

Tag a later commit than the previous release: a follow-up fix, a documentation
change, or, if nothing else changed, an empty commit.

The GitHub workflow checks this before building. It compares the version
reported by `setuptools_scm` with the tag name and aborts with a clear error
instead of building or uploading the wrong release.

## Release checklist

1. Make sure `main` is green and the changelog (if any) is up to date.
2. Create an annotated tag on the commit you intend to release, verify it,
   then push it:

   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   ```

   Check that the tag resolves to the version you intend to publish:

   ```bash
   git tag --points-at HEAD               # must list only v0.1.0
   python -m pip install setuptools_scm   # once
   python -m setuptools_scm               # must print 0.1.0
   ```

   The version must equal the tag name without the leading `v`, and `HEAD`
   must carry no other release tag. Then push:

   ```bash
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

## Moving or re-creating a tag

If a release tag points at the wrong commit, delete it everywhere and create
it again **before** PyPI has accepted the version:

```bash
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0
git push <github-remote> :refs/tags/v0.1.0
# make sure the intended commit is on main first
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
git push <github-remote> v0.1.0
```

Protected tag patterns (GitLab `v*`, GitHub tag rulesets) can block deleting or
re-pushing a tag; relax them temporarily. Once PyPI has accepted a version,
never re-tag it: bump the version instead.

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
uploads them. The upload uses `--skip-existing`: if the GitHub workflow has
already published the same tag (or the job is re-run), the files that are
already on PyPI are skipped instead of failing with an HTTP 400 "File already
exists". It expects a masked and protected `PYPI_API_TOKEN` CI/CD variable
(`TWINE_USERNAME` is set to `__token__` in the job). Protect the `v*` tag
pattern as well, so only maintainers can create release tags.

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
version and tag again. If only some files were already uploaded, add
`--skip-existing` so a retry uploads just the missing ones.

For a one-off build with an explicit version (for example to inspect the
metadata without tagging):

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 python -m build
```
