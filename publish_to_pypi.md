# Publishing a Python Package to PyPI via GitHub Actions

This guide explains how to publish your Python package to PyPI using Git, tags, and GitHub Actions.

## ✨ Prerequisites

* A Python project with a configured `pyproject.toml` or `setup.py`.
* A Git repository hosted on GitHub.
* GitHub Secrets configured: `PYPI_USERNAME` and `PYPI_PASSWORD` (or TestPyPI tokens if testing).
* A GitHub Actions workflow set up to trigger on `v*.*.*` tags.

## ✅ Publishing Steps

### 1. Stage your changes

```bash
git add .
```

Adds all modified or newly created files to the Git staging area.

### 2. Commit your changes

```bash
git commit -m "version *.*.*"
```

Creates a new commit with a message describing the version.

### 3. Create a Git tag for release

```bash
git tag v*.*.*
```

Tags the current commit as a release version. This will trigger the deployment pipeline.

### 4. Push the tag to GitHub

```bash
git push origin v*.*.*
```

Pushes the tag to the remote repository. GitHub Actions will detect this and launch the workflow.

### 5. (Optional) Push your working branch

```bash
git push -u origin test
```

Pushes the `test` branch to GitHub and sets it as the upstream.

## 🛡️ Important Notes

* You cannot re-upload a version that already exists on PyPI.
* To test without affecting the official PyPI, use `--repository testpypi` with `twine upload`.

## ✅ Example GitHub Actions Workflow Snippet

```yaml
on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Build and Publish
        run: |
          python -m build
          python -m twine upload dist/*
        env:
          TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
          TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
```

---
