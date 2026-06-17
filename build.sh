#!/usr/bin/env bash
# Script to automate build for PyPI.

set -euo pipefail

rm -fr dist
black --line-length=79 wnstatmech
python -m pylint wnstatmech
python -m pytest .github/workflows/wnstatmech_test.py

python -m pip install --upgrade build
python -m build
python -m pip install --upgrade twine

echo ""
echo "All version numbers must be the same:"
echo ""

grep version wnstatmech/__about__.py | grep -v ","
grep version CITATION.cff | grep -v "cff-version"
grep Version doc/source/changelog.rst | grep -v Versioning | head -1
grep version pyproject.toml

echo ""
echo "Check the release date:"
echo ""
grep date CITATION.cff
echo ""
