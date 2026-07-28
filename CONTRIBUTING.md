# Contributing to wnstatmech

Thank you for contributing to `wnstatmech`. Contributions may include bug
reports, documentation improvements, tests, numerical validation, and code.

## Getting started

Create an environment with a supported Python version, then install the
package in editable mode:

```bash
python -m pip install -e .
```

Install development tools as needed:

```bash
python -m pip install pytest black pylint
```

Run the test suite with:

```bash
pytest tests
```

The release build script also runs formatting, linting, and tests:

```bash
bash build.sh
```

## Reporting issues and seeking support

Use the [issue tracker](https://github.com/mbradle/wnstatmech/issues) for bug
reports, questions, and feature requests. For a numerical issue, include the
particle properties, temperature, chemical potential or number density,
expected and observed results, package version, and a minimal reproducing
example when possible.

## Pull requests

Keep pull requests focused and describe the scientific or numerical motivation
for the change. Please include tests for new behavior and preserve backward
compatibility unless the change is explicitly discussed as breaking.

For changes to numerical methods or approximations, include validation against
an analytic result, a higher-accuracy calculation, or a documented reference
case. Performance changes should include a representative benchmark when they
affect a hot path.

Format modified Python files with Black's 79-character line length and keep
the package lint-clean:

```bash
black --line-length=79 wnstatmech tests benchmarks
pylint wnstatmech --fail-under 9.8
```

Update the changelog, documentation, and tutorial when a user-visible API or
workflow changes.
