# test-fourier-transform

A set of tests of the Fourier transform.

[![GitHub Actions Status: CI](https://github.com/cyrraz/test-fourier-transform/actions/workflows/ci.yaml/badge.svg)](https://github.com/cyrraz/test-fourier-transform/actions/workflows/ci.yaml?query=branch%3Amain)

## Setting up a development environment

### Nox

The fastest way to start with development is to use `nox`.

`Nox` is an automation tool that helps manage and run development tasks such as testing and linting.
It is especially useful for ensuring your code works across different Python versions and adheres to the project's quality standards.
`Nox` handles everything for you, including setting up a temporary virtual environment for each run.

To set up `nox`:

1. If you don't have it already, install `pipx` by following the instructions on their [website](https://pipx.pypa.io/stable/).
2. Install `nox` using `pipx`:

   ```console
   pipx install nox
   ```

To use `nox`, simply run:

```console
nox
```

This will lint and test the project using multiple Python versions.

You can also run specific nox sessions:

```console
# List all the defined sessions
nox -l

# Run the linter only
nox -s lint

# Run the tests only
nox -s tests
```
