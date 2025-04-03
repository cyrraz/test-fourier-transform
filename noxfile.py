from __future__ import annotations

import nox

nox.options.sessions = ["lint", "tests"]
nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|venv"

PYTHON_ALL_VERSIONS = ["3.10", "3.13"]


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=PYTHON_ALL_VERSIONS, reuse_venv=True)
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install("-e", ".")
    session.install(*nox.project.dependency_groups(pyproject, "test"))
    session.run("pytest", *session.posargs)
