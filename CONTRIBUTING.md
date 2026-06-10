# Contributing to PrefGraph

PrefGraph uses Rust for its core algorithms and Python for its interface layer. Maturin is the only build backend. There is no setuptools-rust path.

## Building from source

You need a Rust toolchain (stable, via rustup) and Python 3.10 or later. The Rust binding crate lives at `rust/crates/rpt-python/`, not at the repo root.

Clone the repo and install in editable mode:

```
git clone https://github.com/rawatpranjal/prefgraph.git
cd prefgraph
pip install -e ".[dev]"
```

That command compiles the Rust extension via maturin and installs the package.

**Important gotcha.** Editable installs with maturin are not live for Python source changes the way a pure-Python editable install would be. If you edit a `.py` file, you must rebuild before your changes take effect in tests:

```
python3.11 -m pip install -e . --no-build-isolation
```

For faster iteration on Rust changes only, you can use maturin directly:

```
maturin develop --release
```

## Running the gate set

Before pushing, run the full local gate set in order. All five must pass.

```
ruff check src/
ruff format --check src/
python3.11 -m mypy src/
pytest
cargo test --manifest-path rust/Cargo.toml
```

If `ruff format --check` reports differences, run `ruff format src/` to apply them, then re-check.

The mypy configuration is in `pyproject.toml`. The test suite uses pytest with coverage; configuration is also in `pyproject.toml`.

## Pull request flow

Open PRs against `main`. The CI workflow runs `pytest` and `cargo test` across the support matrix (Python 3.10 through 3.13 on Linux, plus macOS and Windows on 3.12). It must be green before merge.

Keep the PR focused. One logical change per PR makes review faster and keeps the git history readable. Write a short commit message that describes what changed and why.

The algorithms are documented with paper citations in the source. If you fix an algorithm, cite the theorem or definition from the relevant paper and explain why the old behavior was wrong. Golden test values must come from the paper or an independent oracle, not from the code's own output.

New algorithms are frozen for the current 0.6.x cycle. The API surface is locked. Bug fixes, documentation improvements, and test additions are welcome.
