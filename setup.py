"""Optional Rust extension via setuptools-rust.

Pre-built wheels (normal pip install) don't use this file.
Source installs (sdist) use setuptools-rust to compile the Rust
extension if available. If Rust is missing, optional=True lets
the install succeed with the pure-Python fallback.
"""

from setuptools import setup

try:
    from setuptools_rust import Binding, RustExtension

    rust_extensions = [
        RustExtension(
            "pyrevealed._rust_core",
            path="rust/crates/rpt-python/Cargo.toml",
            binding=Binding.PyO3,
            optional=True,
        )
    ]
except ImportError:
    rust_extensions = []

setup(rust_extensions=rust_extensions)
