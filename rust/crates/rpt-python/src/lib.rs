use pyo3::prelude::*;

mod convert;
mod batch;
#[cfg(feature = "parquet")]
mod parquet_reader;

/// PrefGraph Rust core.
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch::analyze_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch::analyze_menu_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch::build_preference_graph, m)?)?;
    #[cfg(feature = "parquet")]
    m.add_function(wrap_pyfunction!(batch::analyze_parquet_file, m)?)?;
    Ok(())
}
