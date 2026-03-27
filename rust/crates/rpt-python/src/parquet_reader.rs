//! Rust-native Parquet reader for streaming revealed preference analysis.
//!
//! Reads Parquet files directly, groups observations by user, and produces
//! per-user flat f64 arrays ready for `PreferenceGraph::parse_budget()`.
//!
//! Gated behind `#[cfg(feature = "parquet")]`.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, StringArray, Int64Array};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Per-user accumulated observations (wide format).
///
/// Each entry is one observation's worth of (costs, actions) stored flat.
struct UserAccum {
    /// Flat cost values: obs0_cost0, obs0_cost1, ..., obs1_cost0, ...
    costs: Vec<f64>,
    /// Flat action values, same layout as costs
    actions: Vec<f64>,
    /// Number of observations accumulated
    n_obs: usize,
    /// Number of items (K) per observation
    k: usize,
}

impl UserAccum {
    fn new(k: usize) -> Self {
        UserAccum {
            costs: Vec::new(),
            actions: Vec::new(),
            n_obs: 0,
            k,
        }
    }

    fn push_obs(&mut self, costs: &[f64], actions: &[f64]) {
        self.costs.extend_from_slice(costs);
        self.actions.extend_from_slice(actions);
        self.n_obs += 1;
    }

    fn into_flat(self) -> (Vec<f64>, Vec<f64>, usize, usize) {
        (self.costs, self.actions, self.n_obs, self.k)
    }
}

/// Read a wide-format Parquet file and return per-user flat arrays.
///
/// Returns Vec of (user_id, prices_flat, quantities_flat, T, K).
pub fn read_parquet_users_wide(
    path: &str,
    user_col: &str,
    cost_cols: &[String],
    action_cols: &[String],
    chunk_size: usize,
) -> Result<Vec<Vec<(String, Vec<f64>, Vec<f64>, usize, usize)>>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Invalid Parquet: {}", e))?;

    let reader = builder
        .with_batch_size(65536)
        .build()
        .map_err(|e| format!("Cannot build reader: {}", e))?;

    let k = cost_cols.len();
    let mut accum: HashMap<String, UserAccum> = HashMap::new();
    let mut user_order: Vec<String> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| format!("Read error: {}", e))?;
        let schema = batch.schema();

        // Find user column and extract as strings
        let user_col_idx = schema
            .index_of(user_col)
            .map_err(|_| format!("Column '{}' not found", user_col))?;
        let user_array = batch.column(user_col_idx);
        let user_ids = column_to_strings(user_array, user_col)?;

        // Find cost and action column indices
        let cost_indices: Vec<usize> = cost_cols
            .iter()
            .map(|c| schema.index_of(c).map_err(|_| format!("Column '{}' not found", c)))
            .collect::<Result<_, _>>()?;
        let action_indices: Vec<usize> = action_cols
            .iter()
            .map(|c| schema.index_of(c).map_err(|_| format!("Column '{}' not found", c)))
            .collect::<Result<_, _>>()?;

        // Extract cost and action columns as f64
        let cost_arrays: Vec<&Float64Array> = cost_indices
            .iter()
            .map(|&i| {
                batch.column(i).as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| format!("Column '{}' is not Float64", cost_cols[i - cost_indices[0]]))
            })
            .collect::<Result<_, _>>()?;
        let action_arrays: Vec<&Float64Array> = action_indices
            .iter()
            .map(|&i| {
                batch.column(i).as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| format!("Column '{}' is not Float64", action_cols[i - action_indices[0]]))
            })
            .collect::<Result<_, _>>()?;

        let n_rows = batch.num_rows();
        let mut row_costs = vec![0.0f64; k];
        let mut row_actions = vec![0.0f64; k];

        for row in 0..n_rows {
            let uid = &user_ids[row];

            // Extract this row's costs and actions
            for (col, arr) in cost_arrays.iter().enumerate() {
                row_costs[col] = arr.value(row);
            }
            for (col, arr) in action_arrays.iter().enumerate() {
                row_actions[col] = arr.value(row);
            }

            if !accum.contains_key(uid) {
                user_order.push(uid.clone());
                accum.insert(uid.clone(), UserAccum::new(k));
            }
            accum.get_mut(uid).unwrap().push_obs(&row_costs, &row_actions);
        }
    }

    // Convert accumulated data into chunks
    let mut chunks: Vec<Vec<(String, Vec<f64>, Vec<f64>, usize, usize)>> = Vec::new();
    let mut current_chunk: Vec<(String, Vec<f64>, Vec<f64>, usize, usize)> = Vec::new();

    for uid in user_order {
        if let Some(user_accum) = accum.remove(&uid) {
            if user_accum.n_obs >= 2 {
                let (costs, actions, t, k) = user_accum.into_flat();
                current_chunk.push((uid, costs, actions, t, k));

                if current_chunk.len() >= chunk_size {
                    chunks.push(std::mem::take(&mut current_chunk));
                }
            }
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    Ok(chunks)
}

/// Convert an Arrow array column to a Vec<String>, handling String and Int types.
fn column_to_strings(array: &Arc<dyn Array>, col_name: &str) -> Result<Vec<String>, String> {
    match array.data_type() {
        DataType::Utf8 => {
            let str_array = array.as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| format!("Cannot cast '{}' to StringArray", col_name))?;
            Ok((0..str_array.len()).map(|i| str_array.value(i).to_string()).collect())
        }
        DataType::LargeUtf8 => {
            let str_array = array.as_any().downcast_ref::<arrow::array::LargeStringArray>()
                .ok_or_else(|| format!("Cannot cast '{}' to LargeStringArray", col_name))?;
            Ok((0..str_array.len()).map(|i| str_array.value(i).to_string()).collect())
        }
        DataType::Int64 => {
            let int_array = array.as_any().downcast_ref::<Int64Array>()
                .ok_or_else(|| format!("Cannot cast '{}' to Int64Array", col_name))?;
            Ok((0..int_array.len()).map(|i| int_array.value(i).to_string()).collect())
        }
        DataType::Int32 => {
            let int_array = array.as_any().downcast_ref::<arrow::array::Int32Array>()
                .ok_or_else(|| format!("Cannot cast '{}' to Int32Array", col_name))?;
            Ok((0..int_array.len()).map(|i| int_array.value(i).to_string()).collect())
        }
        other => Err(format!(
            "Column '{}' has unsupported type {:?}. Expected String, Int32, or Int64.",
            col_name, other
        )),
    }
}
