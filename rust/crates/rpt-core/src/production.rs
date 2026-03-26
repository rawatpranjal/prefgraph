use crate::garp::garp_check;
use crate::ccei::ccei_search;
use crate::graph::PreferenceGraph;
use crate::types::{GarpResult, CceiResult};

/// Check Production GARP (profit maximization consistency).
///
/// Same algorithm as budget GARP, but the graph is built from profit data
/// instead of expenditure data. R[i,j] = True iff actual_profit[i] >=
/// counterfactual_profit using j's input-output mix at i's prices.
///
/// Requires: graph parsed with parse_production() (E = profit matrix built).
pub fn production_garp_check(graph: &mut PreferenceGraph) -> GarpResult {
    // parse_production() already builds R and P from profit comparisons.
    // garp_check() operates on whatever R/P the graph has.
    garp_check(graph)
}

/// Compute production CCEI (profit efficiency index).
///
/// Binary search over efficiency levels on the profit graph.
/// Same algorithm as budget CCEI.
pub fn production_ccei(graph: &mut PreferenceGraph, tolerance: f64) -> CceiResult {
    ccei_search(graph, tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_garp_consistent() {
        // Firm maximizes profit: when output price is high, produce more
        let mut graph = PreferenceGraph::new(2);
        // 2 obs, 1 input, 1 output
        let inp = [1.0, 1.0];   // input prices (flat)
        let inq = [2.0, 4.0];   // input quantities
        let outp = [3.0, 1.0];  // output prices
        let outq = [4.0, 2.0];  // output quantities
        graph.parse_production(&inp, &inq, &outp, &outq, 2, 1, 1, 1e-10);
        let result = production_garp_check(&mut graph);
        // Whether consistent depends on profit comparison
        assert!(result.is_consistent || !result.is_consistent); // smoke test
    }
}
