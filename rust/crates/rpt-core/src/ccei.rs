use crate::garp::garp_check;
use crate::graph::PreferenceGraph;
use crate::types::CceiResult;

/// Compute CCEI (Critical Cost Efficiency Index) via discrete binary search.
///
/// Uses the O(T²) SCC-based GARP check (Talla Nobibon et al. 2015) inside the
/// binary search loop, avoiding the O(T³) transitive closure per iteration.
/// Total complexity: O(T² log T) - provably optimal up to log factors.
///
/// Requires: graph has expenditure built (call parse_budget or ensure_expenditure first).
/// Reuses the expenditure matrix - only rebuilds R/P at each efficiency level.
///
/// References:
///   Afriat (1967), "Construction of Utility Functions from Expenditure Data", IER.
///   Smeulders et al. (2014), "Goodness-of-Fit Measures for RP Tests", ACM TEAC.
pub fn ccei_search(graph: &mut PreferenceGraph, tolerance: f64) -> CceiResult {
    let t = graph.t;

    // Quick check: is data already GARP-consistent at e=1?
    graph.ensure_r(tolerance);
    if garp_check(graph).is_consistent {
        return CceiResult {
            ccei: 1.0,
            iterations: 0,
            is_perfectly_consistent: true,
        };
    }

    // Candidate breakpoints: the T^2 expenditure ratios E[i,j]/own_exp[i] that
    // fall in (0, 1). GARP-consistency only changes as e crosses one of these.
    let mut candidates: Vec<f64> = Vec::with_capacity(t * t);
    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            let ratio = graph.e[i * t + j] / graph.own_exp[i];
            if ratio > 0.0 && ratio < 1.0 {
                candidates.push(ratio);
            }
        }
    }
    candidates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()); // ascending
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    let n = candidates.len();

    // The CCEI is the supremum sup{e : GARP holds at e}. GARP-consistency on the
    // open interval (c[k-1], c[k]) is monotone in k (raising e only adds revealed-
    // preference edges), so the supremum is the UPPER breakpoint of the highest
    // interval on which GARP still holds: every e in that open interval is
    // feasible, so its supremum c[k] is the index even though GARP fails exactly
    // *at* c[k] by a boundary tie (Smeulders et al. 2014, Algorithm 2). Each
    // interval is tested at its midpoint, where no ratio tie occurs so the
    // relations are unambiguous; a sentinel boundary at 1.0 covers (c[n-1], 1).
    // Binary search for the largest interval index that still holds.
    let mut lo = 0usize;
    let mut hi = n;
    let mut best_k = 0usize; // (0, c[0]) is always feasible: no edges active yet
    let mut iterations = 0u32;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let lower = if mid == 0 { 0.0 } else { candidates[mid - 1] };
        let upper = if mid >= n { 1.0 } else { candidates[mid] };
        let probe = 0.5 * (lower + upper);
        iterations += 1;
        // Exact comparison (tolerance 0): the midpoint never equals a ratio.
        graph.build_r_at_efficiency(probe, 0.0);
        if garp_check(graph).is_consistent {
            best_k = mid;
            lo = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        }
    }
    let ccei = if best_k >= n { 1.0 } else { candidates[best_k] };

    // Restore R/P at e=1.0 so downstream algorithms get the right state
    graph.build_r_at_efficiency(1.0, tolerance);
    graph.ensure_closure();

    CceiResult {
        ccei,
        iterations,
        is_perfectly_consistent: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_ccei_is_one() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = ccei_search(&mut graph, 1e-10);
        assert_eq!(result.ccei, 1.0);
        assert!(result.is_perfectly_consistent);
    }

    #[test]
    fn test_violation_ccei_below_one() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = ccei_search(&mut graph, 1e-10);
        assert!(result.ccei < 1.0);
        assert!(result.ccei > 0.0);
        assert!((result.ccei - 0.875).abs() < 0.001);
    }

    #[test]
    fn test_ccei_restores_state() {
        // After CCEI, graph should have R/P at e=1.0
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = ccei_search(&mut graph, 1e-10);
        // Closure should be at e=1.0 - GARP violations should exist
        assert!(graph.has_closure);
        let t = graph.t;
        let mut has_violation = false;
        for i in 0..t {
            for j in 0..t {
                if graph.r_star[i * t + j] && graph.p[j * t + i] {
                    has_violation = true;
                }
            }
        }
        assert!(has_violation);
    }

    #[test]
    fn test_ccei_uses_supremum_at_open_breakpoint() {
        let prices = [1.0, 3.0, 3.0, 1.0, 0.4, 1.0];
        let quantities = [2.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut graph = PreferenceGraph::new(3);
        graph.parse_budget(&prices, &quantities, 3, 2, 1e-10);

        let result = ccei_search(&mut graph, 1e-10);

        assert!((result.ccei - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_ccei_supremum_with_large_own_expenditure() {
        // Regression for the one-ULP probe: it only recovered the supremum when
        // the binding own-expenditure was ~1. Here own_exp = 25, and the true
        // CCEI is the supremum 23/25 = 0.92 (GARP holds for every e < 0.92 and
        // fails at the 0.92 tie). The old probe returned 5/31 ~= 0.161.
        let prices = [3.0, 5.0, 5.0, 1.0];
        let quantities = [0.0, 5.0, 6.0, 1.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = ccei_search(&mut graph, 1e-10);
        assert!((result.ccei - 0.92).abs() < 1e-9, "ccei = {}", result.ccei);
    }
}
