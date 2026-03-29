use crate::closure::scc_transitive_closure_v2 as scc_transitive_closure;
use crate::graph::PreferenceGraph;
use crate::types::GarpResult;

/// Check GAPP (Generalized Axiom of Price Preference).
///
/// The price-preference dual of GARP. Instead of "was bundle j affordable
/// when i was chosen?", GAPP asks "could observation s have sold at
/// observation t's prices and earned at least as much?"
///
/// R_p[s,t] = True iff p^s @ x^t <= p^t @ x^t  (s's prices are cheaper at t's bundle)
/// P_p[s,t] = True iff p^s @ x^t <  p^t @ x^t  (strictly cheaper)
///
/// Violation: R_p*[s,t] AND P_p[t,s]
pub fn gapp_check(graph: &mut PreferenceGraph) -> GarpResult {
    // Requires expenditure already built (call parse_budget first)
    let t = graph.t;

    // Build price preference matrices (transposed logic from GARP)
    // R_p[s,t] = p^s @ x^t <= p^t @ x^t = E[s,t] <= E[t,t]
    let mut r_p = vec![false; t * t];
    let mut p_p = vec![false; t * t];

    for s in 0..t {
        for ti in 0..t {
            r_p[s * t + ti] = graph.e[s * t + ti] <= graph.e[ti * t + ti] + graph.tolerance;
            p_p[s * t + ti] = graph.e[s * t + ti] < graph.e[ti * t + ti] - graph.tolerance;
        }
        p_p[s * t + s] = false;
    }

    // SCC-optimized transitive closure of R_p
    let mut r_p_star = vec![false; t * t];
    let mut labels = vec![0u32; t];
    let (n_comp, max_scc) = scc_transitive_closure(&r_p, t, &mut r_p_star, &mut labels);

    // Violation check: R_p*[s,t] AND P_p[t,s]
    let mut n_violations = 0u32;
    for s in 0..t {
        for ti in 0..t {
            if r_p_star[s * t + ti] && p_p[ti * t + s] {
                n_violations += 1;
            }
        }
    }

    GarpResult {
        is_consistent: n_violations == 0,
        n_violations,
        max_scc_size: max_scc as u32,
        n_components: n_comp as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gapp_consistent() {
        let mut graph = PreferenceGraph::new(2);
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let result = gapp_check(&mut graph);
        assert!(result.is_consistent);
    }
}
