use crate::graph::PreferenceGraph;

/// DEPRECATED: MPI via Karp's min-mean-weight cycle on negated savings.
///
/// This computes the minimum mean-weight cycle (Karp 1978) on the negated
/// savings graph. This is the WRONG objective: it divides by edge count, not
/// by the sum of budgets in the cycle. For heterogeneous-budget cycles it
/// over-reports MPI (e.g. gives 0.125 instead of the correct 1/9 ≈ 0.1111
/// for p=[[1,1],[1,2]], q=[[3,1],[1,2]]).
///
/// Correct algorithm: `mpi_min_cycle_ratio` (Smeulders & Spieksma 2013, Thm 2).
///
/// Kept for A/B benchmarking via bench_champion_vs_challenger.py.
/// Not called from the batch Engine path as of 2026-06-09.
///
/// Uses Karp's O(T³) min-mean-weight-cycle on negated weights.
/// Reference: Karp (1978); Echenique, Lee & Shum (2011, JPE).
#[deprecated(
    since = "0.1.0",
    note = "Wrong objective (divides by edge count, not budgets). Use mpi_min_cycle_ratio."
)]
#[allow(dead_code)]
pub fn mpi_karp(graph: &PreferenceGraph) -> f64 {
    let t = graph.t;
    if t < 2 {
        return 0.0;
    }

    let inf = f64::MAX / 2.0;

    // Build NEGATED weight matrix: neg_w[i,j] = -(own_exp[i] - E[i,j]) / own_exp[i]
    // on R edges, +inf elsewhere.
    // Karp's finds min-mean cycle on neg_w, then MPI = -min_mean.
    let mut neg_w = vec![inf; t * t];
    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            if i != j && graph.r[i * t + j] {
                let savings = (graph.own_exp[i] - graph.e[i * t + j]) / graph.own_exp[i];
                neg_w[i * t + j] = -savings;
            }
        }
    }

    // Karp's algorithm:
    // d[k][v] = minimum cost of a path of exactly k edges ending at v
    // d[0][v] = 0 for all v
    //
    // min_mean = min_v { max_{k=0..T-1} { (d[T][v] - d[k][v]) / (T - k) } }

    // d is (T+1) × T matrix, stored flat
    let mut d = vec![inf; (t + 1) * t];

    // d[0][v] = 0 for all v
    for v in 0..t {
        d[0 * t + v] = 0.0;
    }

    // Fill d[k][v] for k = 1..T
    for k in 1..=t {
        for v in 0..t {
            let mut best = inf;
            for u in 0..t {
                if neg_w[u * t + v] < inf / 2.0 {
                    let candidate = d[(k - 1) * t + u] + neg_w[u * t + v];
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            d[k * t + v] = best;
        }
    }

    // Compute min mean weight
    let mut min_mean = inf;
    for v in 0..t {
        if d[t * t + v] >= inf / 2.0 {
            continue; // v is not reachable via T edges
        }
        let mut max_ratio = f64::NEG_INFINITY;
        for k in 0..t {
            if d[k * t + v] < inf / 2.0 {
                let ratio = (d[t * t + v] - d[k * t + v]) / ((t - k) as f64);
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
        if max_ratio < min_mean {
            min_mean = max_ratio;
        }
    }

    if min_mean >= inf / 2.0 {
        return 0.0; // No cycles exist
    }

    // MPI = -min_mean (negate back)
    (-min_mean).max(0.0)
}

/// DEPRECATED: MPI via Karp's algorithm with sparse predecessor lists.
///
/// Same wrong objective as mpi_karp() (divides by edge count, not sum of
/// budgets). Kept for A/B benchmarking. See mpi_karp() doc for details.
///
/// For R-density d, inner loop does d·T work per (k,v) pair instead of T.
/// At typical 30-40% density, this saves ~60-70% of iterations vs mpi_karp.
#[deprecated(
    since = "0.1.0",
    note = "Wrong objective (divides by edge count, not budgets). Use mpi_min_cycle_ratio."
)]
#[allow(dead_code)]
pub fn mpi_karp_v2(graph: &PreferenceGraph) -> f64 {
    let t = graph.t;
    if t < 2 {
        return 0.0;
    }

    let inf = f64::MAX / 2.0;

    // Build sparse predecessor lists: preds[v] = [(u, neg_weight(u->v)), ...]
    // Only includes edges where R[u,v] is true (neg_w < inf).
    let mut preds: Vec<Vec<(usize, f64)>> = vec![Vec::new(); t];
    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            if i != j && graph.r[i * t + j] {
                let savings = (graph.own_exp[i] - graph.e[i * t + j]) / graph.own_exp[i];
                preds[j].push((i, -savings));
            }
        }
    }

    // Karp's algorithm with sparse inner loop:
    // d[k][v] = minimum cost of a path of exactly k edges ending at v
    let mut d = vec![inf; (t + 1) * t];

    // d[0][v] = 0 for all v
    for v in 0..t {
        d[v] = 0.0;
    }

    // Fill d[k][v] for k = 1..T using sparse predecessor lists
    for k in 1..=t {
        for v in 0..t {
            let mut best = inf;
            for &(u, neg_w) in &preds[v] {
                let candidate = d[(k - 1) * t + u] + neg_w;
                if candidate < best {
                    best = candidate;
                }
            }
            d[k * t + v] = best;
        }
    }

    // Compute min mean weight (same as original)
    let mut min_mean = inf;
    for v in 0..t {
        if d[t * t + v] >= inf / 2.0 {
            continue;
        }
        let mut max_ratio = f64::NEG_INFINITY;
        for k in 0..t {
            if d[k * t + v] < inf / 2.0 {
                let ratio = (d[t * t + v] - d[k * t + v]) / ((t - k) as f64);
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
        if max_ratio < min_mean {
            min_mean = max_ratio;
        }
    }

    if min_mean >= inf / 2.0 {
        return 0.0;
    }

    (-min_mean).max(0.0)
}

/// Fast approximate MPI: max per-edge savings across violation pairs.
///
/// This is NOT the theoretical MPI (which uses cycle ratios), but a quick
/// upper bound. Use `mpi_min_cycle_ratio` for the theory-correct value.
///
/// Requires: graph has closure, P, and E computed (call garp_check first).
pub fn mpi_fast(graph: &PreferenceGraph) -> f64 {
    let t = graph.t;
    let mut max_mpi = 0.0f64;

    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            if graph.r_star[i * t + j] && graph.p[j * t + i] {
                let own_j = graph.e[j * t + j];
                if own_j > 0.0 {
                    let savings = (own_j - graph.e[j * t + i]) / own_j;
                    if savings > max_mpi {
                        max_mpi = savings;
                    }
                }
            }
        }
    }

    max_mpi.max(0.0)
}

/// MPI via minimum cost-to-budget cycle ratio.
///
/// The Money Pump Index is the maximum, over revealed-preference cycles, of
/// (sum of savings) / (sum of budgets). Smeulders & Spieksma (2013) JPE 121(6)
/// Theorem 2 and Megiddo (1979) show this equals the minimum cost-to-budget
/// cycle ratio, found via Lawler's binary search with Bellman-Ford
/// negative-cycle detection.
///
/// Edge definition (i → j, exists iff R[i,j] and i ≠ j):
///   cost   = E[i,j] - own_exp[i]   (≤ 0 when affordable; negative = savings)
///   budget = own_exp[i]
///
/// Binary search λ ∈ [−1, 0]. For reweighted cost c − λ·b:
///   - negative cycle exists  →  λ > min_ratio  →  hi = mid
///   - zero-residual cycle    →  λ = min_ratio  →  return exact value
///   - no cycle               →  λ < min_ratio  →  lo = mid
///
/// MPI = −min_ratio, clipped to [0, 1].
/// Convergence tolerance: 1e-10; max iterations: 100.
///
/// Mirrors Python `compute_mpi_bounds` (maximum bound) in
/// src/prefgraph/algorithms/mpi.py. Both paths now agree on the
/// correct ratio-of-sums objective.
///
/// Reference: Smeulders & Spieksma (2013) Thm 2; Megiddo (1979);
///            Echenique, Lee & Shum (2011) JPE 119(6) Eq. (2).
pub fn mpi_min_cycle_ratio(graph: &PreferenceGraph) -> f64 {
    let t = graph.t;
    if t < 2 {
        return 0.0;
    }

    let tol = graph.tolerance;

    // Build adjacency list for the ratio graph.
    // Edge i→j exists iff R[i,j] (E[i,j] ≤ own_exp[i] + tol) and i ≠ j.
    let mut adj: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); t];
    let mut has_edges = false;

    for i in 0..t {
        let budget = graph.own_exp[i];
        if budget <= tol {
            continue;
        }
        for j in 0..t {
            if i == j {
                continue;
            }
            if graph.r[i * t + j] {
                let cost = graph.e[i * t + j] - budget; // ≤ 0
                adj[i].push((j, cost, budget));
                has_edges = true;
            }
        }
    }

    if !has_edges || !has_cycle_adj(&adj, t) {
        return 0.0;
    }

    // Binary search for λ* = min cycle cost/budget ratio in [−1, 0].
    let convergence_tol = 1e-10f64;
    let max_iter = 100usize;
    let mut lo = -1.0f64;
    let mut hi = 0.0f64;

    for _ in 0..max_iter {
        if (hi - lo).abs() <= convergence_tol {
            break;
        }
        let mid = (lo + hi) / 2.0;
        let (has_neg, dists) = bellman_ford_ratio(&adj, t, mid, tol);
        if has_neg {
            hi = mid;
        } else {
            if zero_residual_has_cycle(&adj, t, &dists, mid, tol) {
                // Exact minimum ratio found.
                return (-mid).max(0.0).min(1.0);
            }
            lo = mid;
        }
    }

    let min_ratio = (lo + hi) / 2.0;
    (-min_ratio).max(0.0).min(1.0)
}

/// Cycle detection via Kahn's topological sort.
/// Returns true iff the graph contains at least one cycle.
fn has_cycle_adj(adj: &[Vec<(usize, f64, f64)>], n: usize) -> bool {
    let mut indegree = vec![0usize; n];
    for edges in adj.iter() {
        for &(j, _, _) in edges {
            indegree[j] += 1;
        }
    }
    let mut stack: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut visited = 0usize;
    while let Some(node) = stack.pop() {
        visited += 1;
        for &(j, _, _) in &adj[node] {
            indegree[j] -= 1;
            if indegree[j] == 0 {
                stack.push(j);
            }
        }
    }
    visited < n
}

/// Bellman-Ford with super-source (all distances initialised to 0).
///
/// Reweights each edge (i→j) as: w = cost − lambda * budget.
/// Runs at most n−1 relaxation passes (terminates early if stable).
/// Returns (has_negative_cycle, distance_array).
fn bellman_ford_ratio(
    adj: &[Vec<(usize, f64, f64)>],
    n: usize,
    lambda: f64,
    tol: f64,
) -> (bool, Vec<f64>) {
    let mut dist = vec![0.0f64; n];
    let relax_iters = n.saturating_sub(1);

    for _ in 0..relax_iters {
        let mut changed = false;
        for i in 0..n {
            for &(j, cost, budget) in &adj[i] {
                let w = cost - lambda * budget;
                let candidate = dist[i] + w;
                if candidate < dist[j] - tol {
                    dist[j] = candidate;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // One more pass: any further relaxation means a negative cycle exists.
    for i in 0..n {
        for &(j, cost, budget) in &adj[i] {
            let w = cost - lambda * budget;
            if dist[i] + w < dist[j] - tol {
                return (true, dist);
            }
        }
    }

    (false, dist)
}

/// Build the zero-residual graph and check it for cycles.
///
/// An edge (i→j) appears in the zero-residual graph iff its reduced cost
/// |w + dist[i] − dist[j]| ≤ max(tol, 1e-9), where w = cost − lambda * budget.
fn zero_residual_has_cycle(
    adj: &[Vec<(usize, f64, f64)>],
    n: usize,
    dists: &[f64],
    lambda: f64,
    tol: f64,
) -> bool {
    let edge_tol = tol.max(1e-9);
    let mut residual: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for &(j, cost, budget) in &adj[i] {
            let w = cost - lambda * budget;
            let reduced = w + dists[i] - dists[j];
            if reduced.abs() <= edge_tol {
                residual[i].push((j, 0.0, budget));
            }
        }
    }
    has_cycle_adj(&residual, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garp::garp_check_with_closure;

    // -------------------------------------------------------------------------
    // Tests for mpi_min_cycle_ratio (the correct objective).
    //
    // Expected values are derived from the Python reference implementation:
    //   src/prefgraph/algorithms/mpi.py  compute_mpi_bounds(...).maximum_mpi
    // Run with: python3.11 -c "from prefgraph.algorithms.mpi import compute_mpi_bounds; ..."
    // -------------------------------------------------------------------------

    #[test]
    fn test_mpi_ratio_zero_when_consistent() {
        // Consistent 2-obs data: obs0 at high p1, obs1 at high p2.
        // Python compute_mpi_bounds: maximum_mpi = 0.0
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let mpi = mpi_min_cycle_ratio(&graph);
        assert_eq!(mpi, 0.0, "Expected 0.0 for consistent data, got {mpi}");
    }

    #[test]
    fn test_mpi_ratio_positive_when_violation() {
        // 2-obs violation with equal budgets (own_exp[0] = own_exp[1] = 8).
        // Karp and ratio agree when budgets are equal.
        // Python compute_mpi_bounds: maximum_mpi = 0.125 = 1/8.
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let mpi = mpi_min_cycle_ratio(&graph);
        // Exact: savings=(8-7)+(8-7)=2, budgets=8+8=16, MPI=2/16=0.125
        assert!(
            (mpi - 0.125).abs() < 1e-8,
            "Expected 0.125 for equal-budget 2-cycle, got {mpi}"
        );
    }

    #[test]
    fn test_mpi_ratio_task_fixture_unequal_budgets() {
        // Task fixture: p=[[1,1],[1,2]], q=[[3,1],[1,2]].
        // own_exp[0]=4, own_exp[1]=5.
        // Cycle 0→1→0: savings=(4-3)+(5-5)=1, budgets=4+5=9, MPI=1/9.
        // Old Karp (wrong): divides by edge count → (1/4+0)/2 = 1/8 = 0.125.
        // Python compute_mpi_bounds: maximum_mpi = 0.111111... = 1/9.
        let prices = [1.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 1.0, 1.0, 2.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let mpi = mpi_min_cycle_ratio(&graph);
        let expected = 1.0 / 9.0;
        assert!(
            (mpi - expected).abs() < 1e-8,
            "Expected 1/9 ≈ {expected:.8} for task fixture, got {mpi:.8} (old Karp gave 0.125)"
        );
    }

    #[test]
    fn test_mpi_ratio_3obs() {
        // 3-obs data. Python compute_mpi_bounds: maximum_mpi = 4/17 ≈ 0.235294.
        // Worst cycle: 0↔1 (savings=(8.5-6.5)+(8.5-6.5)=4, budgets=8.5+8.5=17).
        // Karp agrees here (budgets in worst cycle are equal); test is still
        // a correctness anchor for the new function.
        let prices = [2.0, 1.0, 1.5, 1.0, 2.0, 1.5, 1.5, 1.5, 2.0];
        let quantities = [3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0];
        let mut graph = PreferenceGraph::new(3);
        graph.parse_budget(&prices, &quantities, 3, 3, 1e-10);
        let mpi = mpi_min_cycle_ratio(&graph);
        let expected = 4.0 / 17.0; // Python: 0.23529411764705882
        assert!(
            (mpi - expected).abs() < 1e-6,
            "Expected 4/17 ≈ {expected:.8} for 3-obs data, got {mpi:.8}"
        );
    }

    #[test]
    fn test_mpi_ratio_le_mpi_fast() {
        // min-cycle-ratio MPI ≤ fast per-edge upper bound (always).
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check_with_closure(&mut graph); // need closure for mpi_fast
        let ratio_mpi = mpi_min_cycle_ratio(&graph);
        let fast = mpi_fast(&graph);
        assert!(
            ratio_mpi <= fast + 1e-10,
            "min-cycle-ratio MPI ({ratio_mpi}) should be ≤ fast MPI ({fast})"
        );
    }

    // -------------------------------------------------------------------------
    // Karp champion/challenger parity tests (deprecated functions, kept for
    // backward-compat benchmarking). These test internal consistency of the
    // two Karp variants, NOT the correctness of the MPI objective.
    // -------------------------------------------------------------------------

    #[test]
    fn test_mpi_karp_v2_matches_champion() {
        // Both Karp variants produce identical (wrong-objective) values.
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        #[allow(deprecated)]
        let champion = mpi_karp(&graph);
        #[allow(deprecated)]
        let challenger = mpi_karp_v2(&graph);
        assert!(
            (champion - challenger).abs() < 1e-12,
            "Karp parity: champion={champion}, challenger={challenger}"
        );
    }

    #[test]
    fn test_mpi_karp_v2_3obs_parity() {
        // 3-observation Karp parity test.
        let prices = [2.0, 1.0, 1.5, 1.0, 2.0, 1.5, 1.5, 1.5, 2.0];
        let quantities = [3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0];
        let mut graph = PreferenceGraph::new(3);
        graph.parse_budget(&prices, &quantities, 3, 3, 1e-10);
        #[allow(deprecated)]
        let champion = mpi_karp(&graph);
        #[allow(deprecated)]
        let challenger = mpi_karp_v2(&graph);
        assert!(
            (champion - challenger).abs() < 1e-12,
            "3obs Karp parity: champion={champion}, challenger={challenger}"
        );
    }

    #[test]
    fn test_mpi_fast_backward_compat() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let garp = garp_check_with_closure(&mut graph);
        assert!(!garp.is_consistent);
        let mpi = mpi_fast(&graph);
        assert!(mpi > 0.0);
    }
}
