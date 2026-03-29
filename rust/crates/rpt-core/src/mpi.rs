use crate::graph::PreferenceGraph;

/// LEGACY: MPI via Karp's algorithm with dense inner loop.
///
/// Superseded by mpi_karp_v2() which uses sparse predecessor lists for a 4.3x
/// speedup at T=500 (benchmarked 2026-03-29). Kept for A/B comparison via
/// bench_champion_vs_challenger.py. Both produce identical results.
///
/// Performance note: the inner loop (line ~58) scans all T nodes per (k,v) pair
/// even when most edges are +inf (no R edge). At typical 30-40% R-density,
/// ~60-70% of iterations do no useful work. The v2 variant avoids this by
/// pre-building sparse predecessor lists.
///
/// Uses Karp's O(T³) min-mean-weight-cycle on negated weights.
/// Reference: Karp (1978); Echenique, Lee & Shum (2011, JPE).
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

/// DEFAULT: MPI via Karp's algorithm with sparse predecessor lists.
///
/// Same algorithm as mpi_karp(), but pre-builds adjacency lists so the
/// inner loop only iterates over actual R-edges instead of scanning all T nodes.
/// Reduces O(T³) inner work to O(T · |E|) where |E| = number of R-edges.
///
/// For R-density d, inner loop does d·T work per (k,v) pair instead of T.
/// At typical 30-40% density, this saves ~60-70% of iterations.
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
/// This is NOT the theoretical MPI (which uses cycle means), but a quick
/// upper bound. Use `mpi_karp` for the theory-correct value.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garp::garp_check_with_closure;

    #[test]
    fn test_mpi_karp_zero_when_consistent() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let mpi = mpi_karp(&graph);
        assert_eq!(mpi, 0.0);
    }

    #[test]
    fn test_mpi_karp_positive_when_violation() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let mpi = mpi_karp(&graph);
        assert!(mpi > 0.0, "MPI should be positive for violation data, got {mpi}");
        // For this 2-cycle data, Karp's MPI should be close to the cycle average
        assert!(mpi < 0.5, "MPI should be moderate, got {mpi}");
    }

    #[test]
    fn test_mpi_karp_v2_matches_champion() {
        // Champion/challenger parity: both must return identical MPI values
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let champion = mpi_karp(&graph);
        let challenger = mpi_karp_v2(&graph);
        assert!(
            (champion - challenger).abs() < 1e-12,
            "MPI mismatch: champion={champion}, challenger={challenger}"
        );
    }

    #[test]
    fn test_mpi_karp_v2_consistent_is_zero() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        assert_eq!(mpi_karp_v2(&graph), 0.0);
    }

    #[test]
    fn test_mpi_karp_v2_3obs_parity() {
        // 3-observation test: champion and challenger must agree
        let prices = [2.0, 1.0, 1.5, 1.0, 2.0, 1.5, 1.5, 1.5, 2.0];
        let quantities = [3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0];
        let mut graph = PreferenceGraph::new(3);
        graph.parse_budget(&prices, &quantities, 3, 3, 1e-10);
        let champion = mpi_karp(&graph);
        let challenger = mpi_karp_v2(&graph);
        assert!(
            (champion - challenger).abs() < 1e-12,
            "3obs MPI mismatch: champion={champion}, challenger={challenger}"
        );
    }

    #[test]
    fn test_mpi_karp_le_mpi_fast() {
        // Karp's cycle-mean MPI should be ≤ the fast per-edge max
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check_with_closure(&mut graph); // need closure for mpi_fast
        let karp = mpi_karp(&graph);
        let fast = mpi_fast(&graph);
        assert!(
            karp <= fast + 1e-10,
            "Karp MPI ({karp}) should be <= fast MPI ({fast})"
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
