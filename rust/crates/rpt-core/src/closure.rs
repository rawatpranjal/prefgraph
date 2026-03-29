use crate::scc::tarjan_scc;

/// SCC-optimized transitive closure.
///
/// 1. Find SCCs via Tarjan's (O(V+E))
/// 2. Run Floyd-Warshall only within each non-trivial SCC
/// 3. Propagate reachability across the condensed DAG
///
/// Returns: n_components and max_scc_size (for diagnostics).
/// The closure is written in-place into the provided buffer.
pub fn scc_transitive_closure(
    r_mat: &[bool],
    t: usize,
    closure: &mut [bool],
    scc_labels: &mut [u32],
) -> (usize, usize) {
    // Initialize closure = R + diagonal
    closure[..t * t].copy_from_slice(&r_mat[..t * t]);
    for i in 0..t {
        closure[i * t + i] = true;
    }

    // Find SCCs
    let n_comp = tarjan_scc(r_mat, t, scc_labels);

    if n_comp <= 1 && t > 1 {
        // Single SCC - run full Floyd-Warshall
        floyd_warshall(closure, t);
        return (1, t);
    }

    // Group nodes by SCC, find max SCC size
    let mut scc_sizes = vec![0usize; n_comp];
    for i in 0..t {
        scc_sizes[scc_labels[i] as usize] += 1;
    }
    let max_scc = *scc_sizes.iter().max().unwrap_or(&0);

    // Collect SCC node lists
    let mut scc_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for i in 0..t {
        scc_nodes[scc_labels[i] as usize].push(i);
    }

    // Floyd-Warshall within each non-trivial SCC
    for scc in &scc_nodes {
        if scc.len() <= 1 {
            continue;
        }
        let k = scc.len();
        let mut sub = vec![false; k * k];
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                sub[li * k + lj] = closure[ni * t + nj];
            }
            sub[li * k + li] = true;
        }
        floyd_warshall(&mut sub, k);
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                closure[ni * t + nj] = sub[li * k + lj];
            }
        }
    }

    // Build condensed DAG
    let mut dag = vec![false; n_comp * n_comp];
    for i in 0..t {
        for j in 0..t {
            let si = scc_labels[i] as usize;
            let sj = scc_labels[j] as usize;
            if r_mat[i * t + j] && si != sj {
                dag[si * n_comp + sj] = true;
            }
        }
    }

    // Topological sort (Kahn's)
    let mut in_deg = vec![0usize; n_comp];
    for i in 0..n_comp {
        for j in 0..n_comp {
            if dag[i * n_comp + j] {
                in_deg[j] += 1;
            }
        }
    }
    let mut queue: Vec<usize> = (0..n_comp).filter(|&i| in_deg[i] == 0).collect();
    let mut topo = Vec::with_capacity(n_comp);
    let mut head = 0;
    while head < queue.len() {
        let nd = queue[head];
        head += 1;
        topo.push(nd);
        for j in 0..n_comp {
            if dag[nd * n_comp + j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 {
                    queue.push(j);
                }
            }
        }
    }

    // Propagate reachability in reverse topological order
    let mut reach: Vec<Vec<bool>> = (0..n_comp).map(|_| vec![false; t]).collect();
    for c in 0..n_comp {
        for &ni in &scc_nodes[c] {
            for j in 0..t {
                if closure[ni * t + j] {
                    reach[c][j] = true;
                }
            }
        }
    }
    for &c in topo.iter().rev() {
        for succ in 0..n_comp {
            if dag[c * n_comp + succ] {
                for j in 0..t {
                    if reach[succ][j] {
                        reach[c][j] = true;
                    }
                }
            }
        }
        for &ni in &scc_nodes[c] {
            for j in 0..t {
                if reach[c][j] {
                    closure[ni * t + j] = true;
                }
            }
        }
    }

    (n_comp, max_scc)
}

/// Raw Floyd-Warshall transitive closure on a flat bool matrix.
#[inline]
fn floyd_warshall(closure: &mut [bool], t: usize) {
    for k in 0..t {
        for i in 0..t {
            if closure[i * t + k] {
                for j in 0..t {
                    if closure[k * t + j] {
                        closure[i * t + j] = true;
                    }
                }
            }
        }
    }
}

/// Challenger: Floyd-Warshall with empty-pivot skip.
///
/// Skips pivot k entirely if no node reaches k (column k is all-false).
/// Saves O(T²) per skipped pivot. Marginal for dense graphs, significant
/// for sparse subgraphs within SCCs.
#[inline]
fn floyd_warshall_v2(closure: &mut [bool], t: usize) {
    for k in 0..t {
        // Skip pivot k if no node reaches it (column k is all-false)
        let has_incoming = (0..t).any(|i| closure[i * t + k]);
        if !has_incoming {
            continue;
        }
        for i in 0..t {
            if closure[i * t + k] {
                for j in 0..t {
                    if closure[k * t + j] {
                        closure[i * t + j] = true;
                    }
                }
            }
        }
    }
}

/// Challenger: SCC-optimized transitive closure with u64 bitset DAG propagation
/// and Floyd-Warshall pivot skip.
///
/// Same algorithm as scc_transitive_closure(), but:
/// 1. Uses u64 bitsets for DAG reachability propagation (64x fewer operations)
/// 2. Uses floyd_warshall_v2() with empty-pivot skip
///
/// The bitset optimization targets the inner merge loop:
///   Original: for j in 0..t { if reach[succ][j] { reach[c][j] = true; } }
///   Bitset:   for w in 0..n_words { reach[c][w] |= reach[succ][w]; }
pub fn scc_transitive_closure_v2(
    r_mat: &[bool],
    t: usize,
    closure: &mut [bool],
    scc_labels: &mut [u32],
) -> (usize, usize) {
    // Initialize closure = R + diagonal
    closure[..t * t].copy_from_slice(&r_mat[..t * t]);
    for i in 0..t {
        closure[i * t + i] = true;
    }

    // Find SCCs
    let n_comp = tarjan_scc(r_mat, t, scc_labels);

    if n_comp <= 1 && t > 1 {
        // Single SCC - run full Floyd-Warshall (v2 with pivot skip)
        floyd_warshall_v2(closure, t);
        return (1, t);
    }

    // Group nodes by SCC, find max SCC size
    let mut scc_sizes = vec![0usize; n_comp];
    for i in 0..t {
        scc_sizes[scc_labels[i] as usize] += 1;
    }
    let max_scc = *scc_sizes.iter().max().unwrap_or(&0);

    // Collect SCC node lists
    let mut scc_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for i in 0..t {
        scc_nodes[scc_labels[i] as usize].push(i);
    }

    // Floyd-Warshall (v2) within each non-trivial SCC
    for scc in &scc_nodes {
        if scc.len() <= 1 {
            continue;
        }
        let k = scc.len();
        let mut sub = vec![false; k * k];
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                sub[li * k + lj] = closure[ni * t + nj];
            }
            sub[li * k + li] = true;
        }
        floyd_warshall_v2(&mut sub, k);
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                closure[ni * t + nj] = sub[li * k + lj];
            }
        }
    }

    // Build condensed DAG
    let mut dag = vec![false; n_comp * n_comp];
    for i in 0..t {
        for j in 0..t {
            let si = scc_labels[i] as usize;
            let sj = scc_labels[j] as usize;
            if r_mat[i * t + j] && si != sj {
                dag[si * n_comp + sj] = true;
            }
        }
    }

    // Topological sort (Kahn's)
    let mut in_deg = vec![0usize; n_comp];
    for i in 0..n_comp {
        for j in 0..n_comp {
            if dag[i * n_comp + j] {
                in_deg[j] += 1;
            }
        }
    }
    let mut queue: Vec<usize> = (0..n_comp).filter(|&i| in_deg[i] == 0).collect();
    let mut topo = Vec::with_capacity(n_comp);
    let mut head = 0;
    while head < queue.len() {
        let nd = queue[head];
        head += 1;
        topo.push(nd);
        for j in 0..n_comp {
            if dag[nd * n_comp + j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 {
                    queue.push(j);
                }
            }
        }
    }

    // Bitset reachability propagation in reverse topological order.
    // Each component's reachability is stored as ceil(T/64) u64 words.
    let n_words = (t + 63) / 64;
    let mut reach: Vec<u64> = vec![0u64; n_comp * n_words];

    // Initialize: set bit j for each node j reachable from component c's closure
    for c in 0..n_comp {
        for &ni in &scc_nodes[c] {
            for j in 0..t {
                if closure[ni * t + j] {
                    reach[c * n_words + (j / 64)] |= 1u64 << (j % 64);
                }
            }
        }
    }

    // Propagate: merge successor reachability via bitwise OR
    for &c in topo.iter().rev() {
        for succ in 0..n_comp {
            if dag[c * n_comp + succ] {
                for w in 0..n_words {
                    reach[c * n_words + w] |= reach[succ * n_words + w];
                }
            }
        }
        // Write back to closure matrix
        for &ni in &scc_nodes[c] {
            for j in 0..t {
                if reach[c * n_words + (j / 64)] & (1u64 << (j % 64)) != 0 {
                    closure[ni * t + j] = true;
                }
            }
        }
    }

    (n_comp, max_scc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_reachability() {
        // 0->1->2
        let mut r = vec![false; 9];
        r[0 * 3 + 1] = true;
        r[1 * 3 + 2] = true;
        let mut closure = vec![false; 9];
        let mut labels = [0u32; 3];
        scc_transitive_closure(&r, 3, &mut closure, &mut labels);
        assert!(closure[0 * 3 + 2]); // 0 reaches 2
        assert!(!closure[2 * 3 + 0]); // 2 does NOT reach 0
    }

    #[test]
    fn test_v2_chain_matches_champion() {
        // 0->1->2: v2 must produce same closure as champion
        let mut r = vec![false; 9];
        r[0 * 3 + 1] = true;
        r[1 * 3 + 2] = true;
        let mut c1 = vec![false; 9];
        let mut c2 = vec![false; 9];
        let mut l1 = [0u32; 3];
        let mut l2 = [0u32; 3];
        scc_transitive_closure(&r, 3, &mut c1, &mut l1);
        scc_transitive_closure_v2(&r, 3, &mut c2, &mut l2);
        assert_eq!(c1, c2, "Closure mismatch on chain graph");
    }

    #[test]
    fn test_v2_cycle_matches_champion() {
        // 0->1->2->0: v2 must produce same closure as champion
        let mut r = vec![false; 9];
        r[0 * 3 + 1] = true;
        r[1 * 3 + 2] = true;
        r[2 * 3 + 0] = true;
        let mut c1 = vec![false; 9];
        let mut c2 = vec![false; 9];
        let mut l1 = [0u32; 3];
        let mut l2 = [0u32; 3];
        scc_transitive_closure(&r, 3, &mut c1, &mut l1);
        scc_transitive_closure_v2(&r, 3, &mut c2, &mut l2);
        assert_eq!(c1, c2, "Closure mismatch on cycle graph");
    }

    #[test]
    fn test_v2_complex_dag_matches_champion() {
        // 5-node graph: 0->1->2, 0->3->4, 2->4 (DAG with multiple paths)
        let t = 5;
        let mut r = vec![false; t * t];
        r[0 * t + 1] = true;
        r[1 * t + 2] = true;
        r[0 * t + 3] = true;
        r[3 * t + 4] = true;
        r[2 * t + 4] = true;
        let mut c1 = vec![false; t * t];
        let mut c2 = vec![false; t * t];
        let mut l1 = vec![0u32; t];
        let mut l2 = vec![0u32; t];
        scc_transitive_closure(&r, t, &mut c1, &mut l1);
        scc_transitive_closure_v2(&r, t, &mut c2, &mut l2);
        assert_eq!(c1, c2, "Closure mismatch on complex DAG");
    }

    #[test]
    fn test_v2_mixed_scc_dag_matches_champion() {
        // Mixed graph: SCC {0,1,2} + DAG 2->3->4
        let t = 5;
        let mut r = vec![false; t * t];
        // SCC: 0->1->2->0
        r[0 * t + 1] = true;
        r[1 * t + 2] = true;
        r[2 * t + 0] = true;
        // DAG tail: 2->3->4
        r[2 * t + 3] = true;
        r[3 * t + 4] = true;
        let mut c1 = vec![false; t * t];
        let mut c2 = vec![false; t * t];
        let mut l1 = vec![0u32; t];
        let mut l2 = vec![0u32; t];
        scc_transitive_closure(&r, t, &mut c1, &mut l1);
        scc_transitive_closure_v2(&r, t, &mut c2, &mut l2);
        assert_eq!(c1, c2, "Closure mismatch on mixed SCC+DAG graph");
    }

    #[test]
    fn test_cycle_all_reach_all() {
        // 0->1->2->0
        let mut r = vec![false; 9];
        r[0 * 3 + 1] = true;
        r[1 * 3 + 2] = true;
        r[2 * 3 + 0] = true;
        let mut closure = vec![false; 9];
        let mut labels = [0u32; 3];
        scc_transitive_closure(&r, 3, &mut closure, &mut labels);
        // All reach all
        for i in 0..3 {
            for j in 0..3 {
                assert!(closure[i * 3 + j], "expected {i}->{j} reachable");
            }
        }
    }
}
