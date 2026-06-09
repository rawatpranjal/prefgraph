use highs::{HighsModelStatus, RowProblem, Sense};
use crate::closure::scc_transitive_closure_v2 as scc_transitive_closure;
use crate::graph::PreferenceGraph;
use crate::types::GarpResult;

/// Check Menu SARP (Strong Axiom for abstract choice).
///
/// Operates on item-space graph (I items as nodes). R[x,y] = True if x was
/// chosen from a menu containing y. SARP violated if R*[x,y] AND R*[y,x]
/// for any x != y (cycle in transitive preferences).
///
/// Requires: graph parsed with parse_menu() (R already built on items).
pub fn menu_sarp_check(graph: &mut PreferenceGraph) -> GarpResult {
    graph.ensure_closure();
    let t = graph.t; // t = n_items for menu data

    // SARP violation: R*[x,y] AND R*[y,x] for x != y
    let mut n_violations = 0u32;
    for x in 0..t {
        for y in (x + 1)..t {
            if graph.r_star[x * t + y] && graph.r_star[y * t + x] {
                n_violations += 1;
            }
        }
    }

    GarpResult {
        is_consistent: n_violations == 0,
        n_violations,
        max_scc_size: graph.max_scc_size as u32,
        n_components: graph.n_components as u32,
    }
}

/// Check Menu WARP (Weak Axiom for abstract choice).
///
/// Simpler than SARP: only checks for direct reversals.
/// Violation if R[x,y] AND R[y,x] for any x != y.
/// No transitive closure needed - O(I²).
pub fn menu_warp_check(graph: &PreferenceGraph) -> GarpResult {
    let t = graph.t;
    let mut n_violations = 0u32;

    for x in 0..t {
        for y in (x + 1)..t {
            if graph.r[x * t + y] && graph.r[y * t + x] {
                n_violations += 1;
            }
        }
    }

    GarpResult {
        is_consistent: n_violations == 0,
        n_violations,
        max_scc_size: 0,
        n_components: 0,
    }
}

/// Threshold: use exact ILP for n_obs <= this, greedy above.
/// Mirrors Python MENU_HM_ILP_THRESHOLD = 60.
const MENU_HM_ILP_THRESHOLD: usize = 60;

/// Menu Houtman-Maks over OBSERVATIONS (Houtman & Maks 1985; Demuynck & Rehbeck
/// 2023 Def 3).
///
/// Returns the largest subset of observations (menu-choice pairs) whose induced
/// item preference graph is acyclic (SARP-consistent).
/// Returns (n_observations_consistent, n_observations_total).
///
/// Exact via ranking ILP (HiGHS MILP) for n_obs <= MENU_HM_ILP_THRESHOLD;
/// greedy FVS over observations above that threshold.
///
/// ILP formulation (mirrors Python _menu_houtman_maks_exact):
///   Variables: z_o ∈ {0,1} (keep-flag per obs), r_i ∈ [0, n_items] (rank per item).
///   For each obs o and edge (chosen, other): big_m*z_o - r[chosen] + r[other] <= big_m - 1.
///   Objective: maximize sum(z_o).
///   A consistent ranking exists iff the kept item graph is acyclic (SARP holds).
pub fn menu_houtman_maks(
    menus: &[Vec<usize>],
    choices: &[usize],
    n_items: usize,
) -> (usize, usize) {
    let n_obs = choices.len();

    if n_obs <= 1 || n_items == 0 {
        return (n_obs, n_obs);
    }

    // Build per-observation edges: (chosen, other) for each menu-choice pair.
    let obs_edges: Vec<Vec<(usize, usize)>> = menus
        .iter()
        .zip(choices.iter())
        .map(|(menu, &choice)| {
            menu.iter()
                .filter(|&&item| item != choice)
                .map(|&item| (choice, item))
                .collect()
        })
        .collect();

    // Build collapsed item preference graph and check for SARP violations.
    let mut r = vec![false; n_items * n_items];
    for edges in &obs_edges {
        for &(chosen, other) in edges {
            if chosen < n_items && other < n_items {
                r[chosen * n_items + other] = true;
            }
        }
    }
    let mut r_star = vec![false; n_items * n_items];
    let mut scc_labels = vec![0u32; n_items];
    scc_transitive_closure(&r, n_items, &mut r_star, &mut scc_labels);

    let has_violation = (0..n_items).any(|x| {
        (x + 1..n_items).any(|y| r_star[x * n_items + y] && r_star[y * n_items + x])
    });

    if !has_violation {
        return (n_obs, n_obs);
    }

    let removed = if n_obs <= MENU_HM_ILP_THRESHOLD {
        menu_hm_exact_ilp(&obs_edges, n_items, n_obs)
    } else {
        menu_hm_greedy(&obs_edges, n_items, n_obs)
    };

    (n_obs - removed.len(), n_obs)
}

/// Exact menu Houtman-Maks over observations via ranking ILP (HiGHS MILP).
///
/// Maximises the number of kept observations subject to the kept item graph
/// being acyclic, encoded as a ranking ILP:
///   - z_o ∈ {0,1}: binary keep-flag per observation
///   - r_i ∈ [0, n_items]: real rank per item
///   - For every edge (chosen, other) of observation o:
///       big_m * z_o - r[chosen] + r[other] <= big_m - 1
///     (active when z_o=1: forces r[chosen] >= r[other]+1; inactive when z_o=0)
///   - Objective: maximize sum(z_o)
///
/// Falls back to greedy if HiGHS fails.
fn menu_hm_exact_ilp(
    obs_edges: &[Vec<(usize, usize)>],
    n_items: usize,
    n_obs: usize,
) -> Vec<usize> {
    let big_m = (n_items + 1) as f64;

    let mut pb = RowProblem::default();

    // z_o columns: binary {0,1}, cost -1 (minimise -sum(z) = maximise sum(z))
    let mut z_cols = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        z_cols.push(pb.add_integer_column(-1.0, 0.0..1.0));
    }

    // r_i columns: continuous [0, n_items], cost 0
    let mut r_cols = Vec::with_capacity(n_items);
    for _ in 0..n_items {
        r_cols.push(pb.add_column(0.0, 0.0..(n_items as f64)));
    }

    // Ranking constraints.
    // For each observation o and edge (chosen, other):
    //   big_m * z_o - r[chosen] + r[other] <= big_m - 1
    //
    // When z_o=1: r[chosen] - r[other] >= 1 (chosen ranks strictly above other).
    // When z_o=0: constraint is -r[chosen]+r[other] <= big_m-1 (always satisfied).
    let mut any_row = false;
    for (o, edges) in obs_edges.iter().enumerate() {
        for &(chosen, other) in edges {
            if chosen < n_items && other < n_items {
                pb.add_row(
                    ..(big_m - 1.0),
                    [
                        (z_cols[o], big_m),
                        (r_cols[chosen], -1.0),
                        (r_cols[other], 1.0),
                    ],
                );
                any_row = true;
            }
        }
    }

    if !any_row {
        return vec![];
    }

    let mut model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            (0..n_obs)
                .filter(|&o| sol.columns()[o] < 0.5)
                .collect()
        }
        _ => menu_hm_greedy(obs_edges, n_items, n_obs),
    }
}

/// Greedy upper bound: iteratively remove the observation that touches the most
/// SARP-violating item pairs, until the remaining item graph is acyclic.
/// Used above MENU_HM_ILP_THRESHOLD and as ILP fallback.
/// Over-removes relative to the optimum (not exact).
fn menu_hm_greedy(
    obs_edges: &[Vec<(usize, usize)>],
    n_items: usize,
    n_obs: usize,
) -> Vec<usize> {
    let mut kept = vec![true; n_obs];
    let mut removed = Vec::new();

    loop {
        // Build item graph from kept observations.
        let mut r = vec![false; n_items * n_items];
        for (o, edges) in obs_edges.iter().enumerate() {
            if kept[o] {
                for &(chosen, other) in edges {
                    if chosen < n_items && other < n_items {
                        r[chosen * n_items + other] = true;
                    }
                }
            }
        }

        // Compute transitive closure.
        let mut r_star = vec![false; n_items * n_items];
        let mut scc_labels = vec![0u32; n_items];
        scc_transitive_closure(&r, n_items, &mut r_star, &mut scc_labels);

        // Find violation pairs (both R*[x,y] and R*[y,x]).
        let mut viol = vec![false; n_items * n_items];
        let mut any_viol = false;
        for x in 0..n_items {
            for y in 0..n_items {
                if x != y && r_star[x * n_items + y] && r_star[y * n_items + x] {
                    viol[x * n_items + y] = true;
                    any_viol = true;
                }
            }
        }

        if !any_viol {
            break;
        }

        // Score each kept observation by how many violating pairs it touches.
        let mut best_o = usize::MAX;
        let mut best_score = 0usize;
        for (o, edges) in obs_edges.iter().enumerate() {
            if !kept[o] {
                continue;
            }
            let score = edges
                .iter()
                .filter(|&&(c, k)| {
                    (c < n_items && k < n_items)
                        && (viol[c * n_items + k] || viol[k * n_items + c])
                })
                .count();
            if score > best_score {
                best_score = score;
                best_o = o;
            }
        }

        if best_o == usize::MAX {
            break;
        }

        kept[best_o] = false;
        removed.push(best_o);
    }

    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_menu_sarp_consistent() {
        // 0 > 1 > 2 (consistent linear order)
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1, 2], vec![1, 2]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 3);
        let result = menu_sarp_check(&mut graph);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_menu_sarp_violation() {
        // 0 > 1, 1 > 0 (direct cycle)
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1], vec![0, 1]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 2);
        let result = menu_sarp_check(&mut graph);
        assert!(!result.is_consistent);
    }

    #[test]
    fn test_menu_warp_consistent() {
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1, 2], vec![1, 2]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 3);
        let result = menu_warp_check(&graph);
        assert!(result.is_consistent);
    }

    #[test]
    fn test_menu_warp_violation() {
        let mut graph = PreferenceGraph::new(3);
        let menus = vec![vec![0, 1], vec![0, 1]];
        let choices = vec![0, 1];
        graph.parse_menu(&menus, &choices, 2);
        let result = menu_warp_check(&graph);
        assert!(!result.is_consistent);
    }

    /// Consistent data: all observations kept.
    #[test]
    fn test_menu_hm_consistent() {
        let menus = vec![vec![0, 1, 2], vec![1, 2]];
        let choices = vec![0, 1];
        let (kept, total) = menu_houtman_maks(&menus, &choices, 3);
        assert_eq!(kept, 2);
        assert_eq!(total, 2);
    }

    /// 4-cycle: menus=[[0,1],[1,2],[2,3],[3,0]], choices=[0,1,2,3].
    /// Canonical test: kept=3, total=4 (one obs removed to break cycle).
    #[test]
    fn test_menu_hm_4cycle() {
        let menus = vec![
            vec![0, 1],
            vec![1, 2],
            vec![2, 3],
            vec![3, 0],
        ];
        let choices = vec![0, 1, 2, 3];
        let (kept, total) = menu_houtman_maks(&menus, &choices, 4);
        assert_eq!(total, 4, "total must be n_obs=4");
        assert_eq!(kept, 3, "4-cycle: remove exactly 1 observation");
    }

    /// Direct 2-cycle: obs 0 chooses 0 over 1, obs 1 chooses 1 over 0.
    /// Remove 1 of 2 observations.
    #[test]
    fn test_menu_hm_2cycle() {
        let menus = vec![vec![0, 1], vec![0, 1]];
        let choices = vec![0, 1];
        let (kept, total) = menu_houtman_maks(&menus, &choices, 2);
        assert_eq!(total, 2);
        assert_eq!(kept, 1);
    }

    /// 3-cycle over items 0,1,2: remove 1 of 3 observations.
    #[test]
    fn test_menu_hm_3cycle() {
        let menus = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let choices = vec![0, 1, 2];
        let (kept, total) = menu_houtman_maks(&menus, &choices, 3);
        assert_eq!(total, 3);
        assert_eq!(kept, 2);
    }
}
