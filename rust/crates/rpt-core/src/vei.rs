use highs::{HighsModelStatus, RowProblem, Sense};

use crate::graph::PreferenceGraph;

/// Result of VEI (Varian's Efficiency Index) computation.
#[derive(Debug, Clone)]
pub struct VeiResult {
    pub success: bool,
    pub efficiency_vector: Vec<f64>,
    pub mean_efficiency: f64,
    pub min_efficiency: f64,
    pub worst_observation: usize,
    pub total_inefficiency: f64,
}

/// Compute per-observation efficiency scores (Varian's Efficiency Index).
///
/// Finds e_i ∈ [0,1] for each observation that maximizes total efficiency
/// subject to preference constraints from the transitive closure.
///
/// LP relaxation: max Σe_i subject to e_i ≥ E[i,j]/own_exp[i] for all (i,j) where R*[i,j].
///
/// Note: This is a polynomial LP relaxation of the true VEI (which is NP-hard).
/// It constrains efficiency via the existing R* structure but does not account for
/// how lowering e_i changes which preferences are revealed.
/// For exact VEI, use `compute_vei_exact()`.
///
/// Requires: graph has R* (closure) and E computed.
pub fn compute_vei(graph: &mut PreferenceGraph) -> VeiResult {
    graph.ensure_closure();
    let t = graph.t;

    if t == 0 {
        return VeiResult {
            success: true,
            efficiency_vector: vec![],
            mean_efficiency: 1.0,
            min_efficiency: 1.0,
            worst_observation: 0,
            total_inefficiency: 0.0,
        };
    }

    // Check if already consistent (no violations → all e_i = 1)
    let mut has_violation = false;
    for i in 0..t {
        for j in 0..t {
            if graph.r_star[i * t + j] && graph.p[j * t + i] {
                has_violation = true;
                break;
            }
        }
        if has_violation {
            break;
        }
    }

    if !has_violation {
        return VeiResult {
            success: true,
            efficiency_vector: vec![1.0; t],
            mean_efficiency: 1.0,
            min_efficiency: 1.0,
            worst_observation: 0,
            total_inefficiency: 0.0,
        };
    }

    // Build LP: maximize Σe_i = minimize -Σe_i
    // Variables: e_0..e_{T-1} ∈ [0, 1]
    // Constraints: for each (i,j) where R*[i,j] and i≠j:
    //   e_i ≥ E[i,j] / own_exp[i]   (rewritten as -e_i ≤ -ratio)
    let mut pb = RowProblem::default();

    let mut cols = Vec::with_capacity(t);
    for _i in 0..t {
        // cost = -1 (minimize -e_i = maximize e_i), bounds [0, 1]
        cols.push(pb.add_column(-1.0, 0.0..1.0));
    }

    // Add constraints from transitive closure
    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            if i == j {
                continue;
            }
            if graph.r_star[i * t + j] {
                let ratio = graph.e[i * t + j] / graph.own_exp[i];
                if ratio > 0.0 && ratio <= 1.0 {
                    // e_i ≥ ratio (lower bound on efficiency for this preference)
                    pb.add_row(ratio.., [(cols[i], 1.0)]);
                }
            }
        }
    }

    let model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            let e_vec: Vec<f64> = (0..t)
                .map(|i| sol.columns()[i].clamp(0.0, 1.0))
                .collect();

            let mean = e_vec.iter().sum::<f64>() / t as f64;
            let min_e = e_vec.iter().cloned().fold(f64::INFINITY, f64::min);
            let worst = e_vec
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let total_ineff = e_vec.iter().map(|&e| 1.0 - e).sum();

            VeiResult {
                success: true,
                efficiency_vector: e_vec,
                mean_efficiency: mean,
                min_efficiency: min_e,
                worst_observation: worst,
                total_inefficiency: total_ineff,
            }
        }
        _ => VeiResult {
            success: false,
            efficiency_vector: vec![0.0; t],
            mean_efficiency: 0.0,
            min_efficiency: 0.0,
            worst_observation: 0,
            total_inefficiency: t as f64,
        },
    }
}

// ---------------------------------------------------------------------------
// Exact VEI: Mononen (2023) Theorem 1 with the canonical-vector convention
// ---------------------------------------------------------------------------

/// Tolerance below which an arc removal cost is treated as zero. Mononen
/// (2023, p. 10): "since the cost of removing revealed preferences that are
/// not strict is zero, we can focus on the strict revealed preference P and
/// strict cycles." Arcs this cheap are removable at infimum cost zero and
/// never bind, so they are dropped from the arc set. The same tolerance is
/// the AddCost survival test: an arc survives adjustment d_i iff
/// cost - d_i > VEI_COST_TOL, and the U-set inclusion test mirrors it.
const VEI_COST_TOL: f64 = 1e-12;

/// Slack on caps carried between the sequential canonical-vector solves so
/// the previous incumbent stays feasible under float summation noise. Grid
/// gaps on real data are orders of magnitude larger, so the slack never
/// admits a different grid point.
const VEI_CAP_EPS: f64 = 1e-9;

/// Hard cap on MILP solves across all stages. Hitting it returns
/// success = false, never a silently truncated answer (the pre-2026 code
/// capped row generation at 50 and silently returned the incumbent).
const VEI_MAX_SOLVES: usize = 1000;

/// Exact Varian index via Mononen (2023) Theorem 1, plus a deterministic
/// canonical efficiency vector.
///
/// Varian's index (Mononen 2023, p. 9) relaxes the revealed preference with
/// observation-specific adjustments d_t in [0,1] (the efficiency level is
/// e_t = 1 - d_t) and is the least average adjustment that makes the relaxed
/// strict preference acyclic:
///
/// ```text
/// I_Var(D) = (1/T) inf sum_t d_t  s.t.  the relaxed relation is acyclic.
/// ```
///
/// Theorem 1 (p. 11) reformulates this as a binary linear program with one
/// variable theta per strict arc, objective min sum theta_ij * cost_ij with
/// cost_ij = (p_i.x_i - p_i.x_j)/(p_i.x_i), and for every strict cycle the
/// U-SET EXPANDED covering constraint: each cycle arc (a,b) may be covered by
/// ANY arc out of a that costs at least cost_ab, because "this adjustment e_t
/// removes the revealed preference x_t R x_t' and all the 'cheaper' revealed
/// preferences" (p. 10). Charging each removed arc independently without the
/// U-set expansion overcounts nested removals at one observation; that was
/// the pre-2026 bug here.
///
/// At most one arc per observation is selected at an optimum ("In the optimal
/// solution, there is at most one removal of this type", p. 11), enforced
/// explicitly with per-observation SOS rows so that the observation spend
/// d_i = sum_j theta_ij cost_ij is linear in every feasible point.
///
/// Row generation follows Algorithm 2 (p. 14): seed with all 2-cycles, then
/// repeatedly solve the subproblem and search for surviving critical cycles
/// (Algorithm 1, p. 13) until none remain. The separation oracle keeps an arc
/// alive iff its AddCost = cost_ij - d_i is positive and greedily breaks each
/// found cycle at its minimum-AddCost arc before continuing the search.
///
/// The optimal adjustment VECTOR is not unique under ties (Mononen only
/// defines the value). PrefGraph reports the canonical vector: among
/// value-optimal solutions, stage B minimizes the maximum adjustment
/// (equivalently maximizes the minimum efficiency), then stage C minimizes
/// each d_i in observation order (lexicographic, so earlier observations keep
/// the benefit of the doubt). Every stage re-runs separation so its incumbent
/// is verified acyclic, and all reported numbers are derived from the binary
/// incumbent (selected arc ratios), never from raw solver column values, so
/// both backends agree exactly on discrete data.
pub fn compute_vei_exact(graph: &mut PreferenceGraph) -> VeiResult {
    use crate::garp::garp_check;

    let t = graph.t;

    if t == 0 {
        return VeiResult {
            success: true,
            efficiency_vector: vec![],
            mean_efficiency: 1.0,
            min_efficiency: 1.0,
            worst_observation: 0,
            total_inefficiency: 0.0,
        };
    }

    // O(T²) GARP check
    let garp = garp_check(graph);
    if garp.is_consistent {
        return vei_all_efficient(t);
    }

    // Collect strict preference arcs with costs.
    // Arc (i,j): obs i strictly revealed preferred over j (P[i,j] = true).
    // Cost: (own_exp[i] - E[i,j]) / own_exp[i] = 1 - ratio.
    let mut arc_from = Vec::new();
    let mut arc_to = Vec::new();
    let mut arc_cost = Vec::new();
    let mut arc_ratio = Vec::new();

    for i in 0..t {
        if graph.own_exp[i] <= 0.0 {
            continue;
        }
        for j in 0..t {
            if i != j && graph.p[i * t + j] {
                let ratio = graph.e[i * t + j] / graph.own_exp[i];
                let cost = 1.0 - ratio;
                if cost > VEI_COST_TOL {
                    arc_from.push(i);
                    arc_to.push(j);
                    arc_cost.push(cost);
                    arc_ratio.push(ratio);
                }
            }
        }
    }

    let n_arcs = arc_from.len();
    if n_arcs == 0 {
        return vei_all_efficient(t);
    }

    // Adjacency list for the DFS oracle: adj[i] = [(to, arc_idx), ...]
    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; t];
    for idx in 0..n_arcs {
        adj[arc_from[idx]].push((arc_to[idx], idx));
    }

    // Per-observation arc lists sorted by cost DESCENDING (ties by arc index,
    // stable) so a U-set is always a prefix. The Python mirror must build the
    // identical order.
    let mut arcs_of_obs: Vec<Vec<usize>> = vec![vec![]; t];
    for idx in 0..n_arcs {
        arcs_of_obs[arc_from[idx]].push(idx);
    }
    for list in arcs_of_obs.iter_mut() {
        list.sort_by(|&a, &b| {
            arc_cost[b]
                .partial_cmp(&arc_cost[a])
                .unwrap()
                .then(a.cmp(&b))
        });
    }

    // Seed cycles: all 2-cycles (Algorithm 2 first step); if none exist,
    // search for longer cycles from the zero solution (Algorithm 2 fallback).
    let mut arc_lookup = vec![usize::MAX; t * t];
    for idx in 0..n_arcs {
        arc_lookup[arc_from[idx] * t + arc_to[idx]] = idx;
    }
    let mut seed_cycles: Vec<Vec<usize>> = Vec::new();
    for idx in 0..n_arcs {
        let (i, j) = (arc_from[idx], arc_to[idx]);
        if i < j {
            let rev = arc_lookup[j * t + i];
            if rev != usize::MAX {
                seed_cycles.push(vec![idx, rev]);
            }
        }
    }
    if seed_cycles.is_empty() {
        seed_cycles = vei_separation(&adj, &arc_from, &arc_cost, &vec![0.0; t], n_arcs, t);
        if seed_cycles.is_empty() {
            // The strict-arc graph is acyclic: every GARP violation runs
            // through zero-cost (non-strict or sub-tolerance) arcs, so the
            // infimum adjustment is zero everywhere.
            return vei_all_efficient(t);
        }
    }

    // U-expanded covering rows, deduplicated with deterministic ordering.
    let mut rows: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();
    for cyc in &seed_cycles {
        rows.insert(vei_u_expand(cyc, &arc_from, &arc_cost, &arcs_of_obs));
    }

    let mut solves = 0usize;

    // ── Stage A: the Theorem 1 value ──────────────────────────────────────
    let mut incumbent: Vec<bool>;
    loop {
        solves += 1;
        if solves > VEI_MAX_SOLVES {
            return vei_failure(t);
        }
        incumbent = match vei_solve_stage(
            &arc_cost,
            &arc_from,
            &arcs_of_obs,
            &rows,
            None,
            false,
            None,
            &[],
        ) {
            Some(th) => th,
            None => return vei_failure(t),
        };
        let d = vei_d_from_theta(&incumbent, &arc_from, &arc_cost, t);
        let new_cycles = vei_separation(&adj, &arc_from, &arc_cost, &d, n_arcs, t);
        if new_cycles.is_empty() {
            break;
        }
        for cyc in &new_cycles {
            rows.insert(vei_u_expand(cyc, &arc_from, &arc_cost, &arcs_of_obs));
        }
    }
    let d_star = vei_d_from_theta(&incumbent, &arc_from, &arc_cost, t);
    let v_star: f64 = d_star.iter().sum();
    // Float-summation slack only: grid totals differ by far more than this,
    // so the budget admits exactly the value-optimal solutions.
    let budget = v_star + 1e-9 * v_star.max(1.0);

    // ── Stage B: among optima, minimize the maximum adjustment ────────────
    loop {
        solves += 1;
        if solves > VEI_MAX_SOLVES {
            return vei_failure(t);
        }
        let th = match vei_solve_stage(
            &arc_cost,
            &arc_from,
            &arcs_of_obs,
            &rows,
            Some(budget),
            true,
            None,
            &[],
        ) {
            Some(th) => th,
            None => return vei_failure(t),
        };
        let d = vei_d_from_theta(&th, &arc_from, &arc_cost, t);
        let new_cycles = vei_separation(&adj, &arc_from, &arc_cost, &d, n_arcs, t);
        if new_cycles.is_empty() {
            incumbent = th;
            break;
        }
        for cyc in &new_cycles {
            rows.insert(vei_u_expand(cyc, &arc_from, &arc_cost, &arcs_of_obs));
        }
    }
    let mut d_inc = vei_d_from_theta(&incumbent, &arc_from, &arc_cost, t);
    let m_b = d_inc.iter().cloned().fold(0.0, f64::max);
    let mut caps: Vec<(usize, f64)> = (0..t).map(|i| (i, m_b + VEI_CAP_EPS)).collect();

    // ── Stage C: lexicographic minimization in observation order ──────────
    // Once d_i is minimized to g_i and capped, every later feasible solution
    // has d_i in [g_i, g_i + eps]: later programs are subsets, so their
    // minima cannot drop below g_i, and the only grid point in that interval
    // is g_i itself. The vector is therefore pinned independent of solver
    // vertex choice. If the incumbent already attains d_i = 0 (the global
    // lower bound) the solve is skipped.
    for i in 0..t {
        if d_inc[i] <= VEI_CAP_EPS {
            caps.push((i, VEI_CAP_EPS));
            continue;
        }
        loop {
            solves += 1;
            if solves > VEI_MAX_SOLVES {
                return vei_failure(t);
            }
            let th = match vei_solve_stage(
                &arc_cost,
                &arc_from,
                &arcs_of_obs,
                &rows,
                Some(budget),
                false,
                Some(i),
                &caps,
            ) {
                Some(th) => th,
                None => return vei_failure(t),
            };
            let d = vei_d_from_theta(&th, &arc_from, &arc_cost, t);
            let new_cycles = vei_separation(&adj, &arc_from, &arc_cost, &d, n_arcs, t);
            if new_cycles.is_empty() {
                incumbent = th;
                d_inc = d;
                break;
            }
            for cyc in &new_cycles {
                rows.insert(vei_u_expand(cyc, &arc_from, &arc_cost, &arcs_of_obs));
            }
        }
        caps.push((i, d_inc[i] + VEI_CAP_EPS));
    }

    // Derive the efficiency vector from the verified incumbent: e_i is the
    // ratio of the selected arc at i (exactly the grid float, not 1 - cost),
    // or 1.0 when no arc is selected. SOS rows guarantee at most one
    // selection per observation; equal-cost arcs share the same ratio, so
    // the vector does not depend on which of them the solver picked.
    let mut efficiency = vec![1.0f64; t];
    for idx in 0..n_arcs {
        if incumbent[idx] && arc_ratio[idx] < efficiency[arc_from[idx]] {
            efficiency[arc_from[idx]] = arc_ratio[idx];
        }
    }

    let mean = efficiency.iter().sum::<f64>() / t as f64;
    let min_e = efficiency.iter().cloned().fold(f64::INFINITY, f64::min);
    // First index attaining the minimum (numpy argmin convention).
    let mut worst = 0usize;
    for (i, &e) in efficiency.iter().enumerate() {
        if e < efficiency[worst] {
            worst = i;
        }
    }
    let total_ineff = efficiency.iter().map(|&e| 1.0 - e).sum();

    VeiResult {
        success: true,
        efficiency_vector: efficiency,
        mean_efficiency: mean,
        min_efficiency: min_e,
        worst_observation: worst,
        total_inefficiency: total_ineff,
    }
}

fn vei_all_efficient(t: usize) -> VeiResult {
    VeiResult {
        success: true,
        efficiency_vector: vec![1.0; t],
        mean_efficiency: 1.0,
        min_efficiency: 1.0,
        worst_observation: 0,
        total_inefficiency: 0.0,
    }
}

/// Loud failure: solver error or solve-cap hit. Never a silent truncation.
fn vei_failure(t: usize) -> VeiResult {
    VeiResult {
        success: false,
        efficiency_vector: vec![0.0; t],
        mean_efficiency: 0.0,
        min_efficiency: 0.0,
        worst_observation: 0,
        total_inefficiency: t as f64,
    }
}

/// Observation spend implied by a binary selection: d_i = sum of selected
/// arc costs at i. Under the SOS rows at most one arc per observation is
/// selected, so d_i is exactly one cost grid float (or 0).
fn vei_d_from_theta(theta: &[bool], arc_from: &[usize], arc_cost: &[f64], t: usize) -> Vec<f64> {
    let mut d = vec![0.0f64; t];
    for (idx, &sel) in theta.iter().enumerate() {
        if sel {
            d[arc_from[idx]] += arc_cost[idx];
        }
    }
    d
}

/// U-set expansion of a cycle into a covering row (Mononen 2023, p. 10):
/// U(x_t, x_t*) = {(x_t, x_t') | p_t.(x_t - x_t*) <= p_t.(x_t - x_t')}, i.e.
/// every arc out of t costing at least the cycle arc. Selecting any of them
/// removes the cycle arc as one of its "cheaper" preferences. Rows are
/// returned as sorted deduplicated column lists so identical rows compare
/// equal in the BTreeSet.
fn vei_u_expand(
    cycle: &[usize],
    arc_from: &[usize],
    arc_cost: &[f64],
    arcs_of_obs: &[Vec<usize>],
) -> Vec<usize> {
    let mut cols = std::collections::BTreeSet::new();
    for &a in cycle {
        let f = arc_from[a];
        let ca = arc_cost[a];
        for &idx in &arcs_of_obs[f] {
            // Sorted descending: the U-set is the prefix with cost >= ca.
            // The closed inequality carries the same tolerance as survival:
            // selecting an arc with cost >= ca - tol forces d_f >= ca - tol,
            // which removes the cycle arc under the survival test.
            if arc_cost[idx] >= ca - VEI_COST_TOL {
                cols.insert(idx);
            } else {
                break;
            }
        }
    }
    cols.into_iter().collect()
}

/// One MILP subproblem over the current row set.
///
/// Columns: binary theta per arc, plus a continuous M when minimizing the
/// maximum adjustment (stage B). Rows: U-expanded covering rows (>= 1),
/// per-observation SOS rows (<= 1), an optional value budget
/// (sum cost*theta <= budget), per-observation max rows (d_i - M <= 0) in
/// stage B, and accumulated per-observation caps (d_i <= cap) in stage C.
/// Objectives: stage A minimizes sum cost*theta, stage B minimizes M,
/// stage C minimizes d_i for the given observation.
#[allow(clippy::too_many_arguments)]
fn vei_solve_stage(
    arc_cost: &[f64],
    arc_from: &[usize],
    arcs_of_obs: &[Vec<usize>],
    rows: &std::collections::BTreeSet<Vec<usize>>,
    budget: Option<f64>,
    minimize_max: bool,
    objective_obs: Option<usize>,
    caps: &[(usize, f64)],
) -> Option<Vec<bool>> {
    let n_arcs = arc_cost.len();
    let mut pb = RowProblem::default();

    let mut cols = Vec::with_capacity(n_arcs);
    for idx in 0..n_arcs {
        let obj = if minimize_max {
            0.0
        } else if let Some(obs) = objective_obs {
            if arc_from[idx] == obs {
                arc_cost[idx]
            } else {
                0.0
            }
        } else {
            arc_cost[idx]
        };
        cols.push(pb.add_integer_column(obj, 0.0..1.0));
    }
    let m_col = if minimize_max {
        Some(pb.add_column(1.0, 0.0..1.0))
    } else {
        None
    };

    // Covering rows: at least one selection per U-expanded cycle row.
    for row in rows {
        let terms: Vec<_> = row.iter().map(|&idx| (cols[idx], 1.0)).collect();
        pb.add_row(1.0.., terms);
    }

    // SOS rows: at most one selected arc per observation (p. 11 dominance).
    for obs_arcs in arcs_of_obs {
        if obs_arcs.len() > 1 {
            let terms: Vec<_> = obs_arcs.iter().map(|&idx| (cols[idx], 1.0)).collect();
            pb.add_row(..1.0, terms);
        }
    }

    // Value budget: restrict to (float-slack) value-optimal solutions.
    if let Some(b) = budget {
        let terms: Vec<_> = (0..n_arcs).map(|idx| (cols[idx], arc_cost[idx])).collect();
        pb.add_row(..b, terms);
    }

    // Stage B max rows: d_i <= M for every observation with arcs.
    if let Some(m) = m_col {
        for obs_arcs in arcs_of_obs {
            if obs_arcs.is_empty() {
                continue;
            }
            let mut terms: Vec<_> = obs_arcs
                .iter()
                .map(|&idx| (cols[idx], arc_cost[idx]))
                .collect();
            terms.push((m, -1.0));
            pb.add_row(..0.0, terms);
        }
    }

    // Stage C caps: d_i <= cap for already-pinned observations.
    for &(obs, cap) in caps {
        if arcs_of_obs[obs].is_empty() {
            continue;
        }
        let terms: Vec<_> = arcs_of_obs[obs]
            .iter()
            .map(|&idx| (cols[idx], arc_cost[idx]))
            .collect();
        pb.add_row(..cap, terms);
    }

    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    // Exactness: the default HiGHS relative MIP gap (1e-4) could accept a
    // suboptimal incumbent between adjacent grid totals on near-tie data.
    model.set_option("mip_rel_gap", 0.0);
    model.set_option("mip_abs_gap", 0.0);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            Some((0..n_arcs).map(|j| sol.columns()[j] > 0.5).collect())
        }
        _ => None,
    }
}

/// Separation oracle (Mononen 2023, Algorithm 1): find strict cycles that
/// survive the adjustment d. An arc is alive iff its AddCost
/// = cost_ij - d_i is positive ("implement the current solution ... and
/// remove all the revealed preferences with an additional cost lower than 0",
/// p. 12, with the selected arc itself at AddCost exactly 0). Each found
/// cycle is greedily broken at its minimum-AddCost arc and the search
/// continues, returning a diverse set of critical cycles per call. Empty
/// return certifies the current solution removes every strict cycle, which
/// is the Algorithm 2 termination condition.
fn vei_separation(
    adj: &[Vec<(usize, usize)>],
    arc_from: &[usize],
    arc_cost: &[f64],
    d: &[f64],
    n_arcs: usize,
    t: usize,
) -> Vec<Vec<usize>> {
    let mut alive: Vec<bool> = (0..n_arcs)
        .map(|idx| arc_cost[idx] - d[arc_from[idx]] > VEI_COST_TOL)
        .collect();
    let mut cycles = Vec::new();
    while let Some(cycle) = vei_find_cycle(adj, &alive, t) {
        // Greedy break at minimum AddCost (ties by arc index, deterministic).
        let break_arc = cycle
            .iter()
            .copied()
            .min_by(|&a, &b| {
                let ca = arc_cost[a] - d[arc_from[a]];
                let cb = arc_cost[b] - d[arc_from[b]];
                ca.partial_cmp(&cb).unwrap().then(a.cmp(&b))
            })
            .expect("cycle is non-empty");
        alive[break_arc] = false;
        cycles.push(cycle);
    }
    cycles
}

/// First cycle among alive arcs, as a list of arc indices, or None if the
/// alive subgraph is acyclic. Standard colored DFS; gray nodes are exactly
/// the current path, so a back edge closes a cycle.
fn vei_find_cycle(adj: &[Vec<(usize, usize)>], alive: &[bool], t: usize) -> Option<Vec<usize>> {
    let mut color = vec![0u8; t]; // 0=white, 1=gray, 2=black
    let mut path_arcs: Vec<usize> = Vec::new();
    let mut path_nodes: Vec<usize> = Vec::new();
    for start in 0..t {
        if color[start] == 0 {
            if let Some(cycle) = vei_cycle_visit(
                start,
                adj,
                alive,
                &mut color,
                &mut path_arcs,
                &mut path_nodes,
            ) {
                return Some(cycle);
            }
        }
    }
    None
}

fn vei_cycle_visit(
    node: usize,
    adj: &[Vec<(usize, usize)>],
    alive: &[bool],
    color: &mut [u8],
    path_arcs: &mut Vec<usize>,
    path_nodes: &mut Vec<usize>,
) -> Option<Vec<usize>> {
    color[node] = 1; // gray
    path_nodes.push(node);

    for &(next, arc_idx) in &adj[node] {
        if !alive[arc_idx] {
            continue;
        }
        if color[next] == 1 {
            // Back edge: next is on the current path, close the cycle.
            let pos = path_nodes
                .iter()
                .rposition(|&n| n == next)
                .expect("gray node is on the current path");
            let mut cycle: Vec<usize> = path_arcs[pos..].to_vec();
            cycle.push(arc_idx);
            return Some(cycle);
        } else if color[next] == 0 {
            path_arcs.push(arc_idx);
            if let Some(cycle) =
                vei_cycle_visit(next, adj, alive, color, path_arcs, path_nodes)
            {
                return Some(cycle);
            }
            path_arcs.pop();
        }
    }

    path_nodes.pop();
    color[node] = 2; // black
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garp::garp_check_with_closure;

    #[test]
    fn test_vei_consistent_all_ones() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check_with_closure(&mut graph);
        let vei = compute_vei(&mut graph);
        assert!(vei.success);
        assert_eq!(vei.mean_efficiency, 1.0);
        assert_eq!(vei.min_efficiency, 1.0);
    }

    #[test]
    fn test_vei_violation_data() {
        // For 2-obs WARP violation: E[0,1]/E[0,0] = 7/8 = 0.875, same for E[1,0]/E[1,1].
        // LP relaxation constrains e_i ≥ 0.875 (from R* constraints) and maximizes Σe_i,
        // so both e_i = 1.0 (since 0.875 < 1.0 and we're maximizing).
        // The LP relaxation does NOT detect cycles - only transitive ratios > 1 would bind.
        // For exact VEI, use compute_vei_exact().
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check_with_closure(&mut graph);
        let vei = compute_vei(&mut graph);
        assert!(vei.success);
        assert_eq!(vei.efficiency_vector.len(), 2);
        // LP relaxation: both e_i = 1.0 (lower bound 0.875 is not binding at max)
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.875 - 1e-6);
        }
    }

    #[test]
    fn test_vei_efficiency_vector_bounded() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let _ = garp_check_with_closure(&mut graph);
        let vei = compute_vei(&mut graph);
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.0 && e <= 1.0, "Efficiency {e} out of [0,1]");
        }
    }

    // --- Exact VEI tests ---

    #[test]
    fn test_vei_exact_consistent() {
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        assert_eq!(vei.mean_efficiency, 1.0);
        assert_eq!(vei.min_efficiency, 1.0);
        assert_eq!(vei.total_inefficiency, 0.0);
    }

    #[test]
    fn test_vei_exact_warp_violation() {
        // 2-obs WARP violation: mutual preference, ratio = 7/8 = 0.875.
        // Exact VEI removes ONE arc (cost 0.125): total inefficiency 0.125,
        // mean efficiency (0.875 + 1.0)/2 = 0.9375. The two value-optimal
        // vectors are (0.875, 1.0) and (1.0, 0.875); the canonical convention
        // (max-min over optima, then lexicographically maximal efficiency in
        // observation order) selects (1.0, 0.875). Oracle-verified in
        // tests/test_vei_exact.py (fixture ANCHOR_2OBS).
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        assert!((vei.mean_efficiency - 0.9375).abs() < 1e-9);
        assert!((vei.total_inefficiency - 0.125).abs() < 1e-9);
        assert!(
            (vei.efficiency_vector[0] - 1.0).abs() < 1e-12
                && (vei.efficiency_vector[1] - 0.875).abs() < 1e-12,
            "canonical vector must be [1.0, 0.875], got {:?}",
            vei.efficiency_vector
        );
        assert_eq!(vei.worst_observation, 1);
    }

    #[test]
    fn test_vei_exact_nested_removal_theorem1() {
        // The Theorem 1 U-set fixture (Mononen 2023, p. 11). T=3, G=3.
        // E = [[39,37,39],[45,55,45],[51,41,51]]. Strict arcs and costs:
        // 0->1 (2/39), 1->0 (2/11), 1->2 (2/11), 2->1 (10/51). Two 2-cycles,
        // (0,1) and (1,2), share observation 1. A single adjustment
        // d_1 = 2/11 removes both arcs out of 1 and breaks both cycles, so
        // the Varian total is 2/11. Charging each removed arc independently
        // (no U-set expansion) pays 2/39 + 2/11 instead. Oracle-verified by
        // exhaustive enumeration in tests/test_vei_exact.py (NESTED_T3).
        let prices = [2.0, 3.0, 5.0, 1.0, 6.0, 6.0, 6.0, 3.0, 5.0];
        let quantities = [3.0, 1.0, 6.0, 1.0, 5.0, 4.0, 3.0, 1.0, 6.0];
        let mut graph = PreferenceGraph::new(3);
        graph.parse_budget(&prices, &quantities, 3, 3, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        let expected_total = 2.0 / 11.0;
        assert!(
            (vei.total_inefficiency - expected_total).abs() < 1e-9,
            "total inefficiency must be 2/11, got {}",
            vei.total_inefficiency
        );
        assert!((vei.mean_efficiency - (1.0 - expected_total / 3.0)).abs() < 1e-9);
        let expected = [1.0, 9.0 / 11.0, 1.0];
        for (i, (&got, &want)) in vei
            .efficiency_vector
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!((got - want).abs() < 1e-12, "e[{i}] = {got}, want {want}");
        }
        assert_eq!(vei.worst_observation, 1);
    }

    #[test]
    fn test_vei_exact_canonical_max_min_stage() {
        // Stage-B fixture: two value-optimal solutions, both totaling 4/11:
        // d = (0,0,1/11,3/11) with max 3/11, and d = (0,0,4/11,0) with max
        // 4/11. The canonical convention minimizes the maximum adjustment, so
        // the vector must be e = (1, 1, 10/11, 8/11). Oracle-verified in
        // tests/test_vei_exact.py (STAGE_B).
        let prices = [3.0, 3.0, 6.0, 2.0, 1.0, 2.0, 6.0, 2.0];
        let quantities = [1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 3.0, 2.0];
        let mut graph = PreferenceGraph::new(4);
        graph.parse_budget(&prices, &quantities, 4, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        assert!((vei.total_inefficiency - 4.0 / 11.0).abs() < 1e-9);
        let expected = [1.0, 1.0, 10.0 / 11.0, 8.0 / 11.0];
        for (i, (&got, &want)) in vei
            .efficiency_vector
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!((got - want).abs() < 1e-12, "e[{i}] = {got}, want {want}");
        }
        assert!((vei.min_efficiency - 8.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    fn test_vei_exact_canonical_lex_stage() {
        // Stage-C fixture: the only strict cycle is the 2-cycle between
        // observations 1 and 2 (both arcs cost 1/8). The two optima are
        // d = (0,1/8,0,0) and d = (0,0,1/8,0); equal max, so the
        // lexicographic stage keeps the earlier observation at efficiency 1:
        // e = (1, 1, 7/8, 1). Oracle-verified in tests/test_vei_exact.py
        // (STAGE_C).
        let prices = [4.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let quantities = [3.0, 1.0, 5.0, 1.0, 4.0, 2.0, 4.0, 1.0];
        let mut graph = PreferenceGraph::new(4);
        graph.parse_budget(&prices, &quantities, 4, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        assert!((vei.total_inefficiency - 0.125).abs() < 1e-9);
        let expected = [1.0, 1.0, 0.875, 1.0];
        for (i, (&got, &want)) in vei
            .efficiency_vector
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!((got - want).abs() < 1e-12, "e[{i}] = {got}, want {want}");
        }
        assert_eq!(vei.worst_observation, 2);
    }

    #[test]
    fn test_vei_exact_bounded() {
        let prices = [2.0, 1.0, 1.0, 2.0];
        let quantities = [3.0, 2.0, 2.0, 3.0];
        let mut graph = PreferenceGraph::new(2);
        graph.parse_budget(&prices, &quantities, 2, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.0 && e <= 1.0, "Efficiency {e} out of [0,1]");
        }
    }

    #[test]
    fn test_vei_exact_3obs_cycle() {
        // 3-obs cycle: each obs can afford the next (circular preferences)
        // p0=[3,1,1], x0=[2,1,1] → own_exp=8
        // p1=[1,3,1], x1=[1,2,1] → own_exp=8
        // p2=[1,1,3], x2=[1,1,2] → own_exp=8
        // E[0,1]=p0·x1=3+3+1=7, E[1,2]=p1·x2=1+2+6=9... too complex, use simple data
        //
        // Simpler: 3 obs, 2 goods, circular strict preferences
        let prices = [2.0, 1.0, 1.0, 2.0, 1.5, 1.5];
        let quantities = [3.0, 1.0, 1.0, 3.0, 2.0, 2.0];
        let mut graph = PreferenceGraph::new(3);
        graph.parse_budget(&prices, &quantities, 3, 2, 1e-10);
        let vei = compute_vei_exact(&mut graph);
        assert!(vei.success);
        // Should have some efficiency < 1 (GARP violated)
        assert!(vei.mean_efficiency < 1.0);
        // All bounded
        for &e in &vei.efficiency_vector {
            assert!(e >= 0.0 && e <= 1.0);
        }
    }
}
