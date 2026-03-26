use highs::{HighsModelStatus, Model, RowProblem, Sense};

/// Solve the Afriat feasibility LP: find utility values U and marginal
/// utilities λ satisfying Afriat's inequalities.
///
/// For all pairs (i,j) where i is revealed preferred to j:
///   U_i - U_j ≤ λ_j * (E[j,i] - E[j,j])
///
/// Variables: [U_0..U_{T-1}, λ_0..λ_{T-1}] (2T variables)
/// Objective: minimize sum(λ) (for a centered solution)
///
/// Returns Some((utilities, lambdas)) if feasible, None otherwise.
pub fn solve_afriat_lp(
    e: &[f64],
    own_exp: &[f64],
    t: usize,
    tolerance: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_vars = 2 * t;
    let lambda_lb = 1e-6;

    let mut pb = RowProblem::default();

    // Variables: U_0..U_{T-1} (bounds: [0, inf]), λ_0..λ_{T-1} (bounds: [lambda_lb, inf])
    // Cost: minimize sum(λ)
    let mut cols = Vec::with_capacity(n_vars);
    for i in 0..t {
        cols.push(pb.add_column(0.0, 0.0..)); // U_i: cost=0, lb=0
    }
    for i in 0..t {
        cols.push(pb.add_column(1.0, lambda_lb..)); // λ_i: cost=1, lb=lambda_lb
    }

    // Constraints: for each (i,j) with i≠j:
    //   U_i - U_j - λ_j * (E[j,i] - E[j,j]) ≤ 0
    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            // Coefficient of λ_j is -(E[j,i] - E[j,j])
            let lambda_coeff = -(e[j * t + i] - own_exp[j]);

            // Row: U_i - U_j + lambda_coeff * λ_j ≤ 0
            pb.add_row(
                ..0.0, // upper bound = 0
                [(cols[i], 1.0), (cols[j], -1.0), (cols[t + j], lambda_coeff)],
            );
        }
    }

    let mut model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            let u: Vec<f64> = (0..t).map(|i| sol.columns()[i]).collect();
            let lambdas: Vec<f64> = (0..t).map(|i| sol.columns()[t + i]).collect();
            Some((u, lambdas))
        }
        _ => None,
    }
}

/// Solve the Houtman-Maks ILP: find the maximum subset of observations
/// that satisfies GARP (via Big-M Afriat formulation).
///
/// Binary vars: z_i (1 = keep observation i)
/// Continuous vars: U_i, λ_i
/// Constraint: U_i - U_j - λ_j*(E[j,i]-E[j,j]) ≤ M*(2 - z_i - z_j)
/// Objective: maximize sum(z_i)
///
/// Returns indices of observations to REMOVE.
pub fn solve_hm_ilp(
    e: &[f64],
    own_exp: &[f64],
    t: usize,
    tolerance: f64,
) -> Vec<usize> {
    let max_exp = own_exp.iter().cloned().fold(0.0f64, f64::max);
    let big_m = (3.0 * max_exp).max(10.0);
    // lambda_lb must be large enough that constraint violations exceed solver tolerance.
    // With diff ~ 1.0 and lambda ~ 0.01, violation ~ 0.01 >> solver_tol (~1e-7).
    let lambda_lb = 0.01;

    let mut pb = RowProblem::default();

    // Variables: z_0..z_{T-1} (binary [0,1]), U_0..U_{T-1}, λ_0..λ_{T-1}
    let mut z_cols = Vec::with_capacity(t);
    let mut u_cols = Vec::with_capacity(t);
    let mut l_cols = Vec::with_capacity(t);

    for _i in 0..t {
        z_cols.push(pb.add_integer_column(-1.0, 0.0..1.0)); // maximize z = minimize -z
    }
    for _i in 0..t {
        u_cols.push(pb.add_column(0.0, 0.0..big_m)); // U_i
    }
    for _i in 0..t {
        l_cols.push(pb.add_column(0.0, lambda_lb..big_m)); // λ_i
    }

    // Constraints: for each (i,j) with i≠j:
    //   U_i - U_j - λ_j*(E[j,i]-E[j,j]) + M*z_i + M*z_j ≤ 2*M
    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            let lambda_coeff = -(e[j * t + i] - own_exp[j]);

            pb.add_row(
                ..2.0 * big_m,
                [
                    (u_cols[i], 1.0),
                    (u_cols[j], -1.0),
                    (l_cols[j], lambda_coeff),
                    (z_cols[i], big_m),
                    (z_cols[j], big_m),
                ],
            );
        }
    }

    let mut model = pb.optimise(Sense::Minimise);
    let solved = model.solve();

    match solved.status() {
        HighsModelStatus::Optimal => {
            let sol = solved.get_solution();
            let mut removed = Vec::new();
            for i in 0..t {
                if sol.columns()[i] < 0.5 {
                    removed.push(i);
                }
            }
            removed
        }
        _ => {
            // Fallback: can't solve ILP, return empty
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Optional Gurobi backend (feature = "gurobi")
// ---------------------------------------------------------------------------

/// Solve the Afriat feasibility LP using Gurobi (requires license).
///
/// Identical formulation to `solve_afriat_lp` but uses Gurobi's LP solver,
/// which is ~10x faster than HiGHS for large MIP problems (Machado, 2024).
#[cfg(feature = "gurobi")]
pub fn solve_afriat_lp_gurobi(
    e: &[f64],
    own_exp: &[f64],
    t: usize,
    tolerance: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    use grb::prelude::*;

    let lambda_lb = 1e-6;

    let mut env = Env::empty().ok()?.start().ok()?;
    env.set(param::OutputFlag, 0).ok()?;

    let mut model = Model::with_env("afriat", &env).ok()?;

    // Variables: U_i (continuous, lb=0) and λ_i (continuous, lb=lambda_lb)
    let u: Vec<Var> = (0..t)
        .map(|i| add_ctsvar!(model, name: &format!("u_{}", i), bounds: 0.0..).unwrap())
        .collect();
    let lam: Vec<Var> = (0..t)
        .map(|i| {
            add_ctsvar!(model, name: &format!("l_{}", i), bounds: lambda_lb..).unwrap()
        })
        .collect();

    // Objective: minimize sum(λ)
    model.set_objective(lam.iter().grb_sum(), Minimize).ok()?;

    // Constraints: U_i - U_j - λ_j * (E[j,i] - E[j,j]) <= 0
    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            let lambda_coeff = -(e[j * t + i] - own_exp[j]);
            model
                .add_constr(
                    &format!("c_{}_{}", i, j),
                    c!(u[i] - u[j] + lambda_coeff * lam[j] <= 0.0),
                )
                .ok()?;
        }
    }

    model.optimize().ok()?;

    match model.status().ok()? {
        Status::Optimal => {
            let u_vals: Vec<f64> = u.iter().map(|v| model.get_obj_attr(attr::X, v).unwrap()).collect();
            let l_vals: Vec<f64> = lam.iter().map(|v| model.get_obj_attr(attr::X, v).unwrap()).collect();
            Some((u_vals, l_vals))
        }
        _ => None,
    }
}

/// Solve the Houtman-Maks ILP using Gurobi (requires license).
///
/// ~10x faster than HiGHS for MIP problems. Same Big-M Afriat formulation.
#[cfg(feature = "gurobi")]
pub fn solve_hm_ilp_gurobi(
    e: &[f64],
    own_exp: &[f64],
    t: usize,
    tolerance: f64,
) -> Vec<usize> {
    use grb::prelude::*;

    let max_exp = own_exp.iter().cloned().fold(0.0f64, f64::max);
    let big_m = (3.0 * max_exp).max(10.0);
    let lambda_lb = 0.01;

    let env = match Env::empty().and_then(|e| e.start()) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut model = match Model::with_env("hm", &env) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    let _ = model.set_param(param::OutputFlag, 0);

    // Variables
    let z: Vec<Var> = (0..t)
        .map(|i| add_binvar!(model, name: &format!("z_{}", i)).unwrap())
        .collect();
    let u: Vec<Var> = (0..t)
        .map(|i| add_ctsvar!(model, name: &format!("u_{}", i), bounds: 0.0..big_m).unwrap())
        .collect();
    let lam: Vec<Var> = (0..t)
        .map(|i| {
            add_ctsvar!(model, name: &format!("l_{}", i), bounds: lambda_lb..big_m).unwrap()
        })
        .collect();

    // Objective: maximize sum(z) = minimize sum(-z)
    model
        .set_objective(z.iter().map(|v| (*v, -1.0)).grb_sum(), Minimize)
        .ok();

    // Constraints
    for i in 0..t {
        for j in 0..t {
            if i == j {
                continue;
            }
            let lambda_coeff = -(e[j * t + i] - own_exp[j]);
            let _ = model.add_constr(
                &format!("c_{}_{}", i, j),
                c!(u[i] - u[j] + lambda_coeff * lam[j] + big_m * z[i] + big_m * z[j] <= 2.0 * big_m),
            );
        }
    }

    if model.optimize().is_err() {
        return Vec::new();
    }

    match model.status() {
        Ok(Status::Optimal) => {
            let mut removed = Vec::new();
            for i in 0..t {
                if let Ok(val) = model.get_obj_attr(attr::X, &z[i]) {
                    if val < 0.5 {
                        removed.push(i);
                    }
                }
            }
            removed
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_afriat_lp_consistent() {
        // Consistent data: should find feasible U, λ
        // p=[[1,2],[2,1]], q=[[4,1],[1,4]]
        // E = [[6,3],[9,6]]
        let e = [6.0, 3.0, 9.0, 6.0];
        let own = [6.0, 6.0];
        let result = solve_afriat_lp(&e, &own, 2, 1e-10);
        assert!(result.is_some());
    }

    #[test]
    fn test_afriat_lp_violation() {
        // Violation data: p=[[2,1],[1,2]], q=[[3,2],[2,3]]
        // E = [[8,7],[7,8]]
        let e = [8.0, 7.0, 7.0, 8.0];
        let own = [8.0, 8.0];
        let result = solve_afriat_lp(&e, &own, 2, 1e-10);
        assert!(result.is_none());
    }

    #[test]
    fn test_hm_ilp_consistent() {
        let e = [6.0, 3.0, 9.0, 6.0];
        let own = [6.0, 6.0];
        let removed = solve_hm_ilp(&e, &own, 2, 1e-10);
        assert_eq!(removed.len(), 0); // No removals needed
    }

    #[test]
    fn test_hm_ilp_violation() {
        let e = [8.0, 7.0, 7.0, 8.0];
        let own = [8.0, 8.0];
        let removed = solve_hm_ilp(&e, &own, 2, 1e-10);
        assert_eq!(removed.len(), 1); // Must remove 1 observation
    }
}
