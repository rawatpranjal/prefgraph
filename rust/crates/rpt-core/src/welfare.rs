use crate::lp::solve_afriat_lp;

/// Welfare analysis result.
pub struct WelfareResult {
    pub cv: f64,               // Compensating variation
    pub ev: f64,               // Equivalent variation
    pub utility_recovered: bool,
}

/// Compute Compensating Variation (CV) and Equivalent Variation (EV).
///
/// CV: How much money at NEW prices makes the consumer as well off as at OLD prices?
/// EV: How much money at OLD prices is the consumer willing to give up to avoid the change?
///
/// Uses Afriat utility recovery (LP) to reconstruct the piecewise-linear utility,
/// then evaluates expenditure functions at reference utility levels.
///
/// Inputs: two sets of (expenditure matrix, own_exp) for baseline and policy periods.
pub fn compute_welfare(
    e_base: &[f64],
    own_base: &[f64],
    t_base: usize,
    e_policy: &[f64],
    own_policy: &[f64],
    t_policy: usize,
    prices_base: &[f64],
    prices_policy: &[f64],
    quantities_base: &[f64],
    quantities_policy: &[f64],
    k: usize,
    tolerance: f64,
) -> WelfareResult {
    // Step 1: Recover Afriat utility from baseline data
    let baseline_lp = solve_afriat_lp(e_base, own_base, t_base, tolerance);

    let (u_base, lambda_base) = match baseline_lp {
        Some((u, l)) => (u, l),
        None => return WelfareResult { cv: 0.0, ev: 0.0, utility_recovered: false },
    };

    // Step 2: Evaluate Afriat utility at baseline and policy average bundles
    // Afriat utility: u(x) = min_k { U_k + lambda_k * p_k @ (x - x_k) }
    let avg_base_bundle: Vec<f64> = (0..k)
        .map(|g| (0..t_base).map(|t| quantities_base[t * k + g]).sum::<f64>() / t_base as f64)
        .collect();

    let avg_policy_bundle: Vec<f64> = (0..k)
        .map(|g| (0..t_policy).map(|t| quantities_policy[t * k + g]).sum::<f64>() / t_policy as f64)
        .collect();

    let u_at_base = afriat_utility(&u_base, &lambda_base, prices_base, quantities_base, t_base, k, &avg_base_bundle);
    let u_at_policy = afriat_utility(&u_base, &lambda_base, prices_base, quantities_base, t_base, k, &avg_policy_bundle);

    // Step 3: Approximate CV and EV
    // CV ≈ (u_base - u_policy) / avg_lambda  (money metric at policy prices)
    // EV ≈ (u_base - u_policy) / avg_lambda  (money metric at base prices)
    let avg_lambda = lambda_base.iter().sum::<f64>() / t_base as f64;
    let utility_diff = u_at_base - u_at_policy;

    let cv = if avg_lambda > tolerance { utility_diff / avg_lambda } else { 0.0 };
    let ev = cv; // First-order approximation: CV ≈ EV

    WelfareResult { cv, ev, utility_recovered: true }
}

/// Evaluate Afriat piecewise-linear utility at a bundle x.
/// u(x) = min_k { U_k + lambda_k * p_k @ (x - x_k) }
fn afriat_utility(
    u: &[f64], lambda: &[f64],
    prices: &[f64], quantities: &[f64],
    t: usize, k: usize,
    x: &[f64],
) -> f64 {
    let mut min_val = f64::INFINITY;
    for obs in 0..t {
        let mut dot = 0.0;
        for g in 0..k {
            dot += prices[obs * k + g] * (x[g] - quantities[obs * k + g]);
        }
        let val = u[obs] + lambda[obs] * dot;
        if val < min_val {
            min_val = val;
        }
    }
    min_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expenditure::build_expenditure;

    #[test]
    fn test_welfare_identical_periods() {
        // Same data for baseline and policy → CV = EV = 0
        let prices = [1.0, 2.0, 2.0, 1.0];
        let quantities = [4.0, 1.0, 1.0, 4.0];
        let mut e = [0.0; 4];
        let mut own = [0.0; 2];
        build_expenditure(&prices, &quantities, 2, 2, &mut e, &mut own);

        let r = compute_welfare(
            &e, &own, 2,
            &e, &own, 2,
            &prices, &prices,
            &quantities, &quantities,
            2, 1e-10,
        );
        assert!(r.utility_recovered);
        assert!(r.cv.abs() < 0.1);
    }
}
