//! Parallelized random data generators for PrefGraph benchmarking.
//!
//! Four generators produce synthetic data in the exact format that
//! Engine.analyze_arrays() / Engine.analyze_menus() expects:
//!
//! - generate_random_budgets:  list[(prices T×K, quantities T×K)]
//! - generate_random_menus:    list[(menus, choices, n_items)]
//! - generate_random_production: list[(prices T×(n_in+n_out), quantities T×(n_in+n_out))]
//! - generate_random_intertemporal: list[(prices T×K, quantities T×K)]
//!
//! All use Rayon for parallelism with deterministic per-user seeding
//! via ChaCha8Rng so results are identical regardless of thread count.

use numpy::PyArray2;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Normal, Uniform};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Seed derivation: master seed → per-user seed Vec for deterministic
// parallel execution. Each user gets its own seed derived sequentially
// from a master RNG, so thread scheduling order doesn't affect results.
// ---------------------------------------------------------------------------

fn derive_user_seeds(master_seed: u64, n_users: usize) -> Vec<u64> {
    let mut master = ChaCha8Rng::seed_from_u64(master_seed);
    (0..n_users).map(|_| master.gen::<u64>()).collect()
}

// ---------------------------------------------------------------------------
// Dirichlet sampling: sample n Gamma(1,1) variates and normalize.
// Gamma(1,1) = Exp(1) = -ln(U(0,1)), so we avoid pulling in a Gamma
// distribution crate. The result lies on the unit simplex.
// ---------------------------------------------------------------------------

fn sample_dirichlet(rng: &mut impl Rng, n: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = (0..n)
        .map(|_| {
            let u: f64 = rng.gen_range(1e-30..1.0);
            -u.ln()
        })
        .collect();
    let sum: f64 = vals.iter().sum();
    for v in vals.iter_mut() {
        *v /= sum;
    }
    vals
}

// ---------------------------------------------------------------------------
// Demand functions for budget data generation.
//
// Cobb-Douglas: q_i = alpha_i * budget / p_i
//   The classic utility U = prod(x_i^alpha_i) yields this analytical demand.
//   Always satisfies GARP when alpha is fixed across observations.
//
// CES: q_i = (alpha_i * p_i^(sigma-1)) * budget / sum_j(alpha_j * p_j^sigma)
//   Constant Elasticity of Substitution with sigma = 1/(1-rho).
//   Reduces to Cobb-Douglas as sigma → 1 and to Leontief as sigma → 0.
//
// Leontief: q_i = alpha_i * budget / dot(p, alpha)
//   Perfect complements: consume goods in fixed proportions alpha.
// ---------------------------------------------------------------------------

/// Cobb-Douglas demand: q_i = alpha_i * budget / p_i
fn cobb_douglas_demand(prices: &[f64], budget: f64, alpha: &[f64]) -> Vec<f64> {
    prices
        .iter()
        .zip(alpha.iter())
        .map(|(&p, &a)| a * budget / p)
        .collect()
}

/// CES demand with elasticity of substitution sigma.
/// q_i = (alpha_i / p_i)^sigma * budget / sum_j((alpha_j / p_j)^sigma * p_j)
fn ces_demand(prices: &[f64], budget: f64, alpha: &[f64], sigma: f64) -> Vec<f64> {
    let n = prices.len();
    // Compute (alpha_i / p_i)^sigma for each good
    let ratios: Vec<f64> = (0..n)
        .map(|i| (alpha[i] / prices[i]).powf(sigma))
        .collect();
    // Denominator: sum_j(ratio_j * p_j)
    let denom: f64 = ratios.iter().zip(prices.iter()).map(|(&r, &p)| r * p).sum();
    if denom <= 0.0 || !denom.is_finite() {
        // Fallback to Cobb-Douglas if numerical issues
        return cobb_douglas_demand(prices, budget, alpha);
    }
    ratios.iter().map(|&r| r * budget / denom).collect()
}

/// Leontief demand: q_i = alpha_i * budget / dot(p, alpha)
fn leontief_demand(prices: &[f64], budget: f64, alpha: &[f64]) -> Vec<f64> {
    let denom: f64 = prices.iter().zip(alpha.iter()).map(|(&p, &a)| p * a).sum();
    if denom <= 0.0 {
        return vec![1e-6; prices.len()];
    }
    alpha.iter().map(|&a| a * budget / denom).collect()
}

/// Apply noise to a demand bundle based on rationality and noise_scale.
/// With probability (1-rationality), multiply each quantity by exp(N(0, noise_scale)).
fn perturb_bundle(quantities: &mut [f64], rng: &mut impl Rng, rationality: f64, noise_scale: f64) {
    if rationality >= 1.0 || noise_scale <= 0.0 {
        return;
    }
    let roll: f64 = rng.gen();
    if roll >= rationality {
        let normal = Normal::new(0.0, noise_scale).unwrap();
        for q in quantities.iter_mut() {
            let noise: f64 = rng.sample(normal);
            *q *= noise.exp();
            if *q < 1e-6 {
                *q = 1e-6;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Budget generator
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BudgetParams {
    n_obs_min: usize,
    n_obs_max: usize,
    n_goods: usize,
    functional_form: u8, // 0=cobb_douglas, 1=ces, 2=leontief
    elasticity: f64,
    rationality: f64,
    noise_scale: f64,
    price_min: f64,
    price_max: f64,
    budget_min: f64,
    budget_max: f64,
}

/// Generate budget data for a single user.
/// Returns (prices_flat, quantities_flat) as row-major T×K arrays.
fn generate_budget_user(seed: u64, params: &BudgetParams) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let k = params.n_goods;

    // Number of observations for this user
    let t = if params.n_obs_min == params.n_obs_max {
        params.n_obs_min
    } else {
        rng.gen_range(params.n_obs_min..=params.n_obs_max)
    };

    // Sample preference parameters (fixed for this user)
    let alpha = sample_dirichlet(&mut rng, k);

    // CES elasticity of substitution
    let sigma = params.elasticity;

    let price_dist = Uniform::new(params.price_min, params.price_max);
    let budget_dist = Uniform::new(params.budget_min, params.budget_max);

    let mut prices_flat = Vec::with_capacity(t * k);
    let mut quantities_flat = Vec::with_capacity(t * k);

    for _ in 0..t {
        // Sample prices and budget
        let prices: Vec<f64> = (0..k).map(|_| rng.sample(price_dist)).collect();
        let budget: f64 = rng.sample(budget_dist);

        // Compute optimal demand
        let mut quantities = match params.functional_form {
            0 => cobb_douglas_demand(&prices, budget, &alpha),
            1 => ces_demand(&prices, budget, &alpha, sigma),
            2 => leontief_demand(&prices, budget, &alpha),
            _ => cobb_douglas_demand(&prices, budget, &alpha),
        };

        // Apply rationality-controlled noise
        perturb_bundle(&mut quantities, &mut rng, params.rationality, params.noise_scale);

        prices_flat.extend_from_slice(&prices);
        quantities_flat.extend_from_slice(&quantities);
    }

    (prices_flat, quantities_flat)
}

/// Generate random budget data for many users in parallel.
///
/// Returns a list of (prices, quantities) numpy array pairs, each shaped (T, K),
/// ready to feed directly into Engine.analyze_arrays().
///
/// functional_form: "cobb_douglas" | "ces" | "leontief"
/// elasticity: CES elasticity of substitution σ (only used when form="ces")
/// rationality: 0.0 = random quantities, 1.0 = exact utility-maximizing demand
/// noise_scale: standard deviation of log-normal perturbation (when rationality < 1.0)
#[pyfunction]
#[pyo3(signature = (
    n_users,
    n_obs_min,
    n_obs_max,
    n_goods,
    functional_form = 0,
    elasticity = 0.5,
    rationality = 0.7,
    noise_scale = 0.3,
    price_min = 0.5,
    price_max = 5.0,
    budget_min = 10.0,
    budget_max = 100.0,
    seed = 42
))]
pub fn generate_random_budgets<'py>(
    py: Python<'py>,
    n_users: usize,
    n_obs_min: usize,
    n_obs_max: usize,
    n_goods: usize,
    functional_form: u8,
    elasticity: f64,
    rationality: f64,
    noise_scale: f64,
    price_min: f64,
    price_max: f64,
    budget_min: f64,
    budget_max: f64,
    seed: u64,
) -> PyResult<Vec<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)>> {
    let params = BudgetParams {
        n_obs_min,
        n_obs_max,
        n_goods,
        functional_form,
        elasticity,
        rationality,
        noise_scale,
        price_min,
        price_max,
        budget_min,
        budget_max,
    };

    let user_seeds = derive_user_seeds(seed, n_users);

    // Generate all user data in parallel with GIL released
    let raw: Vec<(Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        user_seeds
            .par_iter()
            .map(|&s| generate_budget_user(s, &params))
            .collect()
    });

    // Convert to numpy arrays (needs GIL)
    let mut results = Vec::with_capacity(n_users);
    for (p_flat, q_flat) in raw {
        let t = p_flat.len() / n_goods;
        let p_arr = PyArray2::from_vec2_bound(py, &flat_to_rows(&p_flat, t, n_goods))?;
        let q_arr = PyArray2::from_vec2_bound(py, &flat_to_rows(&q_flat, t, n_goods))?;
        results.push((p_arr, q_arr));
    }
    Ok(results)
}

/// Helper: convert a flat row-major Vec<f64> into Vec<Vec<f64>> for PyArray2::from_vec2_bound
fn flat_to_rows(flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| flat[r * cols..(r + 1) * cols].to_vec())
        .collect()
}

// ---------------------------------------------------------------------------
// Menu generator
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MenuParams {
    n_obs_min: usize,
    n_obs_max: usize,
    n_items: usize,
    menu_size_min: usize,
    menu_size_max: usize,
    choice_model: u8, // 0=logit, 1=fixed_ranking, 2=uniform
    temperature: f64,
    rationality: f64,
}

/// Generate menu choice data for a single user.
/// Returns (menus: Vec<Vec<usize>>, choices: Vec<usize>, n_items).
fn generate_menu_user(
    seed: u64,
    params: &MenuParams,
) -> (Vec<Vec<usize>>, Vec<usize>, usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_items = params.n_items;

    // Number of observations for this user
    let t = if params.n_obs_min == params.n_obs_max {
        params.n_obs_min
    } else {
        rng.gen_range(params.n_obs_min..=params.n_obs_max)
    };

    // Sample preference model for this user
    let utilities: Vec<f64> = match params.choice_model {
        0 => {
            // Logit: sample utility values from Uniform(0, 10) for each item
            (0..n_items).map(|_| rng.gen_range(0.0..10.0)).collect()
        }
        1 => {
            // Fixed ranking: utility = n_items - rank (so item with rank 0 has highest utility)
            let mut perm: Vec<usize> = (0..n_items).collect();
            perm.shuffle(&mut rng);
            perm.iter().map(|&rank| (n_items - rank) as f64).collect()
        }
        _ => {
            // Uniform: all items have equal utility (choice is random)
            vec![1.0; n_items]
        }
    };

    let mut menus = Vec::with_capacity(t);
    let mut choices = Vec::with_capacity(t);

    for _ in 0..t {
        // Sample menu size
        let menu_size = if params.menu_size_min == params.menu_size_max {
            params.menu_size_min
        } else {
            rng.gen_range(params.menu_size_min..=params.menu_size_max)
        };
        let menu_size = menu_size.min(n_items); // Can't have more items than exist

        // Sample menu items without replacement (Fisher-Yates partial shuffle)
        let mut items: Vec<usize> = (0..n_items).collect();
        for i in 0..menu_size {
            let j = rng.gen_range(i..n_items);
            items.swap(i, j);
        }
        let mut menu: Vec<usize> = items[..menu_size].to_vec();
        menu.sort(); // Sorted menu for consistency

        // Choose from menu based on model
        let choice = if params.choice_model == 2 {
            // Uniform: always random
            menu[rng.gen_range(0..menu_size)]
        } else {
            // Check rationality: with probability (1-rationality), pick randomly
            let roll: f64 = rng.gen();
            if roll >= params.rationality {
                menu[rng.gen_range(0..menu_size)]
            } else {
                // Model-based choice
                match params.choice_model {
                    0 => {
                        // Logit: P(i) ∝ exp(u_i / temperature)
                        logit_choice(&menu, &utilities, params.temperature, &mut rng)
                    }
                    1 => {
                        // Fixed ranking: pick highest-utility item in menu
                        *menu
                            .iter()
                            .max_by(|&&a, &&b| {
                                utilities[a]
                                    .partial_cmp(&utilities[b])
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap()
                    }
                    _ => menu[rng.gen_range(0..menu_size)],
                }
            }
        };

        menus.push(menu);
        choices.push(choice);
    }

    (menus, choices, n_items)
}

/// Logit (softmax) choice: P(item i) ∝ exp(u_i / temperature).
/// Uses the log-sum-exp trick for numerical stability.
fn logit_choice(menu: &[usize], utilities: &[f64], temperature: f64, rng: &mut impl Rng) -> usize {
    let temp = if temperature <= 0.0 { 1e-6 } else { temperature };

    // Compute log-probabilities with max subtraction for stability
    let log_probs: Vec<f64> = menu.iter().map(|&item| utilities[item] / temp).collect();
    let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
    let sum: f64 = exp_probs.iter().sum();

    // Sample from categorical distribution
    let threshold: f64 = rng.gen::<f64>() * sum;
    let mut cumsum = 0.0;
    for (i, &ep) in exp_probs.iter().enumerate() {
        cumsum += ep;
        if cumsum >= threshold {
            return menu[i];
        }
    }
    // Fallback (rounding): last item
    *menu.last().unwrap()
}

/// Generate random menu choice data for many users in parallel.
///
/// Returns a list of (menus, choices, n_items) tuples ready to feed
/// directly into Engine.analyze_menus().
///
/// choice_model: 0=logit, 1=fixed_ranking, 2=uniform
/// temperature: logit softmax temperature (lower=more deterministic)
/// rationality: probability of following the choice model vs random pick
#[pyfunction]
#[pyo3(signature = (
    n_users,
    n_obs_min,
    n_obs_max,
    n_items,
    menu_size_min,
    menu_size_max,
    choice_model = 0,
    temperature = 1.0,
    rationality = 0.7,
    seed = 42
))]
pub fn generate_random_menus<'py>(
    py: Python<'py>,
    n_users: usize,
    n_obs_min: usize,
    n_obs_max: usize,
    n_items: usize,
    menu_size_min: usize,
    menu_size_max: usize,
    choice_model: u8,
    temperature: f64,
    rationality: f64,
    seed: u64,
) -> PyResult<Vec<(Vec<Vec<usize>>, Vec<usize>, usize)>> {
    let params = MenuParams {
        n_obs_min,
        n_obs_max,
        n_items,
        menu_size_min,
        menu_size_max,
        choice_model,
        temperature,
        rationality,
    };

    let user_seeds = derive_user_seeds(seed, n_users);

    // Generate all user data in parallel with GIL released.
    // PyO3 auto-converts Vec<(Vec<Vec<usize>>, Vec<usize>, usize)> to Python lists.
    let results: Vec<(Vec<Vec<usize>>, Vec<usize>, usize)> = py.allow_threads(|| {
        user_seeds
            .par_iter()
            .map(|&s| generate_menu_user(s, &params))
            .collect()
    });

    Ok(results)
}

// ---------------------------------------------------------------------------
// Production generator
//
// Production data models firms choosing input/output bundles to maximize
// profit. Under Cobb-Douglas technology: output = A * prod(x_i^alpha_i).
// The firm observes input prices w and output prices p, and chooses
// inputs to maximize p * f(x) - w · x.
//
// We generate data as (prices, quantities) where both are T × (n_in + n_out),
// with input prices/quantities in the first n_in columns and output
// prices/quantities in the remaining n_out columns.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ProductionParams {
    n_obs_min: usize,
    n_obs_max: usize,
    n_inputs: usize,
    n_outputs: usize,
    functional_form: u8, // 0=cobb_douglas, 1=ces, 2=leontief
    rationality: f64,
    noise_scale: f64,
}

/// Generate production data for a single firm.
fn generate_production_user(seed: u64, params: &ProductionParams) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_in = params.n_inputs;
    let n_out = params.n_outputs;
    let k = n_in + n_out;

    let t = if params.n_obs_min == params.n_obs_max {
        params.n_obs_min
    } else {
        rng.gen_range(params.n_obs_min..=params.n_obs_max)
    };

    // Technology parameters
    let alpha = sample_dirichlet(&mut rng, n_in);
    // Total factor productivity
    let tfp: f64 = rng.gen_range(1.0..5.0);
    // Output mix coefficients
    let beta = sample_dirichlet(&mut rng, n_out);

    let sigma = 0.5; // CES elasticity for production

    let mut prices_flat = Vec::with_capacity(t * k);
    let mut quantities_flat = Vec::with_capacity(t * k);

    for _ in 0..t {
        // Sample input and output prices
        let input_prices: Vec<f64> = (0..n_in).map(|_| rng.gen_range(0.5..5.0)).collect();
        let output_prices: Vec<f64> = (0..n_out).map(|_| rng.gen_range(1.0..10.0)).collect();

        // Compute optimal inputs given prices (profit-maximizing under CD)
        // For CD production: x_i = alpha_i * total_input_cost / w_i
        let total_cost: f64 = rng.gen_range(10.0..100.0);
        let mut input_quantities = match params.functional_form {
            0 => cobb_douglas_demand(&input_prices, total_cost, &alpha),
            1 => ces_demand(&input_prices, total_cost, &alpha, sigma),
            2 => leontief_demand(&input_prices, total_cost, &alpha),
            _ => cobb_douglas_demand(&input_prices, total_cost, &alpha),
        };

        // Compute output from production function: y = tfp * prod(x_i^alpha_i)
        let total_output: f64 = tfp
            * input_quantities
                .iter()
                .zip(alpha.iter())
                .map(|(&x, &a)| x.powf(a))
                .product::<f64>();

        // Split output across output goods according to beta
        let mut output_quantities: Vec<f64> = beta.iter().map(|&b| b * total_output).collect();

        // Apply noise
        perturb_bundle(
            &mut input_quantities,
            &mut rng,
            params.rationality,
            params.noise_scale,
        );
        perturb_bundle(
            &mut output_quantities,
            &mut rng,
            params.rationality,
            params.noise_scale,
        );

        // Concatenate: [input_prices..., output_prices...]
        prices_flat.extend_from_slice(&input_prices);
        prices_flat.extend_from_slice(&output_prices);
        // Concatenate: [input_quantities..., output_quantities...]
        quantities_flat.extend_from_slice(&input_quantities);
        quantities_flat.extend_from_slice(&output_quantities);
    }

    (prices_flat, quantities_flat)
}

/// Generate random production data for many firms in parallel.
///
/// Returns (prices, quantities) pairs where each array is T × (n_inputs + n_outputs).
/// Input columns come first, output columns follow.
#[pyfunction]
#[pyo3(signature = (
    n_users,
    n_obs_min,
    n_obs_max,
    n_inputs,
    n_outputs,
    functional_form = 0,
    rationality = 0.7,
    noise_scale = 0.3,
    seed = 42
))]
pub fn generate_random_production<'py>(
    py: Python<'py>,
    n_users: usize,
    n_obs_min: usize,
    n_obs_max: usize,
    n_inputs: usize,
    n_outputs: usize,
    functional_form: u8,
    rationality: f64,
    noise_scale: f64,
    seed: u64,
) -> PyResult<Vec<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)>> {
    let params = ProductionParams {
        n_obs_min,
        n_obs_max,
        n_inputs,
        n_outputs,
        functional_form,
        rationality,
        noise_scale,
    };

    let user_seeds = derive_user_seeds(seed, n_users);
    let k = n_inputs + n_outputs;

    let raw: Vec<(Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        user_seeds
            .par_iter()
            .map(|&s| generate_production_user(s, &params))
            .collect()
    });

    let mut results = Vec::with_capacity(n_users);
    for (p_flat, q_flat) in raw {
        let t = p_flat.len() / k;
        let p_arr = PyArray2::from_vec2_bound(py, &flat_to_rows(&p_flat, t, k))?;
        let q_arr = PyArray2::from_vec2_bound(py, &flat_to_rows(&q_flat, t, k))?;
        results.push((p_arr, q_arr));
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Intertemporal generator
//
// Models agents choosing consumption across time periods with exponential
// discounting: U = sum_t delta^t * log(c_t). The agent has a budget and
// faces period-specific prices. Optimal allocation under log utility:
// c_t = (delta^t / p_t) * budget / sum_s(delta^s / p_s).
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct IntertemporalParams {
    n_obs_min: usize,
    n_obs_max: usize,
    n_periods: usize,
    discount_min: f64,
    discount_max: f64,
    rationality: f64,
}

/// Generate intertemporal data for a single agent.
fn generate_intertemporal_user(seed: u64, params: &IntertemporalParams) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let k = params.n_periods;

    let t = if params.n_obs_min == params.n_obs_max {
        params.n_obs_min
    } else {
        rng.gen_range(params.n_obs_min..=params.n_obs_max)
    };

    // True discount factor for this agent
    let delta: f64 = rng.gen_range(params.discount_min..params.discount_max);

    let mut prices_flat = Vec::with_capacity(t * k);
    let mut quantities_flat = Vec::with_capacity(t * k);

    for _ in 0..t {
        // Period-specific prices (interest rates / exchange rates)
        let prices: Vec<f64> = (0..k).map(|_| rng.gen_range(0.5..5.0)).collect();
        let budget: f64 = rng.gen_range(10.0..100.0);

        // Optimal consumption under log utility with exponential discounting:
        // c_t = (delta^t / p_t) * budget / sum_s(delta^s / p_s)
        let discount_weights: Vec<f64> = (0..k)
            .map(|period| delta.powi(period as i32) / prices[period])
            .collect();
        let weight_sum: f64 = discount_weights.iter().sum();

        let mut quantities: Vec<f64> = discount_weights
            .iter()
            .map(|&w| w * budget / weight_sum)
            .collect();

        // Apply rationality-controlled noise
        let noise_scale = 0.3; // Fixed noise scale for intertemporal
        perturb_bundle(&mut quantities, &mut rng, params.rationality, noise_scale);

        prices_flat.extend_from_slice(&prices);
        quantities_flat.extend_from_slice(&quantities);
    }

    (prices_flat, quantities_flat)
}

/// Generate random intertemporal choice data for many agents in parallel.
///
/// Each agent has a true discount factor delta sampled from
/// Uniform(discount_min, discount_max) and makes T consumption allocation
/// decisions across n_periods time periods.
#[pyfunction]
#[pyo3(signature = (
    n_users,
    n_obs_min,
    n_obs_max,
    n_periods,
    discount_min = 0.8,
    discount_max = 0.99,
    rationality = 0.7,
    seed = 42
))]
pub fn generate_random_intertemporal<'py>(
    py: Python<'py>,
    n_users: usize,
    n_obs_min: usize,
    n_obs_max: usize,
    n_periods: usize,
    discount_min: f64,
    discount_max: f64,
    rationality: f64,
    seed: u64,
) -> PyResult<Vec<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)>> {
    let params = IntertemporalParams {
        n_obs_min,
        n_obs_max,
        n_periods,
        discount_min,
        discount_max,
        rationality,
    };

    let user_seeds = derive_user_seeds(seed, n_users);

    let raw: Vec<(Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        user_seeds
            .par_iter()
            .map(|&s| generate_intertemporal_user(s, &params))
            .collect()
    });

    let mut results = Vec::with_capacity(n_users);
    for (p_flat, q_flat) in raw {
        let t = p_flat.len() / n_periods;
        let p_arr = PyArray2::from_vec2_bound(py, &flat_to_rows(&p_flat, t, n_periods))?;
        let q_arr = PyArray2::from_vec2_bound(py, &flat_to_rows(&q_flat, t, n_periods))?;
        results.push((p_arr, q_arr));
    }
    Ok(results)
}
