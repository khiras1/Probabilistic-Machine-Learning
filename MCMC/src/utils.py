import numpy as np
from tqdm import tqdm


def analyze_mcmc_parameters(metropolis_sampler, calculate_mean_error, calculate_covariance_error, calculate_ess, log_target_fn, true_params, param_ranges, param_names=None, args=()):
    """
    Uruchamia MCMC z różnymi ustawieniami parametrów i zbiera metryki wydajności.
    (Pełny docstring jak w poprzedniej wersji)
    """
    mu, cov = true_params
    n_dim = len(mu)

    if param_names is None:
        param_names = [f'param_{i+1}' for i in range(n_dim)]

    results = {
        'proposal_std': {'values': [], 'acceptance_rates': [], 'ess': [], 'mean_errors': [], 'cov_errors': []},
        'initial_x': {'values': [], 'convergence_times': [], 'mean_errors': [], 'cov_errors': [], 'trajectories': []},
        'n_burnin': {'values': [], 'mean_errors': [], 'cov_errors': []},
        'n_samples': {'values': [], 'ess': [], 'mean_errors': [], 'cov_errors': []}
    }

    default_initial_x = np.zeros(n_dim)
    default_std_val = param_ranges.get('proposal_std', [0.5])[0]
    if isinstance(default_std_val, (int, float)):
        default_std = np.full(n_dim, default_std_val)
    else:
        default_std = np.asarray(default_std_val)

    default_burnin = param_ranges.get('n_burnin', [1000])[2]
    default_samples = param_ranges.get('n_samples', [2000])[2]

    # === 1. Testowanie różnych `proposal_std` ===
    if 'proposal_std' in param_ranges:
        print("Rozpoczynanie testów dla 'proposal_std'...")
        for std_val in tqdm(param_ranges['proposal_std'], desc="Testowanie proposal_std"):
            if isinstance(std_val, (int, float)):
                current_std = np.full(n_dim, std_val)
            else:
                current_std = np.asarray(std_val)
                if current_std.shape != (n_dim,) and current_std.size != n_dim: # Elastyczniejsze sprawdzanie
                     print(f"Warning: Kształt proposal_std {current_std.shape} niezgodny z wymiarem {n_dim}. Używam skalara {std_val} dla wszystkich wymiarów.")
                     current_std = np.full(n_dim, std_val if isinstance(std_val, (int, float)) else np.mean(std_val)) #Fallback
                elif current_std.size == n_dim and current_std.shape != (n_dim,):
                     current_std = current_std.reshape(n_dim) # Próba reshape

            results['proposal_std']['values'].append(std_val) 

            samples_df, diagnostics = metropolis_sampler( 
                log_target_fn, default_initial_x, current_std, args=args,
                n_burnin=default_burnin, n_samples=default_samples,
                param_names=param_names, return_diagnostics=True
            )
            
            results['proposal_std']['acceptance_rates'].append(diagnostics.get('acceptance_rate', np.nan))
            ess_per_param = [calculate_ess(samples_df[param].values) for param in param_names]
            results['proposal_std']['ess'].append(np.mean(ess_per_param))
            results['proposal_std']['mean_errors'].append(calculate_mean_error(samples_df, mu, param_names))
            results['proposal_std']['cov_errors'].append(calculate_covariance_error(samples_df, cov, param_names))

    # === 2. Testowanie różnych `initial_x` ===
    if 'initial_x' in param_ranges:
        print("\nRozpoczynanie testów dla 'initial_x'...")
        extended_run_len = default_burnin * 2 + default_samples 
        
        for start_point in tqdm(param_ranges['initial_x'], desc="Testowanie initial_x"):
            start_point = np.asarray(start_point)
            results['initial_x']['values'].append(start_point)

            # Uruchomienie MCMC
            all_samples_df = metropolis_sampler(
                log_target_fn, start_point, default_std, args=args,
                n_burnin=0, n_samples=extended_run_len,
                param_names=param_names, return_diagnostics=False 
            )

            # Zapisanie trajektorii
            trajectory_len = min(extended_run_len, default_burnin * 2)
            results['initial_x']['trajectories'].append(all_samples_df[param_names].iloc[:trajectory_len].values)
            
            # Oszacowanie czasu zbieżności (heurystyka - dostarczona)
            convergence_time = extended_run_len # Domyślnie
            window = default_burnin // 4 
            if window < 10: window = 10
            if len(all_samples_df) > window:
                 running_means = all_samples_df[param_names].rolling(window=window).mean().dropna()
                 if not running_means.empty:
                     final_mean = all_samples_df[param_names].iloc[-default_samples:].mean().values 
                     diffs = np.linalg.norm(running_means.values - final_mean, axis=1) / (np.linalg.norm(final_mean) + 1e-9)
                     threshold = 0.1 
                     converged_indices = np.where(diffs < threshold)[0]
                     if len(converged_indices) > 0:
                         first_converged_idx = running_means.index[converged_indices[0]]
                         check_window_start = first_converged_idx
                         check_window_end = min(first_converged_idx + window, len(all_samples_df)-1)
                         if check_window_end > check_window_start:
                             post_convergence_diffs = np.linalg.norm(all_samples_df[param_names].iloc[check_window_start:check_window_end].mean().values - final_mean) / (np.linalg.norm(final_mean) + 1e-9)
                             if post_convergence_diffs < threshold * 1.5: 
                                 convergence_time = first_converged_idx
            results['initial_x']['convergence_times'].append(convergence_time)

            post_convergence_samples = all_samples_df.iloc[-default_samples:]
            results['initial_x']['mean_errors'].append(calculate_mean_error(post_convergence_samples, mu, param_names))
            results['initial_x']['cov_errors'].append(calculate_covariance_error(post_convergence_samples, cov, param_names))

    # === 3. Testowanie różnych `n_burnin` ===
    if 'n_burnin' in param_ranges:
         print("\nRozpoczynanie testów dla 'n_burnin'...")
         for burnin in tqdm(param_ranges['n_burnin'], desc="Testowanie n_burnin"):
            results['n_burnin']['values'].append(burnin)

            samples_df = metropolis_sampler(
                log_target_fn, default_initial_x, default_std, args=args,
                n_burnin=burnin, n_samples=default_samples,
                param_names=param_names, return_diagnostics=False
            )
            
            results['n_burnin']['mean_errors'].append(calculate_mean_error(samples_df, mu, param_names))
            results['n_burnin']['cov_errors'].append(calculate_covariance_error(samples_df, cov, param_names))

    # === 4. Testowanie różnych `n_samples` ===
    if 'n_samples' in param_ranges:
         print("\nRozpoczynanie testów dla 'n_samples'...")
         for n_s in tqdm(param_ranges['n_samples'], desc="Testowanie n_samples"):
            results['n_samples']['values'].append(n_s)

            samples_df = metropolis_sampler(
                log_target_fn, default_initial_x, default_std, args=args,
                n_burnin=default_burnin, n_samples=n_s, 
                param_names=param_names, return_diagnostics=False 
            )
            
            ess_per_param = [calculate_ess(samples_df[param].values) for param in param_names]
            results['n_samples']['ess'].append(np.mean(ess_per_param))
            results['n_samples']['mean_errors'].append(calculate_mean_error(samples_df, mu, param_names))
            results['n_samples']['cov_errors'].append(calculate_covariance_error(samples_df, cov, param_names))

    print("\nZakończono analizę parametrów MCMC.")
    return results
