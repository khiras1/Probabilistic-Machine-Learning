import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns


def confidence_ellipse(ax, x, y, cov, n_std=2.0, facecolor='none', **kwargs):
    """
    Rysuje elipsę ufności dla rozkładu 2D na podanych osiach.
    n_std=2.0 odpowiada ok. 95% przedziałowi ufności dla rozkładu normalnego.
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    # Skalowanie i obrót elipsy
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_mcmc_results(samples, mu_true, cov_true, param_names=['x1', 'x2']):
    """
    Tworzy wizualizacje wyników próbkowania MCMC dla problemu 2D.
    Pokazuje: rozrzut próbek z elipsami ufności, histogramy brzegowe 
    oraz ścieżki łańcucha (trace plots).
    """
    if not all(p in samples.columns for p in param_names):
        raise ValueError(f"DataFrame 'samples' must contain columns named: {param_names}")
    if 'log_prob' not in samples.columns:
         raise ValueError("DataFrame 'samples' must contain column 'log_prob'")   

    sample_mean = samples[param_names].mean().values
    sample_cov = np.cov(samples[param_names].values.T)
    
    # --- Wykres 1: Rozrzut próbek i elipsy ufności ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    ax1.scatter(samples[param_names[0]], samples[param_names[1]], alpha=0.3, s=5, label='Próbki MCMC')
    ax1.set_xlabel(param_names[0])
    ax1.set_ylabel(param_names[1])
    ax1.set_title('Próbki MCMC i elipsy ufności (95%)')
    
    # Elipsa dla rozkładu prawdziwego
    confidence_ellipse(ax1, mu_true[0], mu_true[1], cov_true, n_std=2.0, 
                       edgecolor='red', linestyle='--', label='Prawdziwa (95%)')
    # Elipsa dla próbek
    confidence_ellipse(ax1, sample_mean[0], sample_mean[1], sample_cov, n_std=2.0, 
                       edgecolor='blue', linestyle='-', label='Estymowana z próbek (95%)')
    
    ax1.plot(mu_true[0], mu_true[1], 'rx', markersize=10, label='Prawdziwa średnia')
    ax1.plot(sample_mean[0], sample_mean[1], 'bx', markersize=10, label='Średnia z próbek')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Wykres 2: Histogramy brzegowe ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, param_name in enumerate(param_names):
        ax = axes2[i]
        # Histogram
        ax.hist(samples[param_name], bins=30, density=True, alpha=0.7, label='Histogram próbek')
        
        # Prawdziwy rozkład brzegowy (normalny)
        mu_p = mu_true[i]
        std_p = np.sqrt(cov_true[i, i])
        x_range = np.linspace(mu_p - 4*std_p, mu_p + 4*std_p, 200)
        pdf_true = (1 / (np.sqrt(2 * np.pi) * std_p)) * np.exp(-0.5 * ((x_range - mu_p) / std_p)**2)
        ax.plot(x_range, pdf_true, 'r-', label='Prawdziwa gęstość')
        
        # Średnie
        ax.axvline(mu_true[i], color='r', linestyle='--', alpha=0.9, label='Prawdziwa średnia')
        ax.axvline(sample_mean[i], color='b', linestyle=':', alpha=0.9, label='Średnia z próbek')
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Gęstość')
        ax.set_title(f'Rozkład brzegowy {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

    # --- Wykres 3: Ścieżki łańcucha (Trace Plots) ---
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Ścieżka dla parametru 1
    axes3[0].plot(samples.index, samples[param_names[0]], alpha=0.7)
    axes3[0].axhline(mu_true[0], color='r', linestyle='--', alpha=0.7, label='Prawdziwa średnia')
    axes3[0].set_ylabel(param_names[0])
    axes3[0].set_title(f'Ścieżka MCMC dla {param_names[0]}')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)
    
    # Ścieżka dla parametru 2
    axes3[1].plot(samples.index, samples[param_names[1]], alpha=0.7)
    axes3[1].axhline(mu_true[1], color='r', linestyle='--', alpha=0.7, label='Prawdziwa średnia')
    axes3[1].set_ylabel(param_names[1])
    axes3[1].set_title(f'Ścieżka MCMC dla {param_names[1]}')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    
    # Ścieżka dla log-prawdopodobieństwa
    axes3[2].plot(samples.index, samples['log_prob'], alpha=0.7, color='green')
    axes3[2].set_xlabel('Indeks próbki (po burn-in)')
    axes3[2].set_ylabel('Log-prawdopodobieństwo')
    axes3[2].set_title('Ścieżka MCMC dla log-prawdopodobieństwa')
    axes3[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_mcmc_efficiency_results(results, true_params, log_target_fn=None, param_names=None):
    """
    Visualizes the MCMC efficiency analysis results stored in the 'results' dict.

    Creates plots showing the impact of proposal_std, initial_x, n_burnin, 
    and n_samples on acceptance rate, ESS, errors, and convergence.
    Uses the user's preferred style for the MCMC trajectory plot.

    Parameters
    ----------
    results : dict
        Dictionary with analysis results from analyze_mcmc_parameters.
    true_params : tuple (mu, cov)
        True parameters (mean vector, covariance matrix) of the target distribution.
    log_target_fn : callable, optional
        Function computing log probability of the target distribution. 
        Needed for plotting target contours in the trajectory plot.
    param_names : list of str, optional
        Names of the parameters (e.g., ['x1', 'x2']). If None, default names
        will be assumed based on dimension.
    """
    mu, cov = true_params
    n_dim = len(mu)

    if param_names is None:
        param_names = [f'param_{i+1}' for i in range(n_dim)]
        print(f"Warning: param_names not provided, using default: {param_names}")
        
    # Determine grid size dynamically
    n_plots_total = 0
    plot_config = {} # Store which plots to generate
    if 'proposal_std' in results and results['proposal_std']['values']: 
        n_plots_total += 2; plot_config['proposal_std'] = True
    if 'initial_x' in results and results['initial_x']['values']: 
        n_plots_total += 2; plot_config['initial_x'] = True # Takes 2 slots
    if 'n_burnin' in results and results['n_burnin']['values']: 
        n_plots_total += 2; plot_config['n_burnin'] = True
    if 'n_samples' in results and results['n_samples']['values']: 
        n_plots_total += 2; plot_config['n_samples'] = True
    
    if n_plots_total == 0:
        print("No results found in the dictionary to plot.")
        return

    n_plot_rows = (n_plots_total + 1) // 2 # Calculate rows needed for 2 columns
        
    fig, axes = plt.subplots(n_plot_rows, 2, figsize=(14, 5 * n_plot_rows))
    axes = axes.flatten() # Flatten axes array for easy indexing
    plot_idx = 0

    # --- 1. Proposal Standard Deviation Analysis (Improved Version) ---
    if plot_config.get('proposal_std'):
        res_std = results['proposal_std']
        std_labels = []
        for std in res_std['values']:
             if isinstance(std, (np.ndarray, list)) and len(std) > 0: std_labels.append(f"[{std[0]:.1f},..]" if len(std) > 1 else f"{std[0]:.1f}")
             elif isinstance(std, (int, float)): std_labels.append(f"{std:.1f}")
             else: std_labels.append(str(std)) 

        # Plot 1.1: Acceptance rate
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            sns.barplot(x=std_labels, y=res_std['acceptance_rates'], ax=ax, palette="viridis", hue=res_std['acceptance_rates'])
            ax.axhline(y=0.234, color='r', linestyle='--', label='Teoret. optimum (≈0.234)') 
            ax.set_xlabel('Proposal Standard Deviation (σ)')
            ax.set_ylabel('Acceptance Rate')
            ax.set_title('Wpływ σ propozycji na akceptację')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            plot_idx += 1

        # Plot 1.2: ESS
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            sns.barplot(x=std_labels, y=res_std['ess'], ax=ax, palette="viridis", hue=res_std['ess'])
            ax.set_xlabel('Proposal Standard Deviation (σ)')
            ax.set_ylabel('Effective Sample Size (ESS)')
            ax.set_title('Wpływ σ propozycji na efektywność (ESS)')
            ax.tick_params(axis='x', rotation=45)
            plot_idx += 1

        # --- 2. Initial Point Analysis ---
        if 'initial_x' in results and results['initial_x']['values']:
            res_init = results['initial_x']
            if not res_init['values']:
                print("Skipping initial_x plots: No values found.")
            else:
                # Plot 2.1: Convergence time vs initial point distance
                if plot_idx < len(axes):
                    ax = axes[plot_idx]
                    initial_points = np.array(res_init['values'])
                    conv_times = np.array(res_init['convergence_times'])
                    # Calculate distance from true mean for each initial point
                    distances = np.linalg.norm(initial_points - mu, axis=1)
                    
                    # Sort by distance for potentially clearer visualization (optional)
                    # sorted_indices = np.argsort(distances)
                    # sorted_distances = distances[sorted_indices]
                    # sorted_conv_times = conv_times[sorted_indices]

                    # Scatter plot: color indicates convergence time
                    sc = ax.scatter(distances, conv_times, c=conv_times, 
                                    cmap='plasma', alpha=0.8, s=50)
                    fig.colorbar(sc, ax=ax, label='Czas zbieżności (heurystyczny)')
                    
                    ax.set_xlabel('Odległość punktu startowego od prawdziwej średniej')
                    ax.set_ylabel('Oszacowany czas zbieżności (iteracje)')
                    ax.set_title('Zależność czasu zbieżności od punktu startowego')
                    ax.grid(True, alpha=0.3)
                    plot_idx += 1
            
            # Plot 2.2: Trajectory visualization
            if log_target_fn is not None and param_names is not None and plot_idx < len(axes):
                ax = axes[plot_idx]
                
                # Create contour plot of the target distribution
                x = np.linspace(mu[0] - 3*np.sqrt(cov[0,0]), mu[0] + 3*np.sqrt(cov[0,0]), 100)
                y = np.linspace(mu[1] - 3*np.sqrt(cov[1,1]), mu[1] + 3*np.sqrt(cov[1,1]), 100)
                X, Y = np.meshgrid(x, y)
                pos = np.dstack((X, Y))
                
                # Calculate PDF values
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i,j] = np.exp(log_target_fn(np.array([X[i,j], Y[i,j]]), mu, np.linalg.inv(cov)))
                
                # Plot contour
                ax.contour(X, Y, Z, levels=10, alpha=0.3, cmap='Blues')
                
                # Plot trajectories for a subset of initial points
                num_trajectories = min(5, len(res_init['trajectories']))
                selected_indices = np.linspace(0, len(res_init['trajectories'])-1, num_trajectories, dtype=int)
                
                for idx in selected_indices:
                    trajectory = res_init['trajectories'][idx]
                    init_point = res_init['values'][idx]
                    conv_time = res_init['convergence_times'][idx]
                    
                    # Plot trajectory with gradient color
                    points = trajectory[:min(500, len(trajectory))]
                    colors = np.arange(len(points))
                    
                    sns.scatterplot(x=points[:, 0], y=points[:, 1], 
                                  hue=colors, palette='viridis',
                                  s=10, alpha=0.5, legend=False, ax=ax)
                    ax.plot(points[:, 0], points[:, 1], alpha=0.3)
                    
                    # Mark initial point
                    ax.scatter(init_point[0], init_point[1], color='red', 
                             s=100, marker='o', label='')
                    
                    # Mark convergence point
                    if conv_time < len(trajectory):
                        ax.scatter(trajectory[conv_time, 0], trajectory[conv_time, 1],
                                 color='green', s=100, marker='*', label='')
                
                # Mark true mean
                ax.scatter(mu[0], mu[1], color='black', s=150, marker='X', label='Prawdziwa średnia')
                
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
                ax.set_title('Trajektorie MCMC z różnych punktów startowych')
                ax.legend(loc='lower right')
                plot_idx += 1

    # --- 3. Burn-in Period Analysis (Improved Version) ---
    if plot_config.get('n_burnin'):
        res_burn = results['n_burnin']
        if not res_burn['values']:
             print("Skipping n_burnin plots: No values found.")
        else:
            burnin_values = res_burn['values']
            
            # Plot 3.1: Mean error vs burn-in size
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(burnin_values, res_burn['mean_errors'], 'o-', color='dodgerblue', linewidth=1.5, markersize=6)
                ax.set_xlabel('Liczba próbek burn-in')
                ax.set_ylabel('Błąd estymacji średniej')
                ax.set_title('Wpływ długości burn-in na błąd średniej')
                ax.grid(True, alpha=0.3)
                plot_idx += 1

            # Plot 3.2: Covariance error vs burn-in size
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(burnin_values, res_burn['cov_errors'], 'o-', color='darkorange', linewidth=1.5, markersize=6)
                ax.set_xlabel('Liczba próbek burn-in')
                ax.set_ylabel('Błąd estymacji kowariancji')
                ax.set_title('Wpływ długości burn-in na błąd kowariancji')
                ax.grid(True, alpha=0.3)
                plot_idx += 1

    # --- 4. Sample Size Analysis (Improved Version) ---
    if plot_config.get('n_samples'):
        res_samp = results['n_samples']
        if not res_samp['values']:
             print("Skipping n_samples plots: No values found.")
        else:
            samples_values = np.array(res_samp['values'])
            mean_errors = np.array(res_samp['mean_errors'])
            cov_errors = np.array(res_samp['cov_errors'])
            ess_values = np.array(res_samp['ess'])
            
            # Plot 4.1: Errors vs sample size
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(samples_values, mean_errors, 'o-', color='dodgerblue', label='Błąd średniej')
                ax.plot(samples_values, cov_errors, 's--', color='darkorange', label='Błąd kowariancji')
                
                # Add 1/sqrt(N) reference line
                valid_mean_err = ~np.isnan(mean_errors) & (mean_errors > 0)
                if len(samples_values[valid_mean_err]) > 0:
                    first_valid_idx = np.where(valid_mean_err)[0][0]
                    ref_y_mean = mean_errors[first_valid_idx] * np.sqrt(samples_values[first_valid_idx] / samples_values)
                    ax.plot(samples_values, ref_y_mean, 'k:', alpha=0.6, label='Ref. ~1/√N')
                
                ax.set_xlabel('Liczba próbek (N, po burn-in)')
                ax.set_ylabel('Błąd estymacji')
                ax.set_title('Wpływ liczby próbek na dokładność')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1

            # Plot 4.2: ESS vs sample size
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                valid_ess = ~np.isnan(ess_values) & (samples_values > 0)
                ess_ratio = np.full_like(ess_values, np.nan, dtype=float)
                ess_ratio[valid_ess] = ess_values[valid_ess] / samples_values[valid_ess]
                
                bars = ax.bar([str(s) for s in samples_values], ess_values, color='mediumseagreen')
                labels = [f'{ratio:.2f}' if not np.isnan(ratio) else 'N/A' for ratio in ess_ratio]
                ax.bar_label(bars, labels=labels, label_type='edge', padding=2, fontsize=9)
                
                ax.set_xlabel('Liczba próbek (N, po burn-in)')
                ax.set_ylabel('Effective Sample Size (ESS)')
                ax.set_title('ESS vs N (współczynnik ESS/N na słupkach)')
                ax.tick_params(axis='x', rotation=45)
                plot_idx += 1

    # Remove any unused axes
    while plot_idx < len(axes.flatten()):
        fig.delaxes(axes.flatten()[plot_idx])
        plot_idx += 1
        
    plt.tight_layout(pad=2.0, h_pad=3.0) 
    plt.show()



def plot_marginal_histograms_grid(samples, mu_true, param_names, cov_highdim, n_cols=3):
    """
    Tworzy histogramy brzegowe dla wszystkich wymiarów w siatce.
    """
    num_params = len(param_names)
    n_rows = (num_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()  

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        ax.hist(samples[param_name], bins=50, density=True, alpha=0.7, label='Histogram próbek')

        mu_p = mu_true[i]
        std_p = np.sqrt(cov_highdim[i, i])
        x_range = np.linspace(mu_p - 4 * std_p, mu_p + 4 * std_p, 200)
        pdf_true = (1 / (np.sqrt(2 * np.pi) * std_p)) * np.exp(-0.5 * ((x_range - mu_p) / std_p)**2)
        ax.plot(x_range, pdf_true, 'r-', label='Prawdziwa gęstość')

        ax.axvline(mu_true[i], color='r', linestyle='--', alpha=0.9, label='Prawdziwa średnia')
        ax.axvline(samples[param_name].mean(), color='b', linestyle=':', alpha=0.9, label='Średnia z próbek')

        ax.set_xlabel(param_name)
        ax.set_ylabel('Gęstość')
        ax.set_title(f'Rozkład brzegowy dla {param_name}')
        if i == 0:
            ax.legend()
        else:
            ax.legend([])
        ax.grid(True, alpha=0.3)

    if num_params < n_rows * n_cols:
        for j in range(num_params, n_rows * n_cols):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_traceplots_grid(samples, mu_true, param_names, n_cols=3):
    """
    Tworzy trace plots dla wszystkich wymiarów w siatce.
    """
    num_vars = len(param_names)
    n_rows = (num_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True)
    axes = axes.flatten()  

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        ax.plot(samples.index, samples[param_name], alpha=0.7)
        ax.axhline(mu_true[i], color='r', linestyle='--', alpha=0.7, label='Prawdziwa średnia')
        ax.set_ylabel(param_name)
        ax.set_title(f'Ścieżka MCMC dla {param_name}')
        if i == 0:
            ax.legend()
        else:
            ax.legend([])
        ax.grid(True, alpha=0.3)

    if num_vars < n_rows * n_cols:
        for j in range(num_vars, n_rows * n_cols):
            fig.delaxes(axes[j])

    plt.xlabel('Indeks próbki (po burn-in)')
    plt.tight_layout()
    plt.show()

def plot_logprob_trace_highdim(samples):
    """
    Tworzy trace plot dla log-prawdopodobieństwa.
    """
    if 'log_prob' not in samples.columns:
        raise ValueError("DataFrame 'samples' must contain column 'log_prob'")

    plt.figure(figsize=(12, 4))
    plt.plot(samples.index, samples['log_prob'], alpha=0.7, color='green')
    plt.xlabel('Indeks próbki (po burn-in)')
    plt.ylabel('Log-prawdopodobieństwo')
    plt.title('Ścieżka MCMC dla log-prawdopodobieństwa')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
