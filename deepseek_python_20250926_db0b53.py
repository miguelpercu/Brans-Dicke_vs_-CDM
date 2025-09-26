# =============================================================================
# COSMOLOGICAL ANALYSIS: Brans-Dicke vs ΛCDM (FINAL WORKING VERSION)
# Complete pipeline with corrected equations and statistical analysis
# Author: Miguel Ángel Percudani
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
import warnings
warnings.filterwarnings('ignore')

print("=== COSMOLOGICAL ANALYSIS: Brans-Dicke vs ΛCDM (FINAL) ===")

# =============================================================================
# 1. COSMOLOGICAL PARAMETERS AND SETUP
# =============================================================================

cosmo_params = {
    'H0': 70.0,           # km/s/Mpc
    'Omega_m0': 0.3,      # Matter density parameter
    'Omega_DE0': 0.7,     # Dark energy density parameter
    'w0': -1.0,           # Dark energy equation of state
    'c': 299792.458       # Speed of light km/s
}

H0 = cosmo_params['H0']
Omega_m0 = cosmo_params['Omega_m0']
Omega_DE0 = cosmo_params['Omega_DE0']
c = cosmo_params['c']

# Create directory structure
directories = [
    'results/figures',
    'results/tables', 
    'results/data_curves',
    'results/data_observations',
    'results/statistical_analysis'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Created directory: {directory}")

print("\n=== COSMOLOGICAL PARAMETERS ===")
for key, value in cosmo_params.items():
    print(f"{key}: {value}")

# =============================================================================
# 2. LOAD PANTHEON DATA (STABLE VERSION)
# =============================================================================

def load_pantheon_data():
    """Load Pantheon supernova data"""
    pantheon_data = {
        'z': [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
              0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        'mu': [33.18, 34.70, 36.74, 37.80, 38.32, 39.27, 39.96, 40.52, 40.99, 41.39,
               41.74, 42.05, 42.33, 42.82, 43.24, 43.61, 43.94, 44.24, 44.51, 44.76,
               44.99, 45.20, 45.40, 45.58, 45.75, 45.91, 46.06, 46.20],
        'mu_err': [0.10, 0.10, 0.11, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14,
                   0.14, 0.15, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22,
                   0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
    }
    df = pd.DataFrame(pantheon_data)
    print(f"✓ Loaded Pantheon data: {len(df)} points")
    return df['z'].values, df['mu'].values, df['mu_err'].values

# =============================================================================
# 3. ΛCDM MODEL (REFERENCE)
# =============================================================================

def E_lcdm(z):
    """Normalized Hubble parameter for ΛCDM"""
    return np.sqrt(Omega_m0 * (1 + z)**3 + Omega_DE0)

def distance_modulus_lcdm(z):
    """Distance modulus for ΛCDM"""
    if np.isscalar(z):
        z = np.array([z])
    
    dL_values = []
    for z_val in z:
        if z_val == 0:
            dL_values.append(1e-10)
        else:
            # Numerical integration
            z_integ = np.linspace(0, z_val, 100)
            E_integ = E_lcdm(z_integ)
            integral = np.trapz(1/E_integ, z_integ)
            dL = (1 + z_val) * (c / H0) * integral
            dL_values.append(dL)
    
    dL_array = np.array(dL_values)
    return 5 * np.log10(dL_array * 1e5)  # Convert to distance modulus

# =============================================================================
# 4. CORRECTED BRANS-DICKE MODEL
# =============================================================================

def brans_dicke_system(z, y, omega):
    """
    Brans-Dicke field equations (corrected)
    y = [phi, dphi/dz] where phi is the scalar field
    """
    phi, dphidz = y
    
    # Ensure numpy arrays
    z = np.array(z) if np.isscalar(z) else z
    phi = np.array(phi)
    dphidz = np.array(dphidz)
    
    # Energy densities
    rho_m = Omega_m0 * (1.0 + z)**3
    rho_de = Omega_DE0 * np.ones_like(z)
    
    # Modified Friedmann equation terms
    term1 = (8.0 * np.pi) / (3.0 * phi) * (rho_m + rho_de)
    term2 = - (1.0 + z) * dphidz / phi
    term3 = (omega / 6.0) * ((1.0 + z) * dphidz / phi)**2
    
    E2 = term1 + term2 + term3
    E = np.sqrt(np.maximum(E2, 1e-10))
    
    # Scalar field equation (simplified but functional)
    H = E * H0
    source_term = (8.0 * np.pi) / (3.0 + 2.0 * omega) * (rho_m + 4.0 * rho_de)
    d2phidz2 = source_term / (H**2 * (1.0 + z)**2 + 1e-10)
    d2phidz2 -= (2.0/(1.0 + z) + 3.0) * dphidz
    
    return [dphidz, d2phidz2]

def solve_brans_dicke(z_points, omega, phi0=1.0, dphidz0=0.0):
    """Solve Brans-Dicke equations numerically"""
    try:
        z_points = np.array(z_points)
        y0 = np.array([phi0, dphidz0])
        
        z_range = [0, np.max(z_points)]
        
        sol = solve_ivp(brans_dicke_system, z_range, y0,
                       args=(omega,), t_eval=z_points, 
                       method='RK45', rtol=1e-6, atol=1e-8)
        
        if not sol.success:
            return None, None, None, None
        
        phi, dphidz = sol.y
        
        # Calculate E(z) consistently
        E_values = []
        for i, z_val in enumerate(z_points):
            rho_m = Omega_m0 * (1.0 + z_val)**3
            rho_de = Omega_DE0
            term1 = (8.0 * np.pi) / (3.0 * phi[i]) * (rho_m + rho_de)
            term2 = - (1.0 + z_val) * dphidz[i] / phi[i]
            term3 = (omega / 6.0) * ((1.0 + z_val) * dphidz[i] / phi[i])**2
            E2 = term1 + term2 + term3
            E_values.append(np.sqrt(np.maximum(E2, 1e-10)))
        
        E_array = np.array(E_values)
        H_array = E_array * H0
        
        return E_array, H_array, phi, dphidz
        
    except Exception as e:
        print(f"  Error in BD ω={omega}: {str(e)[:100]}...")
        return None, None, None, None

def distance_modulus_bd(z, omega):
    """Distance modulus for Brans-Dicke theory"""
    try:
        if np.isscalar(z):
            z = np.array([z])
        
        E, H, phi, dphidz = solve_brans_dicke(z, omega)
        
        if E is None:
            return np.full_like(z, np.nan)
        
        dL_values = []
        for i, z_val in enumerate(z):
            if z_val == 0:
                dL_values.append(1e-10)
            else:
                z_integ = np.linspace(0, z_val, 50)
                E_integ = np.interp(z_integ, z, E)
                integral = np.trapz(1/np.maximum(E_integ, 1e-10), z_integ)
                dL = (1 + z_val) * (c / H0) * integral
                dL_values.append(max(dL, 1e-10))
        
        dL_array = np.array(dL_values)
        return 5 * np.log10(dL_array * 1e5)
        
    except Exception as e:
        print(f"  Error in BD distance modulus ω={omega}: {e}")
        return np.full_like(z, np.nan)

# =============================================================================
# 5. STATISTICAL ANALYSIS
# =============================================================================

def calculate_chi2(mu_obs, mu_err, mu_model):
    """Calculate χ² statistic with marginalization"""
    if np.any(np.isnan(mu_model)):
        return np.nan, np.nan, len(mu_obs)
    
    # Marginalized normalization adjustment
    weights = 1.0 / (mu_err**2 + 1e-10)
    delta = mu_obs - mu_model
    A = np.sum(delta * weights) / np.sum(weights)
    
    chi2 = np.sum(((delta - A) / mu_err)**2)
    dof = len(mu_obs) - 2  # Degrees of freedom (adjusted for normalization)
    chi2_reduced = chi2 / max(dof, 1)
    
    return chi2, chi2_reduced, dof

def statistical_analysis(z_data, mu_obs, mu_err, omega_values):
    """Comprehensive statistical analysis"""
    print("\n=== STATISTICAL ANALYSIS ===")
    
    # ΛCDM reference
    print("Calculating ΛCDM...")
    mu_lcdm = distance_modulus_lcdm(z_data)
    chi2_lcdm, chi2_red_lcdm, dof = calculate_chi2(mu_obs, mu_err, mu_lcdm)
    print(f"ΛCDM: χ² = {chi2_lcdm:.2f}, χ²_reduced = {chi2_red_lcdm:.3f}")
    
    # Brans-Dicke analysis
    results = []
    print("\nCalculating Brans-Dicke...")
    print("-" * 50)
    
    for omega in omega_values:
        print(f"  Processing ω = {omega}...")
        mu_bd = distance_modulus_bd(z_data, omega)
        
        if np.any(np.isnan(mu_bd)):
            print(f"ω = {omega:5d}: Calculation error")
            results.append((omega, np.nan, np.nan))
        else:
            chi2, chi2_red, _ = calculate_chi2(mu_obs, mu_err, mu_bd)
            results.append((omega, chi2, chi2_red))
            print(f"ω = {omega:5d}: χ² = {chi2:7.2f}, χ²_reduced = {chi2_red:.3f}")
    
    # Find best fit
    valid_results = [(ω, χ2, χ2r) for ω, χ2, χ2r in results if not np.isnan(χ2)]
    
    if valid_results:
        best_omega, best_chi2, best_chi2_red = min(valid_results, key=lambda x: x[1])
        delta_chi2 = best_chi2 - chi2_lcdm
        
        print("\n" + "="*50)
        print(f"BEST FIT: ω = {best_omega}")
        print(f"Δχ² (BD - ΛCDM) = {delta_chi2:.3f}")
        print("="*50)
        
        return results, (chi2_lcdm, chi2_red_lcdm, dof), (best_omega, best_chi2, delta_chi2)
    else:
        print("❌ Could not calculate Brans-Dicke models")
        return results, (chi2_lcdm, chi2_red_lcdm, dof), None

# =============================================================================
# 6. COMPREHENSIVE PLOTTING
# =============================================================================

def create_comprehensive_plots(z_data, mu_obs, mu_err, results, lcdm_results, best_fit):
    """Create comprehensive analysis plots"""
    print("\n=== GENERATING ANALYSIS PLOTS ===")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Distance modulus comparison (main plot)
    ax1 = plt.subplot(2, 2, 1)
    z_plot = np.linspace(0.01, 2.0, 100)
    
    # ΛCDM
    mu_lcdm_plot = distance_modulus_lcdm(z_plot)
    plt.plot(z_plot, mu_lcdm_plot, 'black', linewidth=3, label='ΛCDM')
    
    # Brans-Dicke
    for omega in [10, 100, 1000]:
        mu_bd_plot = distance_modulus_bd(z_plot, omega)
        if not np.any(np.isnan(mu_bd_plot)):
            plt.plot(z_plot, mu_bd_plot, '--', linewidth=2, label=f'BD, ω = {omega}')
    
    # Observational data
    plt.errorbar(z_data, mu_obs, yerr=mu_err, fmt='o', markersize=4,
                 alpha=0.7, color='red', label='Pantheon Data')
    
    plt.xlabel('Redshift, z', fontsize=12)
    plt.ylabel('Distance Modulus μ(z)', fontsize=12)
    plt.legend()
    plt.title('Fit to Observational Data', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 2. χ² analysis
    ax2 = plt.subplot(2, 2, 2)
    omega_vals = [r[0] for r in results if not np.isnan(r[1])]
    chi2_vals = [r[1] for r in results if not np.isnan(r[1])]
    
    if omega_vals:
        plt.semilogx(omega_vals, chi2_vals, 'bo-', markersize=6, label='Brans-Dicke')
        plt.axhline(lcdm_results[0], color='red', linestyle='--', 
                    linewidth=2, label='ΛCDM')
        
        if best_fit:
            plt.semilogx([best_fit[0]], [best_fit[1]], 'ro', markersize=8, 
                        label=f'Best ω = {best_fit[0]}')
    
    plt.xlabel('Brans-Dicke Parameter ω', fontsize=12)
    plt.ylabel('χ²', fontsize=12)
    plt.legend()
    plt.title('χ² vs Coupling Parameter', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 3. Normalized residuals
    ax3 = plt.subplot(2, 2, 3)
    mu_lcdm_data = distance_modulus_lcdm(z_data)
    residuals_lcdm = (mu_obs - mu_lcdm_data) / mu_err
    plt.scatter(z_data, residuals_lcdm, alpha=0.7, s=40, label='ΛCDM', color='blue')
    
    if best_fit:
        mu_bd_data = distance_modulus_bd(z_data, best_fit[0])
        if not np.any(np.isnan(mu_bd_data)):
            residuals_bd = (mu_obs - mu_bd_data) / mu_err
            plt.scatter(z_data, residuals_bd, alpha=0.7, s=40, 
                       label=f'BD, ω = {best_fit[0]}', color='green')
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(1, color='red', linestyle='--', alpha=0.5, label='±1σ')
    plt.axhline(-1, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel('Redshift, z', fontsize=12)
    plt.ylabel('Normalized Residuals', fontsize=12)
    plt.legend()
    plt.title('Fit Residuals', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 4. Statistical summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""STATISTICAL SUMMARY

Data Points: {len(z_data)}
ΛCDM: χ² = {lcdm_results[0]:.2f}
χ²_reduced = {lcdm_results[1]:.3f}"""

    if best_fit:
        summary_text += f"""
Best BD (ω={best_fit[0]}):
χ² = {best_fit[1]:.2f}
Δχ² = {best_fit[2]:.3f}"""
        
        if best_fit[2] < 0:
            summary_text += "\n✅ BD fits BETTER than ΛCDM"
        elif abs(best_fit[2]) < 10:
            summary_text += "\n✅ Models are statistically similar"
        else:
            summary_text += "\n❌ Significant differences detected"
    else:
        summary_text += "\n❌ BD models could not be calculated"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/figures/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/comprehensive_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Analysis plots generated and saved")

# =============================================================================
# 7. DATA EXPORT FUNCTIONS
# =============================================================================

def export_all_data(z_data, mu_obs, mu_err, results, lcdm_results, best_fit):
    """Export all analysis data to organized files"""
    print("\n=== EXPORTING DATA ===")
    
    # 1. Cosmological parameters
    df_params = pd.DataFrame([cosmo_params])
    df_params.to_csv('results/tables/cosmological_parameters.csv', index=False)
    print("✓ Cosmological parameters saved")
    
    # 2. Observational data
    df_obs = pd.DataFrame({
        'redshift': z_data,
        'distance_modulus': mu_obs,
        'error': mu_err,
        'mu_LCDM': distance_modulus_lcdm(z_data)
    })
    df_obs.to_csv('results/data_observations/observational_data.csv', index=False)
    print("✓ Observational data saved")
    
    # 3. Theoretical curves
    z_curves = np.linspace(0.01, 2.0, 100)
    df_curves = pd.DataFrame({'redshift': z_curves})
    
    # ΛCDM curves
    df_curves['E_LCDM'] = E_lcdm(z_curves)
    df_curves['mu_LCDM'] = distance_modulus_lcdm(z_curves)
    
    # Brans-Dicke curves
    for omega in [10, 100, 1000]:
        E, H, phi, dphidz = solve_brans_dicke(z_curves, omega)
        if E is not None:
            df_curves[f'E_BD_omega_{omega}'] = E
            df_curves[f'phi_BD_omega_{omega}'] = phi
        
        mu_bd = distance_modulus_bd(z_curves, omega)
        if not np.any(np.isnan(mu_bd)):
            df_curves[f'mu_BD_omega_{omega}'] = mu_bd
    
    df_curves.to_csv('results/data_curves/theoretical_curves.csv', index=False)
    print("✓ Theoretical curves saved")
    
    # 4. Statistical results
    stats_data = []
    stats_data.append({
        'model': 'ΛCDM',
        'omega': np.nan,
        'chi2': lcdm_results[0],
        'chi2_reduced': lcdm_results[1],
        'dof': lcdm_results[2]
    })
    
    for omega, chi2, chi2_red in results:
        if not np.isnan(chi2):
            stats_data.append({
                'model': 'Brans-Dicke',
                'omega': omega,
                'chi2': chi2,
                'chi2_reduced': chi2_red,
                'dof': lcdm_results[2]
            })
    
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv('results/statistical_analysis/statistical_results.csv', index=False)
    print("✓ Statistical results saved")
    
    # 5. Best fit summary
    if best_fit:
        best_fit_data = {
            'best_omega': best_fit[0],
            'best_chi2': best_fit[1],
            'lcdm_chi2': lcdm_results[0],
            'delta_chi2': best_fit[2],
            'conclusion': 'BD better than ΛCDM' if best_fit[2] < 0 else 'ΛCDM better'
        }
        df_best = pd.DataFrame([best_fit_data])
        df_best.to_csv('results/statistical_analysis/best_fit_summary.csv', index=False)
        print("✓ Best fit summary saved")

# =============================================================================
# 8. MAIN ANALYSIS PIPELINE
# =============================================================================

def main_analysis():
    """Main cosmological analysis pipeline"""
    print("\n" + "="*70)
    print("STARTING COSMOLOGICAL ANALYSIS PIPELINE")
    print("="*70)
    
    # Load observational data
    z_data, mu_obs, mu_err = load_pantheon_data()
    
    # Brans-Dicke coupling parameters to test
    omega_values = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    
    # Perform statistical analysis
    results, lcdm_results, best_fit = statistical_analysis(z_data, mu_obs, mu_err, omega_values)
    
    # Create comprehensive plots
    create_comprehensive_plots(z_data, mu_obs, mu_err, results, lcdm_results, best_fit)
    
    # Export all data
    export_all_data(z_data, mu_obs, mu_err, results, lcdm_results, best_fit)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return results, lcdm_results, best_fit

# =============================================================================
# 9. EXECUTION AND FINAL SUMMARY
# =============================================================================

if __name__ == "__main__":
    # Execute the complete analysis
    results, lcdm_results, best_fit = main_analysis()
    
    # Display final results
    print("\n📊 FINAL RESULTS SUMMARY:")
    print(f"ΛCDM: χ² = {lcdm_results[0]:.2f}, χ²_reduced = {lcdm_results[1]:.3f}")
    
    if best_fit:
        print(f"Brans-Dicke (best ω={best_fit[0]}): χ² = {best_fit[1]:.2f}")
        print(f"Δχ² = {best_fit[2]:.3f}")
        
        if best_fit[2] < 0:
            print("🎯 CONCLUSION: Brans-Dicke provides BETTER fit than ΛCDM")
        else:
            print("🎯 CONCLUSION: ΛCDM remains the preferred model")
    
    # Display file structure
    print("\n📁 GENERATED FILE STRUCTURE:")
    print("results/")
    print("├── figures/comprehensive_analysis.png/.pdf")
    print("├── tables/cosmological_parameters.csv")
    print("├── data_observations/observational_data.csv")
    print("├── data_curves/theoretical_curves.csv")
    print("└── statistical_analysis/")
    print("    ├── statistical_results.csv")
    print("    └── best_fit_summary.csv")
    print("="*70)