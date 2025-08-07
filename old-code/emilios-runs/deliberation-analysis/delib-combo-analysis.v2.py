import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_json_files(directory):
    """Load all JSON files from a directory and return combined data"""
    data = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return pd.DataFrame()
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    file_data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(file_data, list):
                        for item in file_data:
                            item['filename'] = filename
                            data.append(item)
                    else:
                        file_data['filename'] = filename
                        data.append(file_data)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    return pd.DataFrame(data)

def extract_allocation_data(df):
    """Extract and clean allocation data"""
    df_clean = df.copy()
    
    # Convert allocation columns to numeric
    allocation_cols = ['consumer_allocation', 'industrial_allocation']
    for col in allocation_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Calculate previous choice allocation (escalation measure)
    # This assumes the "previous choice" is reflected in the second allocation
    if 'first_choice' in df_clean.columns:
        # For high responsibility condition (has first_choice)
        df_clean['previous_choice_allocation'] = df_clean.apply(
            lambda row: row['consumer_allocation'] if row.get('first_choice') == 'consumer' 
            else row['industrial_allocation'] if row.get('first_choice') == 'industrial'
            else np.nan, axis=1
        )
    else:
        # For low responsibility condition (use product_choice)
        df_clean['previous_choice_allocation'] = df_clean.apply(
            lambda row: row['consumer_allocation'] if row.get('product_choice') == 'consumer'
            else row['industrial_allocation'] if row.get('product_choice') == 'industrial'
            else np.nan, axis=1
        )
    
    # Convert to millions for easier interpretation
    df_clean['previous_choice_allocation_M'] = df_clean['previous_choice_allocation'] / 1_000_000
    
    return df_clean

def perform_assumption_tests(df, dv_col, group_col):
    """Perform normality and homogeneity tests"""
    results = {}
    
    # Normality tests by group
    groups = df[group_col].unique()
    normality_results = {}
    
    for group in groups:
        group_data = df[df[group_col] == group][dv_col].dropna()
        if len(group_data) > 3:  # Need at least 3 observations for Shapiro-Wilk
            stat, p = shapiro(group_data)
            normality_results[group] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}
        else:
            normality_results[group] = {'statistic': np.nan, 'p_value': np.nan, 'normal': False}
    
    results['normality'] = normality_results
    
    # Homogeneity of variance test
    group_data_list = [df[df[group_col] == group][dv_col].dropna() for group in groups]
    group_data_list = [group for group in group_data_list if len(group) > 0]
    
    if len(group_data_list) >= 2:
        levene_stat, levene_p = levene(*group_data_list)
        results['homogeneity'] = {
            'statistic': levene_stat, 
            'p_value': levene_p, 
            'homogeneous': levene_p > 0.05
        }
    else:
        results['homogeneity'] = {'statistic': np.nan, 'p_value': np.nan, 'homogeneous': False}
    
    return results

def calculate_descriptive_stats(df, dv_col, group_col):
    """Calculate descriptive statistics by group"""
    stats_list = []
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][dv_col].dropna()
        
        if len(group_data) > 0:
            n = len(group_data)
            mean = group_data.mean()
            std = group_data.std()
            se = std / np.sqrt(n)
            ci_lower = mean - 1.96 * se
            ci_upper = mean + 1.96 * se
            
            stats_list.append({
                'Group': group,
                'N': n,
                'Mean': mean,
                'SD': std,
                'SE': se,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper
            })
        else:
            stats_list.append({
                'Group': group,
                'N': 0,
                'Mean': np.nan,
                'SD': np.nan,
                'SE': np.nan,
                'CI_Lower': np.nan,
                'CI_Upper': np.nan
            })
    
    return pd.DataFrame(stats_list)

def perform_statistical_tests(df, dv_col, group_col):
    """Perform appropriate statistical tests"""
    results = {}
    
    groups = df[group_col].unique()
    if len(groups) != 2:
        results['error'] = f"Expected 2 groups, found {len(groups)}"
        return results
    
    group1_data = df[df[group_col] == groups[0]][dv_col].dropna()
    group2_data = df[df[group_col] == groups[1]][dv_col].dropna()
    
    if len(group1_data) == 0 or len(group2_data) == 0:
        results['error'] = "One or both groups have no data"
        return results
    
    # Independent t-test
    t_stat, t_p = ttest_ind(group1_data, group2_data, equal_var=False)  # Welch's t-test
    results['t_test'] = {
        'statistic': t_stat,
        'p_value': t_p,
        'significant': t_p < 0.05,
        'group1_mean': group1_data.mean(),
        'group2_mean': group2_data.mean(),
        'mean_difference': group1_data.mean() - group2_data.mean()
    }
    
    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_p = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    results['mann_whitney'] = {
        'statistic': u_stat,
        'p_value': u_p,
        'significant': u_p < 0.05
    }
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                         (len(group2_data) - 1) * group2_data.var()) / 
                        (len(group1_data) + len(group2_data) - 2))
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
    results['effect_size'] = {
        'cohens_d': cohens_d,
        'magnitude': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    }
    
    return results

def create_visualizations(df, dv_col, group_col, output_dir='plots'):
    """Create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=group_col, y=dv_col)
    plt.title(f'Distribution of {dv_col} by {group_col}')
    plt.ylabel('Previous Choice Allocation (Millions $)')
    plt.xlabel('Collaboration Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_{group_col}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=group_col, y=dv_col)
    plt.title(f'Density Distribution of {dv_col} by {group_col}')
    plt.ylabel('Previous Choice Allocation (Millions $)')
    plt.xlabel('Collaboration Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/violinplot_{group_col}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Histogram by group
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    groups = df[group_col].unique()
    
    for i, group in enumerate(groups):
        group_data = df[df[group_col] == group][dv_col].dropna()
        axes[i].hist(group_data, bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{group} (n={len(group_data)})')
        axes[i].set_xlabel('Previous Choice Allocation (Millions $)')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(group_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {group_data.mean():.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/histograms_{group_col}.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(symm_df, asymm_df, output_file='escalation_analysis_report.txt'):
    """Generate comprehensive analysis report"""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ESCALATION OF COMMITMENT ANALYSIS - COLLABORATION TYPES COMPARISON\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Combine data for overall analysis
        symm_df['collaboration_type'] = 'symmetrical'
        asymm_df['collaboration_type'] = 'asymmetrical'
        combined_df = pd.concat([symm_df, asymm_df], ignore_index=True)
        
        f.write(f"Total subjects analyzed: {len(combined_df)}\n\n")
        
        # Sample sizes by condition
        f.write("=" * 80 + "\n")
        f.write("SAMPLE SIZES BY CONDITION\n")
        f.write("=" * 80 + "\n")
        
        condition_counts = combined_df.groupby(['collaboration_type']).size()
        f.write("Collaboration types:\n")
        for collab_type, count in condition_counts.items():
            f.write(f"  {collab_type}: {count}\n")
        f.write("\n")
        
        # Check for other conditions if they exist
        if 'user_condition' in combined_df.columns:
            condition_counts = combined_df['user_condition'].value_counts()
            f.write("Conditions found:\n")
            for condition, count in condition_counts.items():
                f.write(f"  {condition}: {count}\n")
            f.write("\n")
        
        if 'first_choice' in combined_df.columns or 'product_choice' in combined_df.columns:
            responsibility_col = 'first_choice' if 'first_choice' in combined_df.columns else 'product_choice'
            resp_counts = combined_df[responsibility_col].notna().sum()
            f.write(f"Records with decision data: {resp_counts}\n\n")
        
        # Descriptive Statistics
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE DESCRIPTIVE STATISTICS\n")
        f.write("=" * 80 + "\n")
        
        desc_stats = calculate_descriptive_stats(combined_df, 'previous_choice_allocation_M', 'collaboration_type')
        
        f.write(f"{'Condition':<35} {'N':<8} {'Mean ($M)':<12} {'SD ($M)':<10} {'SE ($M)':<10} {'95% CI':<20}\n")
        f.write("-" * 95 + "\n")
        
        for _, row in desc_stats.iterrows():
            if row['N'] > 0:
                ci_str = f"[{row['CI_Lower']:.2f}, {row['CI_Upper']:.2f}]"
                f.write(f"{row['Group']:<35} {row['N']:<8} {row['Mean']:<12.2f} {row['SD']:<10.2f} {row['SE']:<10.2f} {ci_str:<20}\n")
            else:
                f.write(f"{row['Group']:<35} {row['N']:<8} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<20}\n")
        
        f.write("\n")
        
        # Assumption Testing
        f.write("ASSUMPTION CHECKING:\n")
        f.write("-" * 40 + "\n")
        
        assumptions = perform_assumption_tests(combined_df, 'previous_choice_allocation_M', 'collaboration_type')
        
        f.write("Normality Tests (Shapiro-Wilk):\n")
        for group, result in assumptions['normality'].items():
            if not pd.isna(result['p_value']):
                symbol = "✓" if result['normal'] else "✗"
                normal_status = "Normal" if result['normal'] else "Non-normal"
                f.write(f"  {group}: p = {result['p_value']:.3f} ({symbol} {normal_status})\n")
            else:
                f.write(f"  {group}: Insufficient data for test\n")
        
        f.write(f"\nHomogeneity of Variance (Levene's test): ")
        if not pd.isna(assumptions['homogeneity']['p_value']):
            f.write(f"F = {assumptions['homogeneity']['statistic']:.3f}, p = {assumptions['homogeneity']['p_value']:.3f}\n")
            homo_symbol = "✓" if assumptions['homogeneity']['homogeneous'] else "✗"
            homo_status = "Homogeneous" if assumptions['homogeneity']['homogeneous'] else "Heterogeneous"
            f.write(f"  Result: {homo_symbol} {homo_status}\n\n")
        else:
            f.write("Could not compute\n\n")
        
        # Statistical Tests
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL COMPARISON\n")
        f.write("=" * 80 + "\n")
        
        test_results = perform_statistical_tests(combined_df, 'previous_choice_allocation_M', 'collaboration_type')
        
        if 'error' not in test_results:
            # T-test results
            t_result = test_results['t_test']
            f.write("Independent Samples t-test (Welch's):\n")
            f.write(f"  t({len(combined_df)-2}) = {t_result['statistic']:.3f}, p = {t_result['p_value']:.3f}\n")
            f.write(f"  Mean difference = {t_result['mean_difference']:.3f}\n")
            f.write(f"  Significant: {'Yes' if t_result['significant'] else 'No'} (α = 0.05)\n\n")
            
            # Mann-Whitney U test
            u_result = test_results['mann_whitney']
            f.write("Mann-Whitney U test (non-parametric):\n")
            f.write(f"  U = {u_result['statistic']:.3f}, p = {u_result['p_value']:.3f}\n")
            f.write(f"  Significant: {'Yes' if u_result['significant'] else 'No'} (α = 0.05)\n\n")
            
            # Effect size
            effect = test_results['effect_size'] 
            f.write("Effect Size:\n")
            f.write(f"  Cohen's d = {effect['cohens_d']:.3f} ({effect['magnitude']} effect)\n\n")
            
            # Interpretation
            f.write("INTERPRETATION:\n")
            f.write("-" * 40 + "\n")
            
            groups = combined_df['collaboration_type'].unique()
            group1_mean = t_result['group1_mean']
            group2_mean = t_result['group2_mean']
            
            if t_result['significant']:
                higher_group = groups[0] if group1_mean > group2_mean else groups[1]
                lower_group = groups[1] if group1_mean > group2_mean else groups[0]
                f.write(f"There is a statistically significant difference in escalation of commitment\n")
                f.write(f"between collaboration types. {higher_group.title()} collaboration shows\n")
                f.write(f"significantly higher escalation (M = {max(group1_mean, group2_mean):.2f}) than\n")
                f.write(f"{lower_group} collaboration (M = {min(group1_mean, group2_mean):.2f}).\n\n")
            else:
                f.write(f"There is no statistically significant difference in escalation of commitment\n")
                f.write(f"between symmetrical (M = {group1_mean:.2f}) and asymmetrical (M = {group2_mean:.2f})\n")
                f.write(f"collaboration types.\n\n")
        else:
            f.write(f"Error in statistical analysis: {test_results['error']}\n\n")
        
        # Additional Analysis
        f.write("=" * 80 + "\n")
        f.write("ADDITIONAL ANALYSIS\n") 
        f.write("=" * 80 + "\n")
        
        # Allocation patterns
        f.write("Allocation Patterns:\n")
        f.write("-" * 20 + "\n")
        
        for collab_type in combined_df['collaboration_type'].unique():
            subset = combined_df[combined_df['collaboration_type'] == collab_type]
            f.write(f"\n{collab_type.title()} Collaboration:\n")
            f.write(f"  Mean allocation to previous choice: ${subset['previous_choice_allocation_M'].mean():.2f}M\n")
            f.write(f"  Range: ${subset['previous_choice_allocation_M'].min():.2f}M - ${subset['previous_choice_allocation_M'].max():.2f}M\n")
            f.write(f"  Median: ${subset['previous_choice_allocation_M'].median():.2f}M\n")
            
            # Escalation threshold analysis (>10M = escalation)
            escalation_count = (subset['previous_choice_allocation_M'] > 10).sum()
            escalation_rate = escalation_count / len(subset) * 100
            f.write(f"  Escalation rate (>$10M): {escalation_count}/{len(subset)} ({escalation_rate:.1f}%)\n")

def main():
    """Main analysis function"""
    print("Starting Escalation of Commitment Analysis...")
    
    # Load data from both collaboration conditions
    print("Loading symmetrical collaboration data...")
    symm_df = load_json_files('/Users/leo/Documents/GitHub/escalation-commitment/symm_deliberation_runs')
    
    print("Loading asymmetrical collaboration data...")
    asymm_df = load_json_files('/Users/leo/Documents/GitHub/escalation-commitment/asymm_deliberation_runs')

    if symm_df.empty and asymm_df.empty:
        print("No data found in either directory. Please check file paths.")
        return
    
    # Process data
    print("Processing data...")
    if not symm_df.empty:
        symm_df = extract_allocation_data(symm_df)
    if not asymm_df.empty:
        asymm_df = extract_allocation_data(asymm_df)
    
    # Create visualizations
    print("Creating visualizations...")
    if not symm_df.empty and not asymm_df.empty:
        # Combine for visualization
        symm_df['collaboration_type'] = 'symmetrical'
        asymm_df['collaboration_type'] = 'asymmetrical'
        combined_df = pd.concat([symm_df, asymm_df], ignore_index=True)
        
        create_visualizations(combined_df, 'previous_choice_allocation_M', 'collaboration_type')
    
    # Generate report
    print("Generating analysis report...")
    generate_report(symm_df, asymm_df)
    
    print("Analysis complete! Check 'escalation_analysis_report.txt' for results.")
    print("Visualizations saved in 'plots/' directory.")

if __name__ == "__main__":
    main()