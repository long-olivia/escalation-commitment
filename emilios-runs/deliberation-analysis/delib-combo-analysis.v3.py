import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
import json

def comprehensive_escalation_analysis(asymm_folder_path, symm_folder_path):
    """
    Comprehensive escalation of commitment analysis for symmetrical vs asymmetrical collaboration
    """
    
    print("=" * 80)
    print("COMPREHENSIVE ESCALATION OF COMMITMENT ANALYSIS")
    print("=" * 80)
    print(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def load_folder_data(folder_path, collaboration_type):
        """Load all JSON files from a folder and add collaboration type"""
        all_files = glob.glob(os.path.join(folder_path, "*.json"))
        data_records = []
        
        for file in all_files:
            try:
                with open(file, 'r') as f:
                    json_data = json.load(f)
                
                # Handle both list and dict formats
                if isinstance(json_data, list):
                    if len(json_data) > 0:
                        json_data = json_data[0]  # Take first item if it's a list
                    else:
                        print(f"Warning: Empty list in {file}")
                        continue
                
                # Extract relevant data from JSON based on your actual structure
                record = {
                    'collaboration_type': collaboration_type,
                    'source_file': os.path.basename(file),
                    'n': json_data.get('n'),
                    'first_choice': json_data.get('first_choice'),
                    'first_reasoning': json_data.get('first_reasoning', ''),
                }
                
                # Convert investment amount based on your data structure
                # You'll need to map your actual data to investment amounts
                # For now, using 'n' as a proxy - adjust this based on your needs
                if record['n'] is not None:
                    record['investment_amount'] = float(record['n'])  # Adjust this mapping
                else:
                    print(f"Warning: No 'n' field found in {file}")
                    continue
                
                data_records.append(record)
                
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
        
        if data_records:
            return pd.DataFrame(data_records)
        else:
            return pd.DataFrame()
    
    # Load asymmetrical data
    print(f"Loading asymmetrical data from: {asymm_folder_path}")
    asymm_data = load_folder_data(asymm_folder_path, 'asymmetrical')
    
    # Load symmetrical data  
    print(f"Loading symmetrical data from: {symm_folder_path}")
    symm_data = load_folder_data(symm_folder_path, 'symmetrical')
    
    # Debug: Show JSON structure from first file
    sample_files = glob.glob(os.path.join(asymm_folder_path, "*.json"))
    if sample_files:
        print("\nDEBUG: Sample JSON structure from first file:")
        with open(sample_files[0], 'r') as f:
            sample_json = json.load(f)
        print(json.dumps(sample_json, indent=2)[:500] + "..." if len(str(sample_json)) > 500 else json.dumps(sample_json, indent=2))
    
    # Combine datasets
    df = pd.concat([asymm_data, symm_data], ignore_index=True)
    
    if df.empty:
        raise ValueError("No data found in specified folders!")
    
    # Debug: Show what columns we actually loaded
    print(f"\nDEBUG: Loaded columns: {df.columns.tolist()}")
    print("DEBUG: First few rows:")
    print(df.head())
    
    print(f"Loaded {len(asymm_data)} asymmetrical records")
    print(f"Loaded {len(symm_data)} symmetrical records")
    
    print(f"Total subjects analyzed: {len(df)}")
    print()
    
    # Check if we have the investment_amount column
    if 'investment_amount' not in df.columns:
        print("ERROR: No 'investment_amount' column found. Please check your data mapping.")
        return None
    
    # Remove any rows with missing investment amounts
    df = df.dropna(subset=['investment_amount'])
    
    # ============================================================================
    # CORE DESCRIPTIVE STATISTICS
    # ============================================================================
    print("=" * 80)
    print("CORE DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    # Sample sizes
    print("Sample Sizes by Condition:")
    print("-" * 40)
    sample_sizes = df.groupby('collaboration_type').size()
    for condition, n in sample_sizes.items():
        print(f"  {condition}: {n}")
    print(f"  Total: {len(df)}")
    print()
    
    # Central tendency and variability
    print("Descriptive Statistics:")
    print("-" * 80)
    print(f"{'Condition':<20} {'N':<8} {'Mean':<12} {'SD':<10} {'SE':<10} {'Median':<10} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    descriptives = {}
    for condition in df['collaboration_type'].unique():
        subset = df[df['collaboration_type'] == condition]['investment_amount']
        n = len(subset)
        mean = subset.mean()
        sd = subset.std()
        se = sd / np.sqrt(n)
        median = subset.median()
        min_val = subset.min()
        max_val = subset.max()
        
        descriptives[condition] = {
            'n': n, 'mean': mean, 'sd': sd, 'se': se,
            'median': median, 'min': min_val, 'max': max_val
        }
        
        print(f"{condition:<20} {n:<8} {mean:<12.2f} {sd:<10.2f} {se:<10.2f} {median:<10.2f} {min_val:<8.2f} {max_val:<8.2f}")
    print()
    
    # Confidence intervals
    print("95% Confidence Intervals:")
    print("-" * 40)
    for condition, stats_dict in descriptives.items():
        ci_lower = stats_dict['mean'] - 1.96 * stats_dict['se']
        ci_upper = stats_dict['mean'] + 1.96 * stats_dict['se']
        print(f"  {condition}: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print()
    
    # Distribution characteristics
    print("Distribution Characteristics:")
    print("-" * 40)
    for condition in df['collaboration_type'].unique():
        subset = df[df['collaboration_type'] == condition]['investment_amount']
        skewness = stats.skew(subset)
        kurt = stats.kurtosis(subset)
        print(f"  {condition}:")
        print(f"    Skewness: {skewness:.3f}")
        print(f"    Kurtosis: {kurt:.3f}")
        
        # Outliers (>3 SD from mean)
        outliers = subset[np.abs(subset - subset.mean()) > 3 * subset.std()]
        print(f"    Outliers (>3 SD): {len(outliers)}")
    print()
    
    # ============================================================================
    # STATISTICAL ASSUMPTIONS
    # ============================================================================
    print("=" * 80)
    print("STATISTICAL ASSUMPTIONS TESTING")
    print("=" * 80)
    
    # Normality tests
    print("Normality Tests (Shapiro-Wilk):")
    print("-" * 40)
    normality_results = {}
    for condition in df['collaboration_type'].unique():
        subset = df[df['collaboration_type'] == condition]['investment_amount']
        stat, p = shapiro(subset)
        is_normal = "✓" if p > 0.05 else "✗"
        normality_results[condition] = p > 0.05
        print(f"  {condition}: W = {stat:.4f}, p = {p:.4f} ({is_normal} {'Normal' if p > 0.05 else 'Non-normal'})")
    print()
    
    # Homogeneity of variance
    print("Homogeneity of Variance (Levene's Test):")
    print("-" * 40)
    groups = [df[df['collaboration_type'] == condition]['investment_amount'] for condition in df['collaboration_type'].unique()]
    levene_stat, levene_p = levene(*groups)
    is_homogeneous = "✓" if levene_p > 0.05 else "✗"
    print(f"  F = {levene_stat:.3f}, p = {levene_p:.4f}")
    print(f"  Result: {is_homogeneous} {'Homogeneous' if levene_p > 0.05 else 'Heterogeneous'} variances")
    print()
    
    # ============================================================================
    # HYPOTHESIS TESTING
    # ============================================================================
    print("=" * 80)
    print("HYPOTHESIS TESTING")
    print("=" * 80)
    
    # Main effect test (Independent samples t-test)
    print("Main Effect: Collaboration Type Comparison")
    print("-" * 50)
    
    group1_name, group2_name = df['collaboration_type'].unique()
    group1_data = df[df['collaboration_type'] == group1_name]['investment_amount']
    group2_data = df[df['collaboration_type'] == group2_name]['investment_amount']
    
    # Regular t-test
    t_stat, t_p = stats.ttest_ind(group1_data, group2_data)
    
    # Welch's t-test (unequal variances)
    welch_t, welch_p = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_sd = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                        (len(group2_data) - 1) * group2_data.var()) / 
                       (len(group1_data) + len(group2_data) - 2))
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_sd
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    print(f"Independent Samples t-test:")
    print(f"  t({len(group1_data) + len(group2_data) - 2}) = {t_stat:.3f}, p = {t_p:.4f}")
    print(f"Welch's t-test (unequal variances):")
    print(f"  t = {welch_t:.3f}, p = {welch_p:.4f}")
    print(f"Mean difference: {group1_data.mean() - group2_data.mean():.3f}")
    print(f"Cohen's d = {cohens_d:.3f} ({effect_size_interp} effect)")
    print(f"Significant: {'Yes' if welch_p < 0.05 else 'No'} (α = 0.05)")
    print()
    
    # Non-parametric alternative
    print("Non-parametric Test (Mann-Whitney U):")
    print("-" * 40)
    u_stat, u_p = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    print(f"  U = {u_stat:.1f}, p = {u_p:.4f}")
    print(f"  Significant: {'Yes' if u_p < 0.05 else 'No'} (α = 0.05)")
    print()
    
    # ============================================================================
    # PUBLICATION SUMMARY
    # ============================================================================
    print("=" * 80)
    print("SUMMARY FOR PUBLICATION")
    print("=" * 80)
    
    print("Key Statistical Results:")
    print("-" * 30)
    print(f"• Sample: N = {len(df)} ({descriptives[group1_name]['n']} {group1_name}, {descriptives[group2_name]['n']} {group2_name})")
    print(f"• {group1_name}: M = {descriptives[group1_name]['mean']:.2f}, SD = {descriptives[group1_name]['sd']:.2f}")
    print(f"• {group2_name}: M = {descriptives[group2_name]['mean']:.2f}, SD = {descriptives[group2_name]['sd']:.2f}")
    print(f"• Welch's t-test: t = {welch_t:.3f}, p = {welch_p:.4f}")
    print(f"• Effect size: Cohen's d = {cohens_d:.3f} ({effect_size_interp})")
    print(f"• Non-parametric: U = {u_stat:.1f}, p = {u_p:.4f}")
    
    if not all(normality_results.values()) or levene_p <= 0.05:
        print(f"• Note: Statistical assumptions violated - report robust tests")
    
    print()
    print("Analysis completed successfully!")
    
    return {
        'descriptives': descriptives,
        'main_effect': {'t': welch_t, 'p': welch_p, 'd': cohens_d},
        'nonparametric': {'U': u_stat, 'p': u_p},
        'assumptions': {'normality': normality_results, 'homogeneity': levene_p > 0.05}
    }

# Usage example:
if __name__ == "__main__":
    # Your actual folder paths
    asymm_folder = "/Users/leo/Documents/GitHub/escalation-commitment/asymm_deliberation_runs"
    symm_folder = "/Users/leo/Documents/GitHub/escalation-commitment/symm_deliberation_runs"
    
    results = comprehensive_escalation_analysis(asymm_folder, symm_folder)