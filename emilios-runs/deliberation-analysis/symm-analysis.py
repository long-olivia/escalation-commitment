import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from datetime import datetime
import glob

def load_experimental_data(data_directory="/Users/leo/Documents/GitHub/escalation-commitment/symm_deliberation_runs"):
    """
    Load all experimental data from JSON files
    """
    data = []
    
    # Load high responsibility data
    high_pos_files = glob.glob(f"{data_directory}/high_positive_*.json")
    high_neg_files = glob.glob(f"{data_directory}/high_negative_*.json")
    
    # Load low responsibility data  
    low_consumer_pos_files = glob.glob(f"{data_directory}/low_consumer_positive_*.json")
    low_consumer_neg_files = glob.glob(f"{data_directory}/low_consumer_negative_*.json")
    low_industrial_pos_files = glob.glob(f"{data_directory}/low_industrial_positive_*.json")
    low_industrial_neg_files = glob.glob(f"{data_directory}/low_industrial_negative_*.json")
    
    # Process high responsibility files
    for file_path in high_pos_files:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            for entry in file_data:
                data.append({
                    'condition': 'high_positive',
                    'responsibility': 'high',
                    'consequences': 'positive',
                    'first_choice': entry.get('first_choice', ''),
                    'consumer_allocation': float(entry.get('consumer_allocation', 0)),
                    'industrial_allocation': float(entry.get('industrial_allocation', 0)),
                    'total_allocation': float(entry.get('consumer_allocation', 0)) + float(entry.get('industrial_allocation', 0)),
                    'first_reasoning': entry.get('first_reasoning', ''),
                    'second_reasoning': entry.get('second_reasoning', ''),
                    'file_source': file_path
                })
    
    for file_path in high_neg_files:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            for entry in file_data:
                data.append({
                    'condition': 'high_negative',
                    'responsibility': 'high',
                    'consequences': 'negative',
                    'first_choice': entry.get('first_choice', ''),
                    'consumer_allocation': float(entry.get('consumer_allocation', 0)),
                    'industrial_allocation': float(entry.get('industrial_allocation', 0)),
                    'total_allocation': float(entry.get('consumer_allocation', 0)) + float(entry.get('industrial_allocation', 0)),
                    'first_reasoning': entry.get('first_reasoning', ''),
                    'second_reasoning': entry.get('second_reasoning', ''),
                    'file_source': file_path
                })
    
    # Process low responsibility files
    for file_path in low_consumer_pos_files + low_industrial_pos_files:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            for entry in file_data:
                data.append({
                    'condition': 'low_positive',
                    'responsibility': 'low',
                    'consequences': 'positive',
                    'first_choice': entry.get('product_choice', ''),
                    'consumer_allocation': float(entry.get('consumer_allocation', 0)),
                    'industrial_allocation': float(entry.get('industrial_allocation', 0)),
                    'total_allocation': float(entry.get('consumer_allocation', 0)) + float(entry.get('industrial_allocation', 0)),
                    'reasoning': entry.get('reasoning', ''),
                    'file_source': file_path
                })
    
    for file_path in low_consumer_neg_files + low_industrial_neg_files:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            for entry in file_data:
                data.append({
                    'condition': 'low_negative',
                    'responsibility': 'low',
                    'consequences': 'negative',
                    'first_choice': entry.get('product_choice', ''),
                    'consumer_allocation': float(entry.get('consumer_allocation', 0)),
                    'industrial_allocation': float(entry.get('industrial_allocation', 0)),
                    'total_allocation': float(entry.get('consumer_allocation', 0)) + float(entry.get('industrial_allocation', 0)),
                    'reasoning': entry.get('reasoning', ''),
                    'file_source': file_path
                })
    
    return pd.DataFrame(data)

def calculate_escalation_allocation(row):
    """
    Calculate allocation to the previously chosen division (escalation measure)
    """
    first_choice = row['first_choice'].lower()
    if first_choice == 'consumer':
        return row['consumer_allocation']
    elif first_choice == 'industrial':
        return row['industrial_allocation']
    else:
        # For low responsibility, they didn't make first choice, use the division they were told received funding
        if 'consumer' in row['file_source'].lower():
            return row['consumer_allocation']
        elif 'industrial' in row['file_source'].lower():
            return row['industrial_allocation']
        else:
            return np.nan

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def run_analysis(data_directory="/Users/leo/Documents/GitHub/escalation-commitment/symm_deliberation_runs"):
    """
    Main analysis function replicating Staw (1976) methodology
    """
    
    print("=" * 80)
    print("ESCALATION OF COMMITMENT ANALYSIS - STAW (1976) REPLICATION")
    print("=" * 80)
    print(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_experimental_data(data_directory)
    
    if df.empty:
        print("ERROR: No data files found. Please check the data directory path.")
        return None
    
    # Calculate escalation allocation (allocation to previously chosen division)
    df['escalation_allocation'] = df.apply(calculate_escalation_allocation, axis=1)
    
    # Convert to millions for easier interpretation
    df['escalation_allocation_M'] = df['escalation_allocation'] / 1_000_000
    df['consumer_allocation_M'] = df['consumer_allocation'] / 1_000_000
    df['industrial_allocation_M'] = df['industrial_allocation'] / 1_000_000
    
    print(f"Total subjects analyzed: {len(df)}")
    
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    # Group by condition and calculate descriptive statistics
    descriptive_stats = df.groupby('condition')['escalation_allocation_M'].agg(['count', 'mean', 'std', 'sem']).round(2)
    
    print(f"{'Condition':<35} {'N':<8} {'Mean ($M)':<12} {'SD ($M)':<12} {'SE ($M)':<12}")
    print("-" * 80)
    for condition, stats in descriptive_stats.iterrows():
        condition_name = condition.replace('_', ' ').title()
        print(f"{condition_name:<35} {stats['count']:<8} {stats['mean']:<12} {stats['std']:<12} {stats['sem']:<12}")
    
    print("\n" + "=" * 80)
    print("PRELIMINARY ANALYSIS")
    print("=" * 80)
    
    # Check division choices in high responsibility conditions
    high_resp_data = df[df['responsibility'] == 'high']
    if not high_resp_data.empty:
        choice_counts = high_resp_data['first_choice'].value_counts()
        print("Division choices in high responsibility conditions:")
        print(f"  Consumer: {choice_counts.get('consumer', 0)}, Industrial: {choice_counts.get('industrial', 0)}")
        print("  (Allowing for 2x2 analysis)")
    
    print("\n" + "=" * 80)
    print("MAIN EFFECTS ANALYSIS")
    print("=" * 80)
    
    # Main effect of personal responsibility
    high_resp = df[df['responsibility'] == 'high']['escalation_allocation_M']
    low_resp = df[df['responsibility'] == 'low']['escalation_allocation_M']
    
    if len(high_resp) > 0 and len(low_resp) > 0:
        t_stat_resp, p_val_resp = ttest_ind(high_resp, low_resp)
        
        print("MAIN EFFECT OF PERSONAL RESPONSIBILITY:")
        print(f"  High Personal Responsibility: M = ${high_resp.mean():.2f}M (n = {len(high_resp)})")
        print(f"  Low Personal Responsibility:  M = ${low_resp.mean():.2f}M (n = {len(low_resp)})")
        print(f"  Difference: ${high_resp.mean() - low_resp.mean():.2f}M")
        print(f"  t({len(high_resp) + len(low_resp) - 2}) = {t_stat_resp:.3f}, p = {p_val_resp:.3f}")
        print(f"  Result: {'✓ Significant' if p_val_resp < 0.05 else '✗ Not significant'}")
        print(f"  Original study: High (11.08M) vs Low (8.89M)")
    else:
        print("MAIN EFFECT OF PERSONAL RESPONSIBILITY:")
        print(f"  High Personal Responsibility: M = ${high_resp.mean():.2f}M (n = {len(high_resp)})")
        print(f"  Low Personal Responsibility:  M = ${low_resp.mean():.2f}M (n = {len(low_resp)})")
        print("  ⚠️  Cannot compute t-test: Missing data for one or both conditions")
        t_stat_resp, p_val_resp = np.nan, np.nan
    
    # Main effect of decision consequences
    positive_conseq = df[df['consequences'] == 'positive']['escalation_allocation_M']
    negative_conseq = df[df['consequences'] == 'negative']['escalation_allocation_M']
    
    if len(positive_conseq) > 0 and len(negative_conseq) > 0:
        t_stat_conseq, p_val_conseq = ttest_ind(positive_conseq, negative_conseq)
        
        print(f"\nMAIN EFFECT OF DECISION CONSEQUENCES:")
        print(f"  Positive Consequences: M = ${positive_conseq.mean():.2f}M (n = {len(positive_conseq)})")
        print(f"  Negative Consequences: M = ${negative_conseq.mean():.2f}M (n = {len(negative_conseq)})")
        print(f"  Difference: ${positive_conseq.mean() - negative_conseq.mean():.2f}M")
        print(f"  t({len(positive_conseq) + len(negative_conseq) - 2}) = {t_stat_conseq:.3f}, p = {p_val_conseq:.3f}")
        print(f"  Result: {'✓ Significant main effect' if p_val_conseq < 0.05 else '✗ Not significant'}")
        print(f"  Original study: Negative (11.20M) vs Positive (8.77M)")
    else:
        print(f"\nMAIN EFFECT OF DECISION CONSEQUENCES:")
        print(f"  Positive Consequences: M = ${positive_conseq.mean():.2f}M (n = {len(positive_conseq)})")
        print(f"  Negative Consequences: M = ${negative_conseq.mean():.2f}M (n = {len(negative_conseq)})")
        print("  ⚠️  Cannot compute t-test: Missing data for one or both conditions")
        t_stat_conseq, p_val_conseq = np.nan, np.nan
    
    print("\n" + "=" * 80)
    print("INTERACTION OF PERSONAL RESPONSIBILITY AND DECISION CONSEQUENCES")
    print("=" * 80)
    
    # Calculate cell means
    cell_means = df.groupby(['responsibility', 'consequences'])['escalation_allocation_M'].agg(['mean', 'count']).round(2)
    
    print("CELL MEANS:")
    for (resp, conseq), stats in cell_means.iterrows():
        condition_name = f"{resp.title()} Responsibility + {conseq.title()}"
        print(f"  {condition_name:<30} : M = ${stats['mean']:.2f}M (n = {stats['count']})")
    
    # Critical interaction test: High Responsibility + Negative vs all others
    hr_neg = df[(df['responsibility'] == 'high') & (df['consequences'] == 'negative')]['escalation_allocation_M']
    all_others = df[~((df['responsibility'] == 'high') & (df['consequences'] == 'negative'))]['escalation_allocation_M']
    
    if len(hr_neg) > 0 and len(all_others) > 0:
        t_stat_int, p_val_int = ttest_ind(hr_neg, all_others)
        effect_size_int = cohens_d(hr_neg, all_others)
        
        print(f"\nCRITICAL INTERACTION TEST:")
        print("High Personal Responsibility + Negative Consequences vs. All Other Conditions")
        print(f"  High Responsibility + Negative: M = ${hr_neg.mean():.2f}M (n = {len(hr_neg)})")
        print(f"  All Other Conditions:           M = ${all_others.mean():.2f}M (n = {len(all_others)})")
        print(f"  Difference: ${hr_neg.mean() - all_others.mean():.2f}M")
        print(f"  t(df = {len(hr_neg) + len(all_others) - 2}) = {t_stat_int:.3f}, p = {p_val_int:.3f}")
        print(f"  Result: {'✓ Significant interaction effect' if p_val_int < 0.05 else '✗ Not significant'}")
        print(f"  Original study: High Resp + Negative (13.07M) significantly > all others")
        print(f"  Effect size (Cohen's d): {effect_size_int:.3f}")
    else:
        print(f"\nCRITICAL INTERACTION TEST:")
        print("High Personal Responsibility + Negative Consequences vs. All Other Conditions")
        print(f"  High Responsibility + Negative: M = ${hr_neg.mean():.2f}M (n = {len(hr_neg)})")
        print(f"  All Other Conditions:           M = ${all_others.mean():.2f}M (n = {len(all_others)})")
        print("  ⚠️  Cannot compute interaction test: Missing data for comparison")
        t_stat_int, p_val_int, effect_size_int = np.nan, np.nan, np.nan
    
    # Additional pairwise comparisons
    lr_neg = df[(df['responsibility'] == 'low') & (df['consequences'] == 'negative')]['escalation_allocation_M']
    lr_pos = df[(df['responsibility'] == 'low') & (df['consequences'] == 'positive')]['escalation_allocation_M']
    hr_pos = df[(df['responsibility'] == 'high') & (df['consequences'] == 'positive')]['escalation_allocation_M']
    
    print(f"\nADDITIONAL PAIRWISE COMPARISONS:")
    
    if len(lr_neg) > 0 and len(lr_pos) > 0:
        t_lr_conseq, p_lr_conseq = ttest_ind(lr_neg, lr_pos)
        print(f"  Low Responsibility: Negative vs Positive consequences")
        print(f"    t(df = {len(lr_neg) + len(lr_pos) - 2}) = {t_lr_conseq:.3f}, p = {p_lr_conseq:.3f}")
        print(f"    Original study: t = 1.20, n.s. {'✓ Matches' if p_lr_conseq > 0.05 else '✗ Differs'}")
    else:
        print(f"  Low Responsibility: Negative vs Positive consequences")
        print(f"    ⚠️  Cannot compute: Missing low responsibility data")
    
    if len(hr_pos) > 0 and len(lr_pos) > 0:
        t_pos_resp, p_pos_resp = ttest_ind(hr_pos, lr_pos)
        print(f"  Positive Consequences: High vs Low responsibility")
        print(f"    t(df = {len(hr_pos) + len(lr_pos) - 2}) = {t_pos_resp:.3f}, p = {p_pos_resp:.3f}")
        print(f"    Original study: t = 1.13, n.s. {'✓ Matches' if p_pos_resp > 0.05 else '✗ Differs'}")
    else:
        print(f"  Positive Consequences: High vs Low responsibility")
        print(f"    ⚠️  Cannot compute: Missing data for one or both conditions")
    
    print("\n" + "=" * 80)
    print("ESCALATION OF COMMITMENT - PRIMARY HYPOTHESIS TEST")
    print("=" * 80)
    
    print("ESCALATION HYPOTHESIS: High Responsibility subjects will allocate more")
    print("money to previously chosen alternatives after NEGATIVE consequences")
    
    hr_pos_escalation = df[(df['responsibility'] == 'high') & (df['consequences'] == 'positive')]['escalation_allocation_M']
    hr_neg_escalation = df[(df['responsibility'] == 'high') & (df['consequences'] == 'negative')]['escalation_allocation_M']
    
    print(f"\nHigh Responsibility Conditions:")
    print(f"  After Positive Outcomes: M = ${hr_pos_escalation.mean():.2f}M (n = {len(hr_pos_escalation)})")
    print(f"  After Negative Outcomes: M = ${hr_neg_escalation.mean():.2f}M (n = {len(hr_neg_escalation)})")
    
    if len(hr_pos_escalation) > 0 and len(hr_neg_escalation) > 0:
        t_escalation, p_escalation = ttest_ind(hr_neg_escalation, hr_pos_escalation)
        escalation_effect_size = cohens_d(hr_neg_escalation, hr_pos_escalation)
        
        print(f"  Escalation Effect: ${hr_neg_escalation.mean() - hr_pos_escalation.mean():.2f}M")
        print(f"  t(df = {len(hr_neg_escalation) + len(hr_pos_escalation) - 2}) = {t_escalation:.3f}, p = {p_escalation:.3f}")
        
        print(f"\nHYPOTHESIS TEST RESULTS:")
        escalation_direction = "✓ Confirmed" if hr_neg_escalation.mean() > hr_pos_escalation.mean() else "✗ Not confirmed"
        escalation_significance = "✓ Significant" if p_escalation < 0.05 else "✗ Not significant"
        print(f"  Direction: {escalation_direction}")
        print(f"  Statistical Significance: {escalation_significance}")
        print(f"  Original Study Comparison: Expected positive escalation effect")
        print(f"  Effect Size (Cohen's d): {escalation_effect_size:.3f}")
    else:
        print(f"  ⚠️  Cannot compute escalation test: Missing data for one or both conditions")
        t_escalation, p_escalation, escalation_effect_size = np.nan, np.nan, np.nan
        
        print(f"\nHYPOTHESIS TEST RESULTS:")
        print(f"  ⚠️  Cannot test escalation hypothesis: Need both positive and negative high responsibility data")
    
    print("\n" + "=" * 80)
    print("EFFECT SIZES SUMMARY (Cohen's d)")
    print("=" * 80)
    
    if len(high_resp) > 0 and len(low_resp) > 0:
        resp_effect_size = cohens_d(high_resp, low_resp)
        print(f"  Personal Responsibility main effect: d = {resp_effect_size:.3f}")
    else:
        print(f"  Personal Responsibility main effect: d = N/A (insufficient data)")
    
    if len(negative_conseq) > 0 and len(positive_conseq) > 0:
        conseq_effect_size = cohens_d(negative_conseq, positive_conseq)
        print(f"  Decision Consequences main effect:   d = {conseq_effect_size:.3f}")
    else:
        print(f"  Decision Consequences main effect:   d = N/A (insufficient data)")
    
    if 'effect_size_int' in locals():
        print(f"  Interaction effect (HR+Neg vs others): d = {effect_size_int:.3f}")
    else:
        print(f"  Interaction effect (HR+Neg vs others): d = N/A (insufficient data)")
    
    print(f"\nEffect Size Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large")
    
    print("\n" + "=" * 80)
    print("SUMMARY FOR RESULTS SECTION")
    print("=" * 80)
    print("This analysis replicates the statistical approach of Staw (1976).")
    print("Key findings to report:")
    print("")
    print("1. DESCRIPTIVE STATISTICS: Report the four cell means and standard deviations")
    print("2. MAIN EFFECTS: Report t-tests for Responsibility and Consequences")
    print("3. INTERACTION: Report the critical test of High Resp + Negative vs. others")
    print("4. ESCALATION: Report the primary escalation hypothesis test")
    print("5. EFFECT SIZES: Include Cohen's d for all major comparisons")
    print("")
    print("Expected pattern from original study:")
    print("- High Responsibility + Negative should show highest allocation")
    print("- This condition should be significantly different from all others")
    print("- Main effects of both Responsibility and Consequences")
    print("")
    print("LLM-specific insights:")
    print("- Do LLMs show the same escalation bias as humans?")
    print("- Are LLMs affected by personal responsibility manipulation?")
    print("- How do LLMs respond to negative feedback?")
    
    # Return summary statistics for further analysis
    summary = {
        'descriptive_stats': descriptive_stats,
        'main_effects': {
            'responsibility': {'t': t_stat_resp, 'p': p_val_resp, 'effect_size': resp_effect_size},
            'consequences': {'t': t_stat_conseq, 'p': p_val_conseq, 'effect_size': conseq_effect_size}
        },
        'interaction': {'t': t_stat_int, 'p': p_val_int, 'effect_size': effect_size_int},
        'escalation_hypothesis': {'t': t_escalation, 'p': p_escalation, 'effect_size': escalation_effect_size},
        'cell_means': cell_means,
        'raw_data': df
    }
    
    return summary

if __name__ == "__main__":
    # Run the analysis
    results = run_analysis("/Users/leo/Documents/GitHub/escalation-commitment/symm_deliberation_runs")
    
    # Optionally save results to file
    if results is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary statistics
        with open(f"escalation_analysis_summary_{timestamp}.json", 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            summary_for_json = {
                'descriptive_stats': results['descriptive_stats'].to_dict(),
                'main_effects': {
                    k: {k2: float(v2) if isinstance(v2, (np.float64, np.int64)) else v2 
                        for k2, v2 in v.items()} 
                    for k, v in results['main_effects'].items()
                },
                'interaction': {k: float(v) if isinstance(v, (np.float64, np.int64)) else v 
                              for k, v in results['interaction'].items()},
                'escalation_hypothesis': {k: float(v) if isinstance(v, (np.float64, np.int64)) else v 
                                        for k, v in results['escalation_hypothesis'].items()},
                'cell_means': results['cell_means'].to_dict(),
                'analysis_timestamp': timestamp
            }
            json.dump(summary_for_json, f, indent=4)
        
        # Save raw data with analysis variables
        results['raw_data'].to_csv(f"escalation_raw_data_{timestamp}.csv", index=False)
        
        print(f"\n\nAnalysis complete! Results saved to:")
        print(f"- escalation_analysis_summary_{timestamp}.json")
        print(f"- escalation_raw_data_{timestamp}.csv")