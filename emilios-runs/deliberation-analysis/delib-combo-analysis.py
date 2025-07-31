import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import glob

def load_data(base_dir=".", symm_dir="/Users/leo/Documents/GitHub/escalation-commitment/symm_deliberation_runs", asymm_dir="/Users/leo/Documents/GitHub/escalation-commitment/asymm_deliberation_runs"):
    """Load data from both symmetrical and asymmetrical collaboration runs"""
    
    data = []
    
    # Load symmetrical collaboration data
    symm_path = os.path.join(base_dir, symm_dir, "*.json")
    symm_files = glob.glob(symm_path)
    
    for file_path in symm_files:
        with open(file_path, 'r') as f:
            try:
                content = json.load(f)
                if isinstance(content, list) and len(content) > 0:
                    record = content[0]  # Take first record
                    record['collaboration_type'] = 'symmetrical'
                    record['file_path'] = file_path
                    
                    # Extract condition from filename
                    filename = os.path.basename(file_path)
                    if 'positive' in filename.lower():
                        record['condition'] = 'positive'
                    elif 'negative' in filename.lower():
                        record['condition'] = 'negative'
                    
                    # Extract responsibility level
                    if 'high' in filename.lower():
                        record['responsibility'] = 'high'
                    elif 'low' in filename.lower():
                        record['responsibility'] = 'low'
                    
                    data.append(record)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Load asymmetrical collaboration data
    asymm_path = os.path.join(base_dir, asymm_dir, "*.json")
    asymm_files = glob.glob(asymm_path)
    
    for file_path in asymm_files:
        with open(file_path, 'r') as f:
            try:
                content = json.load(f)
                if isinstance(content, list) and len(content) > 0:
                    record = content[0]  # Take first record
                    record['collaboration_type'] = 'asymmetrical'
                    record['file_path'] = file_path
                    
                    # Extract condition from filename
                    filename = os.path.basename(file_path)
                    if 'positive' in filename.lower():
                        record['condition'] = 'positive'
                    elif 'negative' in filename.lower():
                        record['condition'] = 'negative'
                    
                    # Extract responsibility level
                    if 'high' in filename.lower():
                        record['responsibility'] = 'high'
                    elif 'low' in filename.lower():
                        record['responsibility'] = 'low'
                    
                    data.append(record)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return pd.DataFrame(data)

def calculate_total_allocation(row):
    """Calculate total allocation from consumer and industrial allocations"""
    try:
        if 'consumer_allocation' in row and 'industrial_allocation' in row:
            consumer = float(str(row['consumer_allocation']).replace('$', '').replace(',', ''))
            industrial = float(str(row['industrial_allocation']).replace('$', '').replace(',', ''))
            return consumer + industrial
        return np.nan
    except:
        return np.nan

def get_previous_choice_allocation(row):
    """Get allocation to previously chosen division"""
    try:
        consumer = float(str(row['consumer_allocation']).replace('$', '').replace(',', ''))
        industrial = float(str(row['industrial_allocation']).replace('$', '').replace(',', ''))
        
        # For high responsibility, we have first_choice
        if 'first_choice' in row and pd.notna(row['first_choice']):
            if 'consumer' in str(row['first_choice']).lower():
                return consumer
            elif 'industrial' in str(row['first_choice']).lower():
                return industrial
        
        # For low responsibility, we have product_choice
        if 'product_choice' in row and pd.notna(row['product_choice']):
            if 'consumer' in str(row['product_choice']).lower():
                return consumer
            elif 'industrial' in str(row['product_choice']).lower():
                return industrial
        
        return np.nan
    except:
        return np.nan

def cohen_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def analyze_escalation_collaboration(df):
    """Main analysis function"""
    
    print("=" * 80)
    print("ESCALATION OF COMMITMENT ANALYSIS - COLLABORATION TYPES COMPARISON")
    print("=" * 80)
    print(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total subjects analyzed: {len(df)}")
    print()
    
    # Calculate allocations
    df['total_allocation'] = df.apply(calculate_total_allocation, axis=1)
    df['previous_choice_allocation'] = df.apply(get_previous_choice_allocation, axis=1)
    
    # Convert to millions for easier reading
    df['total_allocation_M'] = df['total_allocation'] / 1_000_000
    df['previous_choice_allocation_M'] = df['previous_choice_allocation'] / 1_000_000
    
    # Filter out any rows with missing data
    df_clean = df.dropna(subset=['previous_choice_allocation_M', 'collaboration_type', 'responsibility', 'condition'])
    
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    # Create condition labels
    df_clean['full_condition'] = (df_clean['collaboration_type'].str.title() + ' ' + 
                                 df_clean['responsibility'].str.title() + ' Resp + ' + 
                                 df_clean['condition'].str.title())
    
    desc_stats = df_clean.groupby('full_condition')['previous_choice_allocation_M'].agg([
        'count', 'mean', 'std'
    ]).round(2)
    desc_stats['se'] = (desc_stats['std'] / np.sqrt(desc_stats['count'])).round(2)
    
    print(f"{'Condition':<40} {'N':<8} {'Mean ($M)':<12} {'SD ($M)':<12} {'SE ($M)':<12}")
    print("-" * 80)
    for condition, row in desc_stats.iterrows():
        print(f"{condition:<40} {row['count']:<8} {row['mean']:<12} {row['std']:<12} {row['se']:<12}")
    
    print()
    print("=" * 80)
    print("COLLABORATION TYPE EFFECTS ANALYSIS")
    print("=" * 80)
    
    # Main effect of collaboration type
    symm_data = df_clean[df_clean['collaboration_type'] == 'symmetrical']['previous_choice_allocation_M']
    asymm_data = df_clean[df_clean['collaboration_type'] == 'asymmetrical']['previous_choice_allocation_M']
    
    if len(symm_data) > 0 and len(asymm_data) > 0:
        t_stat, p_val = stats.ttest_ind(symm_data, asymm_data)
        effect_size = cohen_d(symm_data, asymm_data)
        
        print(f"MAIN EFFECT OF COLLABORATION TYPE:")
        print(f"  Symmetrical Collaboration:   M = ${symm_data.mean():.2f}M (n = {len(symm_data)})")
        print(f"  Asymmetrical Collaboration: M = ${asymm_data.mean():.2f}M (n = {len(asymm_data)})")
        print(f"  Difference: ${symm_data.mean() - asymm_data.mean():.2f}M")
        print(f"  t({len(symm_data) + len(asymm_data) - 2}) = {t_stat:.3f}, p = {p_val:.3f}")
        print(f"  Cohen's d = {effect_size:.3f}")
        print(f"  Result: {'✓ Significant' if p_val < 0.05 else '✗ Not significant'}")
        print()
    
    # Main effect of responsibility
    high_resp = df_clean[df_clean['responsibility'] == 'high']['previous_choice_allocation_M']
    low_resp = df_clean[df_clean['responsibility'] == 'low']['previous_choice_allocation_M']
    
    if len(high_resp) > 0 and len(low_resp) > 0:
        t_stat, p_val = stats.ttest_ind(high_resp, low_resp)
        effect_size = cohen_d(high_resp, low_resp)
        
        print(f"MAIN EFFECT OF PERSONAL RESPONSIBILITY:")
        print(f"  High Personal Responsibility: M = ${high_resp.mean():.2f}M (n = {len(high_resp)})")
        print(f"  Low Personal Responsibility:  M = ${low_resp.mean():.2f}M (n = {len(low_resp)})")
        print(f"  Difference: ${high_resp.mean() - low_resp.mean():.2f}M")
        print(f"  t({len(high_resp) + len(low_resp) - 2}) = {t_stat:.3f}, p = {p_val:.3f}")
        print(f"  Cohen's d = {effect_size:.3f}")
        print(f"  Result: {'✓ Significant' if p_val < 0.05 else '✗ Not significant'}")
        print()
    
    # Main effect of decision consequences
    positive = df_clean[df_clean['condition'] == 'positive']['previous_choice_allocation_M']
    negative = df_clean[df_clean['condition'] == 'negative']['previous_choice_allocation_M']
    
    if len(positive) > 0 and len(negative) > 0:
        t_stat, p_val = stats.ttest_ind(positive, negative)
        effect_size = cohen_d(positive, negative)
        
        print(f"MAIN EFFECT OF DECISION CONSEQUENCES:")
        print(f"  Positive Consequences: M = ${positive.mean():.2f}M (n = {len(positive)})")
        print(f"  Negative Consequences: M = ${negative.mean():.2f}M (n = {len(negative)})")
        print(f"  Difference: ${positive.mean() - negative.mean():.2f}M")
        print(f"  t({len(positive) + len(negative) - 2}) = {t_stat:.3f}, p = {p_val:.3f}")
        print(f"  Cohen's d = {effect_size:.3f}")
        print(f"  Result: {'✓ Significant' if p_val < 0.05 else '✗ Not significant'}")
        print()
    
    print("=" * 80)
    print("COLLABORATION TYPE × RESPONSIBILITY × CONSEQUENCES INTERACTION")
    print("=" * 80)
    
    # Create all 8 conditions
    conditions = ['symmetrical', 'asymmetrical']
    responsibilities = ['high', 'low']
    consequences = ['positive', 'negative']
    
    print("CELL MEANS:")
    cell_means = {}
    for collab in conditions:
        for resp in responsibilities:
            for cons in consequences:
                subset = df_clean[
                    (df_clean['collaboration_type'] == collab) &
                    (df_clean['responsibility'] == resp) &
                    (df_clean['condition'] == cons)
                ]
                if len(subset) > 0:
                    mean_val = subset['previous_choice_allocation_M'].mean()
                    cell_means[f"{collab}_{resp}_{cons}"] = mean_val
                    print(f"  {collab.title()} {resp.title()} Resp + {cons.title():<10}: M = ${mean_val:.2f}M (n = {len(subset)})")
    
    print()
    print("=" * 80)
    print("ESCALATION HYPOTHESIS TESTS BY COLLABORATION TYPE")
    print("=" * 80)
    
    # Test escalation for each collaboration type
    for collab_type in ['symmetrical', 'asymmetrical']:
        print(f"\n{collab_type.upper()} COLLABORATION:")
        
        # High responsibility conditions for this collaboration type
        high_pos = df_clean[
            (df_clean['collaboration_type'] == collab_type) &
            (df_clean['responsibility'] == 'high') &
            (df_clean['condition'] == 'positive')
        ]['previous_choice_allocation_M']
        
        high_neg = df_clean[
            (df_clean['collaboration_type'] == collab_type) &
            (df_clean['responsibility'] == 'high') &
            (df_clean['condition'] == 'negative')
        ]['previous_choice_allocation_M']
        
        if len(high_pos) > 0 and len(high_neg) > 0:
            t_stat, p_val = stats.ttest_ind(high_neg, high_pos)  # Test if negative > positive
            effect_size = cohen_d(high_neg, high_pos)
            
            print(f"  After Positive Outcomes: M = ${high_pos.mean():.2f}M (n = {len(high_pos)})")
            print(f"  After Negative Outcomes: M = ${high_neg.mean():.2f}M (n = {len(high_neg)})")
            print(f"  Escalation Effect: ${high_neg.mean() - high_pos.mean():.2f}M")
            print(f"  t({len(high_pos) + len(high_neg) - 2}) = {t_stat:.3f}, p = {p_val:.3f}")
            print(f"  Cohen's d = {effect_size:.3f}")
            escalation_direction = "✓ Confirmed" if high_neg.mean() > high_pos.mean() else "✗ Not confirmed"
            significance = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
            print(f"  Direction: {escalation_direction}")
            print(f"  Statistical Significance: {significance}")
    
    print()
    print("=" * 80)
    print("COLLABORATION TYPE COMPARISON IN ESCALATION CONDITIONS")
    print("=" * 80)
    
    # Compare collaboration types specifically in high responsibility + negative conditions
    symm_high_neg = df_clean[
        (df_clean['collaboration_type'] == 'symmetrical') &
        (df_clean['responsibility'] == 'high') &
        (df_clean['condition'] == 'negative')
    ]['previous_choice_allocation_M']
    
    asymm_high_neg = df_clean[
        (df_clean['collaboration_type'] == 'asymmetrical') &
        (df_clean['responsibility'] == 'high') &
        (df_clean['condition'] == 'negative')
    ]['previous_choice_allocation_M']
    
    if len(symm_high_neg) > 0 and len(asymm_high_neg) > 0:
        t_stat, p_val = stats.ttest_ind(symm_high_neg, asymm_high_neg)
        effect_size = cohen_d(symm_high_neg, asymm_high_neg)
        
        print("HIGH RESPONSIBILITY + NEGATIVE CONSEQUENCES:")
        print(f"  Symmetrical Collaboration:   M = ${symm_high_neg.mean():.2f}M (n = {len(symm_high_neg)})")
        print(f"  Asymmetrical Collaboration: M = ${asymm_high_neg.mean():.2f}M (n = {len(asymm_high_neg)})")
        print(f"  Difference: ${symm_high_neg.mean() - asymm_high_neg.mean():.2f}M")
        print(f"  t({len(symm_high_neg) + len(asymm_high_neg) - 2}) = {t_stat:.3f}, p = {p_val:.3f}")
        print(f"  Cohen's d = {effect_size:.3f}")
        print(f"  Result: {'✓ Significant' if p_val < 0.05 else '✗ Not significant'}")
    
    print()
    print("=" * 80)
    print("EFFECT SIZES SUMMARY (Cohen's d)")
    print("=" * 80)
    print("Effect Size Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large")
    print()
    
    # Calculate and display all major effect sizes
    effects = []
    
    if len(symm_data) > 0 and len(asymm_data) > 0:
        effects.append(("Collaboration Type main effect", cohen_d(symm_data, asymm_data)))
    
    if len(high_resp) > 0 and len(low_resp) > 0:
        effects.append(("Personal Responsibility main effect", cohen_d(high_resp, low_resp)))
    
    if len(positive) > 0 and len(negative) > 0:
        effects.append(("Decision Consequences main effect", cohen_d(positive, negative)))
    
    for effect_name, effect_size in effects:
        print(f"  {effect_name}: d = {effect_size:.3f}")
    
    print()
    print("=" * 80)
    print("SUMMARY FOR RESULTS SECTION")
    print("=" * 80)
    print("This analysis compares symmetrical vs asymmetrical collaboration")
    print("in escalation of commitment scenarios.")
    print()
    print("Key findings to report:")
    print("1. DESCRIPTIVE STATISTICS: Report means and SDs for all 8 conditions")
    print("2. MAIN EFFECTS: Test Collaboration Type, Responsibility, and Consequences")
    print("3. ESCALATION: Test escalation hypothesis within each collaboration type")
    print("4. COLLABORATION COMPARISON: Compare collaboration types in escalation conditions")
    print("5. EFFECT SIZES: Include Cohen's d for all major comparisons")
    print()
    print("Research Questions:")
    print("- Do LLMs show different escalation patterns in symmetrical vs asymmetrical collaboration?")
    print("- Is the escalation bias stronger in one collaboration type?")
    print("- How does the hierarchical structure (asymmetrical) vs equal partnership")
    print("  (symmetrical) affect decision-making under negative feedback?")
    
    return df_clean

def main():
    """Main execution function"""
    # Load and analyze data
    try:
        df = load_data()
        if len(df) == 0:
            print("No data files found. Please check the directory structure.")
            return
        
        df_analyzed = analyze_escalation_collaboration(df)
        
        # Optionally save results to CSV
        output_file = f"escalation_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_analyzed.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()