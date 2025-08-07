import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import os

def load_experiment_results(results_dir="experiment_results"):
    """Load all experiment results from JSON files"""
    results = []
    
    # Load all results file
    all_results_path = os.path.join(results_dir, "all_results.json")
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r') as f:
            results = json.load(f)
    else:
        print("No all_results.json found. Loading individual files...")
        # Load individual condition files
        condition_files = [
            "high_responsibility_positive.json",
            "high_responsibility_negative.json",
            "low_responsibility_division_a_positive.json",
            "low_responsibility_division_a_negative.json",
            "low_responsibility_division_b_positive.json",
            "low_responsibility_division_b_negative.json"
        ]
        
        for filename in condition_files:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    condition_results = json.load(f)
                    results.extend(condition_results)
    
    return results

def prepare_data_for_analysis(results):
    """Prepare data for escalation of commitment analysis"""
    
    # Filter for valid results only
    valid_results = [r for r in results if r.get('commitment') is not None]
    
    if not valid_results:
        print("No valid results found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(valid_results)
    
    # Calculate commitment percentage (out of total $20M allocation)
    df['commitment_percentage'] = (df['commitment'] / 20000000) * 100
    
    # For high responsibility subjects, we need to map their choice to the condition
    # For low responsibility subjects, we use the previous_choice
    df['chosen_division'] = df.apply(lambda row: 
        row['first_choice'] if row['responsibility'] == 'high' 
        else row['previous_choice'], axis=1)
    
    # Create binary variables for ANOVA
    df['responsibility_binary'] = df['responsibility'].map({'high': 1, 'low': 0})
    df['condition_binary'] = df['condition'].map({'positive': 1, 'negative': 0})
    
    return df

def calculate_escalation_metrics(df):
    """Calculate key escalation of commitment metrics"""
    
    # Group by responsibility and condition
    grouped = df.groupby(['responsibility', 'condition'])
    
    metrics = {}
    
    for (responsibility, condition), group in grouped:
        key = f"{responsibility}_{condition}"
        
        metrics[key] = {
            'n': len(group),
            'mean_commitment': group['commitment'].mean(),
            'std_commitment': group['commitment'].std(),
            'mean_commitment_pct': group['commitment_percentage'].mean(),
            'std_commitment_pct': group['commitment_percentage'].std(),
            'median_commitment': group['commitment'].median(),
            'min_commitment': group['commitment'].min(),
            'max_commitment': group['commitment'].max()
        }
    
    return metrics

def perform_2x2_anova(df):
    """Perform 2x2 ANOVA on commitment levels"""
    
    # Check if we have data for all conditions
    condition_counts = df.groupby(['responsibility', 'condition']).size()
    print("Sample sizes by condition:")
    print(condition_counts)
    print()
    
    if len(condition_counts) < 4:
        print("Warning: Not all conditions have data. ANOVA may not be meaningful.")
        return None, None
    
    # Perform 2x2 ANOVA using commitment as dependent variable
    formula = 'commitment ~ C(responsibility) * C(condition)'
    model = ols(formula, data=df).fit()
    anova_results = anova_lm(model, typ=2)
    
    return model, anova_results

def detect_escalation_of_commitment(df, metrics):
    """Detect if escalation of commitment occurred"""
    
    print("=== ESCALATION OF COMMITMENT ANALYSIS ===\n")
    
    # Theory: High responsibility subjects should show more commitment 
    # to their original choice, especially in negative feedback conditions
    
    # Calculate means for each condition
    try:
        high_pos_commitment = metrics.get('high_positive', {}).get('mean_commitment', 0)
        high_neg_commitment = metrics.get('high_negative', {}).get('mean_commitment', 0)
        low_pos_commitment = metrics.get('low_positive', {}).get('mean_commitment', 0)
        low_neg_commitment = metrics.get('low_negative', {}).get('mean_commitment', 0)
        
        # If we don't have aggregated low responsibility data, calculate it
        if low_pos_commitment == 0 or low_neg_commitment == 0:
            # Calculate from raw data
            low_pos_data = df[(df['responsibility'] == 'low') & (df['condition'] == 'positive')]
            low_neg_data = df[(df['responsibility'] == 'low') & (df['condition'] == 'negative')]
            
            if len(low_pos_data) > 0:
                low_pos_commitment = low_pos_data['commitment'].mean()
            if len(low_neg_data) > 0:
                low_neg_commitment = low_neg_data['commitment'].mean()
        
    except KeyError as e:
        print(f"Missing condition data: {e}")
        return None
    
    print("Mean Commitment by Condition:")
    print(f"High Responsibility + Positive Feedback: ${high_pos_commitment:,.0f}")
    print(f"High Responsibility + Negative Feedback: ${high_neg_commitment:,.0f}")
    print(f"Low Responsibility + Positive Feedback: ${low_pos_commitment:,.0f}")
    print(f"Low Responsibility + Negative Feedback: ${low_neg_commitment:,.0f}")
    print()
    
    # Key indicators of escalation of commitment:
    # 1. Interaction effect: High responsibility subjects should increase commitment 
    #    more in negative conditions relative to low responsibility subjects
    
    high_diff = high_neg_commitment - high_pos_commitment
    low_diff = low_neg_commitment - low_pos_commitment
    interaction_effect = high_diff - low_diff
    
    print("Escalation Indicators:")
    print(f"High Responsibility: Negative - Positive = ${high_diff:,.0f}")
    print(f"Low Responsibility: Negative - Positive = ${low_diff:,.0f}")
    print(f"Interaction Effect: ${interaction_effect:,.0f}")
    print()
    
    # Interpretation
    escalation_detected = False
    
    if interaction_effect > 0:
        print("✓ ESCALATION OF COMMITMENT DETECTED")
        print("High responsibility subjects increased commitment more in negative feedback conditions")
        escalation_detected = True
    elif interaction_effect < 0:
        print("✗ REVERSE ESCALATION DETECTED")
        print("Low responsibility subjects increased commitment more in negative feedback conditions")
    else:
        print("→ NO DIFFERENTIAL ESCALATION")
        print("Both groups responded similarly to feedback")
    
    return {
        'escalation_detected': escalation_detected,
        'interaction_effect': interaction_effect,
        'high_responsibility_diff': high_diff,
        'low_responsibility_diff': low_diff,
        'means': {
            'high_positive': high_pos_commitment,
            'high_negative': high_neg_commitment,
            'low_positive': low_pos_commitment,
            'low_negative': low_neg_commitment
        }
    }

def create_visualizations(df, metrics):
    """Create visualizations for the analysis"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Escalation of Commitment Analysis', fontsize=16, fontweight='bold')
    
    # 1. Bar plot of mean commitment by condition
    ax1 = axes[0, 0]
    
    # Prepare data for bar plot - use direct calculation from dataframe
    condition_means = df.groupby(['responsibility', 'condition'])['commitment'].mean()
    
    conditions = []
    commitments = []
    colors = []
    
    for (responsibility, condition), commitment in condition_means.items():
        conditions.append(f"{responsibility.title()}\n{condition.title()}")
        commitments.append(commitment)
        # Color coding: blue for high responsibility, orange for low responsibility
        colors.append('#1f77b4' if responsibility == 'high' else '#ff7f0e')
    
    bars = ax1.bar(conditions, commitments, color=colors, alpha=0.7)
    ax1.set_title('Mean Commitment by Condition')
    ax1.set_ylabel('Mean Commitment ($)')
    ax1.tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for bar, value in zip(bars, commitments):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
                f'${value:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Interaction plot
    ax2 = axes[0, 1]
    
    # Calculate means for interaction plot
    interaction_data = df.groupby(['responsibility', 'condition'])['commitment'].mean().reset_index()
    
    for responsibility in ['high', 'low']:
        data = interaction_data[interaction_data['responsibility'] == responsibility]
        if len(data) > 0:
            ax2.plot(data['condition'], data['commitment'], 
                    marker='o', linewidth=2, markersize=8, 
                    label=f'{responsibility.title()} Responsibility')
    
    ax2.set_title('Interaction Plot: Responsibility × Condition')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Mean Commitment ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = axes[1, 0]
    
    # Create combined condition variable for box plot
    df['condition_combined'] = df['responsibility'] + '_' + df['condition']
    
    sns.boxplot(data=df, x='condition_combined', y='commitment', ax=ax3)
    ax3.set_title('Distribution of Commitment by Condition')
    ax3.set_xlabel('Condition')
    ax3.set_ylabel('Commitment ($)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Histogram of commitment percentages
    ax4 = axes[1, 1]
    
    for responsibility in ['high', 'low']:
        data = df[df['responsibility'] == responsibility]['commitment_percentage']
        if len(data) > 0:
            ax4.hist(data, alpha=0.6, bins=10, label=f'{responsibility.title()} Responsibility')
    
    ax4.set_title('Distribution of Commitment Percentages')
    ax4.set_xlabel('Commitment Percentage (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def perform_additional_analyses(df):
    """Perform additional analyses specific to the escalation of commitment design"""
    
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSES")
    print("="*60)
    
    # 1. Choice balance analysis (only for high responsibility)
    high_resp_data = df[df['responsibility'] == 'high']
    if len(high_resp_data) > 0:
        choice_counts = high_resp_data['first_choice'].value_counts()
        print(f"\nInitial Choice Distribution (High Responsibility):")
        for choice, count in choice_counts.items():
            print(f"  Division {choice.upper()}: {count} subjects ({count/len(high_resp_data)*100:.1f}%)")
        
        # Test for choice balance
        from scipy.stats import chi2_contingency
        choice_matrix = pd.crosstab(high_resp_data['first_choice'], high_resp_data['condition'])
        chi2, p_value, _, _ = chi2_contingency(choice_matrix)
        print(f"  Choice balance test: χ² = {chi2:.3f}, p = {p_value:.3f}")
    
    # 2. Allocation completeness check
    total_allocations = df['division_a_allocation'] + df['division_b_allocation']
    incomplete_allocations = total_allocations[total_allocations != 20000000]
    
    print(f"\nAllocation Completeness:")
    print(f"  Complete allocations ($20M): {len(df) - len(incomplete_allocations)}")
    print(f"  Incomplete allocations: {len(incomplete_allocations)}")
    
    if len(incomplete_allocations) > 0:
        print(f"  Mean incomplete allocation: ${incomplete_allocations.mean():,.0f}")
    
    # 3. Commitment range analysis
    print(f"\nCommitment Statistics:")
    print(f"  Overall mean: ${df['commitment'].mean():,.0f} ({df['commitment_percentage'].mean():.1f}%)")
    print(f"  Overall std: ${df['commitment'].std():,.0f} ({df['commitment_percentage'].std():.1f}%)")
    print(f"  Range: ${df['commitment'].min():,.0f} - ${df['commitment'].max():,.0f}")
    
    # 4. Effect size calculation
    high_pos = df[(df['responsibility'] == 'high') & (df['condition'] == 'positive')]['commitment']
    high_neg = df[(df['responsibility'] == 'high') & (df['condition'] == 'negative')]['commitment']
    low_pos = df[(df['responsibility'] == 'low') & (df['condition'] == 'positive')]['commitment']
    low_neg = df[(df['responsibility'] == 'low') & (df['condition'] == 'negative')]['commitment']
    
    if len(high_pos) > 0 and len(high_neg) > 0:
        high_effect = (high_neg.mean() - high_pos.mean()) / high_pos.std()
        print(f"\nEffect Sizes (Cohen's d):")
        print(f"  High Responsibility (Negative - Positive): {high_effect:.3f}")
    
    if len(low_pos) > 0 and len(low_neg) > 0:
        low_effect = (low_neg.mean() - low_pos.mean()) / low_pos.std()
        print(f"  Low Responsibility (Negative - Positive): {low_effect:.3f}")

def generate_report(df, metrics, anova_results, escalation_results):
    """Generate a comprehensive analysis report"""
    
    print("\n" + "="*60)
    print("ESCALATION OF COMMITMENT EXPERIMENT REPORT")
    print("="*60)
    
    print(f"\nSample Size: {len(df)} subjects")
    print(f"High Responsibility: {len(df[df['responsibility'] == 'high'])}")
    print(f"Low Responsibility: {len(df[df['responsibility'] == 'low'])}")
    
    print("\n" + "-"*40)
    print("DESCRIPTIVE STATISTICS")
    print("-"*40)
    
    # Calculate statistics directly from dataframe to ensure accuracy
    for (responsibility, condition), group in df.groupby(['responsibility', 'condition']):
        print(f"\n{responsibility.title()} Responsibility, {condition.title()} Condition:")
        print(f"  N = {len(group)}")
        print(f"  Mean Commitment: ${group['commitment'].mean():,.0f} ({group['commitment_percentage'].mean():.1f}%)")
        print(f"  Std Deviation: ${group['commitment'].std():,.0f}")
        print(f"  Range: ${group['commitment'].min():,.0f} - ${group['commitment'].max():,.0f}")
    
    print("\n" + "-"*40)
    print("2x2 ANOVA RESULTS")
    print("-"*40)
    
    if anova_results is not None:
        print(anova_results)
        
        # Interpret results
        p_responsibility = anova_results.loc['C(responsibility)', 'PR(>F)']
        p_condition = anova_results.loc['C(condition)', 'PR(>F)']
        p_interaction = anova_results.loc['C(responsibility):C(condition)', 'PR(>F)']
        
        print(f"\nInterpretation:")
        print(f"  Main effect of Responsibility: {'Significant' if p_responsibility < 0.05 else 'Not significant'} (p = {p_responsibility:.4f})")
        print(f"  Main effect of Condition: {'Significant' if p_condition < 0.05 else 'Not significant'} (p = {p_condition:.4f})")
        print(f"  Interaction Effect: {'Significant' if p_interaction < 0.05 else 'Not significant'} (p = {p_interaction:.4f})")
    else:
        print("ANOVA could not be performed due to missing conditions.")
    
    print("\n" + "-"*40)
    print("ESCALATION OF COMMITMENT ANALYSIS")
    print("-"*40)
    
    if escalation_results:
        print(f"Interaction Effect: ${escalation_results['interaction_effect']:,.0f}")
        print(f"Escalation Detected: {'YES' if escalation_results['escalation_detected'] else 'NO'}")
        
        if anova_results is not None:
            p_interaction = anova_results.loc['C(responsibility):C(condition)', 'PR(>F)']
            if p_interaction < 0.05:
                print(f"Statistical Support: SIGNIFICANT (p = {p_interaction:.4f})")
            else:
                print(f"Statistical Support: Not significant (p = {p_interaction:.4f})")
    
    print("\n" + "-"*40)
    print("CONCLUSIONS")
    print("-"*40)
    
    if escalation_results and escalation_results['escalation_detected']:
        print("✓ Escalation of commitment was detected.")
        print("  High responsibility subjects showed greater commitment to their")
        print("  original choices when faced with negative feedback.")
        if anova_results is not None and anova_results.loc['C(responsibility):C(condition)', 'PR(>F)'] < 0.05:
            print("  This effect is statistically significant.")
    else:
        print("✗ Escalation of commitment was not detected.")
        print("  The pattern does not support the escalation of commitment hypothesis.")

def main_analysis(results_dir="experiment_results", save_plots=True):
    """Main analysis function"""
    
    print("Loading experiment results...")
    results = load_experiment_results(results_dir)
    
    if not results:
        print("No results found!")
        return None
    
    print(f"Loaded {len(results)} results")
    
    # Prepare data
    df = prepare_data_for_analysis(results)
    if df is None:
        return None
    
    print(f"Prepared {len(df)} valid results for analysis")
    
    # Calculate metrics
    metrics = calculate_escalation_metrics(df)
    
    # Perform ANOVA
    model, anova_results = perform_2x2_anova(df)
    
    # Detect escalation
    escalation_results = detect_escalation_of_commitment(df, metrics)
    
    # Additional analyses
    perform_additional_analyses(df)
    
    # Create visualizations
    if save_plots:
        fig = create_visualizations(df, metrics)
        plt.savefig(os.path.join(results_dir, 'escalation_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # Generate report
    generate_report(df, metrics, anova_results, escalation_results)
    
    return {
        'dataframe': df,
        'metrics': metrics,
        'anova_results': anova_results,
        'escalation_results': escalation_results,
        'model': model
    }

if __name__ == "__main__":
    # Run the analysis
    analysis_results = main_analysis()
    
    if analysis_results:
        print("\nAnalysis complete! Check the generated plots and results.")
    else:
        print("Analysis failed. Please check your data files.")