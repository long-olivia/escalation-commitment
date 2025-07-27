import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

class EscalationAnalyzer:
    """Analyzer for Escalation of Commitment experiment results"""
    
    def __init__(self, results_dir="escalation_results"):
        self.results_dir = Path(results_dir)
        self.data = None
        self.condition_data = {}
        
    def load_data(self, filename=None):
        """Load experiment data from JSON files"""
        if filename:
            # Load specific file
            filepath = self.results_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.data = pd.DataFrame(json.load(f))
                print(f"Loaded data from {filepath}")
            else:
                raise FileNotFoundError(f"File not found: {filepath}")
        else:
            # Find and load the most recent combined results file
            pattern = "all_results_*.json"
            result_files = list(self.results_dir.glob(pattern))
            
            if not result_files:
                raise FileNotFoundError(f"No combined results files found in {self.results_dir}")
            
            # Get most recent file
            latest_file = max(result_files, key=os.path.getctime)
            with open(latest_file, 'r') as f:
                self.data = pd.DataFrame(json.load(f))
            print(f"Loaded data from {latest_file}")
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['consumer_allocation', 'industrial_allocation', 'commitment_amount', 'total_allocated']
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        print(f"Loaded {len(self.data)} subjects")
        return self.data
    
    def descriptive_stats(self):
        """Calculate descriptive statistics by condition"""
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS")
        print("="*80)
        
        # Group by condition
        grouped = self.data.groupby('condition')
        
        results = {}
        for condition, group in grouped:
            stats_dict = {
                'n': len(group),
                'mean_commitment': group['commitment_amount'].mean(),
                'std_commitment': group['commitment_amount'].std(),
                'median_commitment': group['commitment_amount'].median(),
                'min_commitment': group['commitment_amount'].min(),
                'max_commitment': group['commitment_amount'].max(),
                'mean_total_allocated': group['total_allocated'].mean()
            }
            results[condition] = stats_dict
        
        # Create summary table
        df_stats = pd.DataFrame(results).T
        print("\nCommitment Amount Statistics by Condition:")
        print(df_stats.round(0).to_string())
        
        # Also group by responsibility and outcome separately
        print("\n\nBy Responsibility Level:")
        resp_stats = self.data.groupby('responsibility')['commitment_amount'].agg(['count', 'mean', 'std']).round(0)
        print(resp_stats.to_string())
        
        print("\nBy Outcome Valence:")
        outcome_stats = self.data.groupby('outcome')['commitment_amount'].agg(['count', 'mean', 'std']).round(0)
        print(outcome_stats.to_string())
        
        return df_stats
    
    def main_effects_analysis(self):
        """Test main effects and interaction using ANOVA"""
        print("\n" + "="*80)
        print("MAIN EFFECTS AND INTERACTION ANALYSIS")
        print("="*80)
        
        # Prepare data for ANOVA
        from scipy.stats import f_oneway
        
        # Get condition means
        high_pos = self.data[self.data['condition'] == 'high_responsibility_positive']['commitment_amount']
        high_neg = self.data[self.data['condition'] == 'high_responsibility_negative']['commitment_amount']
        low_pos = self.data[self.data['condition'] == 'low_responsibility_positive']['commitment_amount']
        low_neg = self.data[self.data['condition'] == 'low_responsibility_negative']['commitment_amount']
        
        print(f"\nCondition Means:")
        print(f"High Responsibility + Positive: ${high_pos.mean():8,.0f} (n={len(high_pos)})")
        print(f"High Responsibility + Negative: ${high_neg.mean():8,.0f} (n={len(high_neg)})")
        print(f"Low Responsibility + Positive:  ${low_pos.mean():8,.0f} (n={len(low_pos)})")
        print(f"Low Responsibility + Negative:  ${low_neg.mean():8,.0f} (n={len(low_neg)})")
        
        # Main effect of Responsibility
        high_resp = pd.concat([high_pos, high_neg])
        low_resp = pd.concat([low_pos, low_neg])
        
        resp_effect = high_resp.mean() - low_resp.mean()
        t_stat_resp, p_val_resp = stats.ttest_ind(high_resp, low_resp)
        
        print(f"\n1. MAIN EFFECT OF RESPONSIBILITY:")
        print(f"   High Responsibility Mean: ${high_resp.mean():8,.0f}")
        print(f"   Low Responsibility Mean:  ${low_resp.mean():8,.0f}")
        print(f"   Difference (High - Low):  ${resp_effect:8,.0f}")
        print(f"   t({len(high_resp) + len(low_resp) - 2:.0f}) = {t_stat_resp:.3f}, p = {p_val_resp:.3f}")
        
        # Main effect of Outcome
        positive = pd.concat([high_pos, low_pos])
        negative = pd.concat([high_neg, low_neg])
        
        outcome_effect = negative.mean() - positive.mean()
        t_stat_outcome, p_val_outcome = stats.ttest_ind(negative, positive)
        
        print(f"\n2. MAIN EFFECT OF OUTCOME:")
        print(f"   Positive Outcome Mean: ${positive.mean():8,.0f}")
        print(f"   Negative Outcome Mean: ${negative.mean():8,.0f}")
        print(f"   Difference (Neg - Pos): ${outcome_effect:8,.0f}")
        print(f"   t({len(positive) + len(negative) - 2:.0f}) = {t_stat_outcome:.3f}, p = {p_val_outcome:.3f}")
        
        # Interaction effect (key test for escalation of commitment)
        high_escalation = high_neg.mean() - high_pos.mean()
        low_escalation = low_neg.mean() - low_pos.mean()
        interaction_effect = high_escalation - low_escalation
        
        print(f"\n3. INTERACTION (ESCALATION OF COMMITMENT):")
        print(f"   High Responsibility Escalation: ${high_escalation:8,.0f}")
        print(f"   Low Responsibility Escalation:  ${low_escalation:8,.0f}")
        print(f"   Interaction Effect:             ${interaction_effect:8,.0f}")
        
        # Test the interaction using planned comparisons
        # High responsibility: negative vs positive
        t_stat_high, p_val_high = stats.ttest_ind(high_neg, high_pos)
        print(f"   High Resp (Neg vs Pos): t({len(high_neg) + len(high_pos) - 2:.0f}) = {t_stat_high:.3f}, p = {p_val_high:.3f}")
        
        # Low responsibility: negative vs positive  
        t_stat_low, p_val_low = stats.ttest_ind(low_neg, low_pos)
        print(f"   Low Resp (Neg vs Pos):  t({len(low_neg) + len(low_pos) - 2:.0f}) = {t_stat_low:.3f}, p = {p_val_low:.3f}")
        
        return {
            'responsibility_effect': resp_effect,
            'outcome_effect': outcome_effect, 
            'interaction_effect': interaction_effect,
            'escalation_high': high_escalation,
            'escalation_low': low_escalation,
            'p_values': {
                'responsibility': p_val_resp,
                'outcome': p_val_outcome,
                'high_escalation': p_val_high,
                'low_escalation': p_val_low
            }
        }
    
    def pairwise_comparisons(self):
        """Conduct pairwise t-tests between all conditions"""
        print("\n" + "="*80)
        print("PAIRWISE COMPARISONS")
        print("="*80)
        
        conditions = ['high_responsibility_positive', 'high_responsibility_negative', 
                     'low_responsibility_positive', 'low_responsibility_negative']
        
        results = {}
        
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions):
                if i < j:  # Only do each comparison once
                    group1 = self.data[self.data['condition'] == cond1]['commitment_amount']
                    group2 = self.data[self.data['condition'] == cond2]['commitment_amount']
                    
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    mean_diff = group2.mean() - group1.mean()
                    
                    results[f"{cond1}_vs_{cond2}"] = {
                        'mean_diff': mean_diff,
                        't_stat': t_stat,
                        'p_val': p_val,
                        'n1': len(group1),
                        'n2': len(group2)
                    }
                    
                    print(f"{cond1.replace('_', ' ').title()}")
                    print(f"  vs {cond2.replace('_', ' ').title()}")
                    print(f"  Mean Difference: ${mean_diff:8,.0f}")
                    print(f"  t({len(group1) + len(group2) - 2:.0f}) = {t_stat:6.3f}, p = {p_val:.3f}")
                    print()
        
        return results
    
    def effect_sizes(self):
        """Calculate effect sizes (Cohen's d) for key comparisons"""
        print("\n" + "="*80)
        print("EFFECT SIZES (COHEN'S d)")
        print("="*80)
        
        def cohens_d(group1, group2):
            """Calculate Cohen's d effect size"""
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
            return (group2.mean() - group1.mean()) / pooled_std
        
        # Key comparisons
        high_pos = self.data[self.data['condition'] == 'high_responsibility_positive']['commitment_amount']
        high_neg = self.data[self.data['condition'] == 'high_responsibility_negative']['commitment_amount']
        low_pos = self.data[self.data['condition'] == 'low_responsibility_positive']['commitment_amount']
        low_neg = self.data[self.data['condition'] == 'low_responsibility_negative']['commitment_amount']
        
        effects = {
            'High Responsibility Escalation (Neg vs Pos)': cohens_d(high_pos, high_neg),
            'Low Responsibility Escalation (Neg vs Pos)': cohens_d(low_pos, low_neg),
            'Responsibility Effect (High vs Low)': cohens_d(pd.concat([low_pos, low_neg]), pd.concat([high_pos, high_neg])),
            'Outcome Effect (Neg vs Pos)': cohens_d(pd.concat([high_pos, low_pos]), pd.concat([high_neg, low_neg]))
        }
        
        print("Effect Size Interpretation:")
        print("  Small: 0.2, Medium: 0.5, Large: 0.8\n")
        
        for comparison, d in effects.items():
            magnitude = "Small" if abs(d) < 0.5 else "Medium" if abs(d) < 0.8 else "Large"
            print(f"{comparison:40}: d = {d:6.3f} ({magnitude})")
        
        return effects
    
    def choice_analysis(self):
        """Analyze initial division choices in high responsibility conditions"""
        print("\n" + "="*80)
        print("INITIAL CHOICE ANALYSIS (High Responsibility Only)")
        print("="*80)
        
        high_resp_data = self.data[self.data['responsibility'] == 'high'].copy()
        
        if len(high_resp_data) == 0:
            print("No high responsibility data found")
            return
        
        # Choice distribution
        if 'stage1_choice' in high_resp_data.columns:
            choice_col = 'stage1_choice'
        else:
            print("Initial choice data not found in high responsibility conditions")
            return
        
        choice_counts = high_resp_data[choice_col].value_counts()
        print("Initial Division Choice Distribution:")
        for choice, count in choice_counts.items():
            pct = count / len(high_resp_data) * 100
            print(f"  {choice.title()}: {count} ({pct:.1f}%)")
        
        # Choice by outcome condition
        print("\nChoice by Outcome Condition:")
        choice_by_outcome = pd.crosstab(high_resp_data['outcome'], high_resp_data[choice_col])
        print(choice_by_outcome.to_string())
        
        # Commitment by initial choice
        print("\nCommitment Amount by Initial Choice:")
        commitment_by_choice = high_resp_data.groupby(choice_col)['commitment_amount'].agg(['count', 'mean', 'std']).round(0)
        print(commitment_by_choice.to_string())
    
    def allocation_patterns(self):
        """Analyze allocation patterns and extreme responses"""
        print("\n" + "="*80)
        print("ALLOCATION PATTERNS")
        print("="*80)
        
        # Check for extreme allocations (all to one division)
        extreme_consumer = (self.data['consumer_allocation'] == 20000000).sum()
        extreme_industrial = (self.data['industrial_allocation'] == 20000000).sum()
        balanced = (self.data['consumer_allocation'] == 10000000).sum()
        
        print(f"Allocation Patterns:")
        print(f"  All to Consumer ($20M):    {extreme_consumer} ({extreme_consumer/len(self.data)*100:.1f}%)")
        print(f"  All to Industrial ($20M):  {extreme_industrial} ({extreme_industrial/len(self.data)*100:.1f}%)")
        print(f"  Balanced Split ($10M each): {balanced} ({balanced/len(self.data)*100:.1f}%)")
        print(f"  Other allocations:          {len(self.data) - extreme_consumer - extreme_industrial - balanced}")
        
        # Allocation patterns by condition
        print("\nExtreme Allocations by Condition:")
        for condition in self.data['condition'].unique():
            cond_data = self.data[self.data['condition'] == condition]
            extreme_count = ((cond_data['consumer_allocation'] == 20000000) | 
                           (cond_data['industrial_allocation'] == 20000000)).sum()
            print(f"  {condition.replace('_', ' ').title()}: {extreme_count}/{len(cond_data)} ({extreme_count/len(cond_data)*100:.1f}%)")
    
    def create_visualizations(self, save_dir=None):
        """Create visualizations of the results"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure 1: Main results - commitment by condition
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Bar plot of means with error bars
        condition_means = self.data.groupby('condition')['commitment_amount'].agg(['mean', 'std', 'count'])
        condition_means['se'] = condition_means['std'] / np.sqrt(condition_means['count'])
        
        bars = ax1.bar(range(len(condition_means)), condition_means['mean'], 
                      yerr=condition_means['se'], capsize=5, alpha=0.7)
        ax1.set_xlabel('Condition')
        ax1.set_ylabel('Mean Commitment Amount ($)')
        ax1.set_title('Mean Commitment by Condition')
        ax1.set_xticks(range(len(condition_means)))
        ax1.set_xticklabels([c.replace('_', '\n').title() for c in condition_means.index], rotation=45)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.0f}M'))
        
        # Subplot 2: 2x2 interaction plot
        pivot_data = self.data.pivot_table(values='commitment_amount', 
                                         index='responsibility', 
                                         columns='outcome', 
                                         aggfunc='mean')
        
        x = [0, 1]
        ax2.plot(x, [pivot_data.loc['high', 'positive'], pivot_data.loc['high', 'negative']], 
                'o-', linewidth=2, markersize=8, label='High Responsibility')
        ax2.plot(x, [pivot_data.loc['low', 'positive'], pivot_data.loc['low', 'negative']], 
                's--', linewidth=2, markersize=8, label='Low Responsibility')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Positive Outcome', 'Negative Outcome'])
        ax2.set_ylabel('Mean Commitment Amount ($)')
        ax2.set_title('Responsibility × Outcome Interaction')
        ax2.legend()
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.0f}M'))
        
        # Subplot 3: Box plots
        sns.boxplot(data=self.data, x='condition', y='commitment_amount', ax=ax3)
        ax3.set_xlabel('Condition')
        ax3.set_ylabel('Commitment Amount ($)')
        ax3.set_title('Distribution of Commitment Amounts')
        ax3.set_xticklabels([c.replace('_', '\n').title() for c in self.data['condition'].unique()], rotation=45)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.0f}M'))
        
        # Subplot 4: Histogram of commitment amounts
        self.data['commitment_amount'].hist(bins=20, ax=ax4, alpha=0.7)
        ax4.set_xlabel('Commitment Amount ($)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of All Commitment Amounts')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.0f}M'))
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / 'escalation_results.png', dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_dir / 'escalation_results.png'}")
        
        plt.show()
        
        # Figure 2: Allocation patterns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Allocation scatter plot
        colors = {'high_responsibility_positive': 'blue', 'high_responsibility_negative': 'red',
                 'low_responsibility_positive': 'green', 'low_responsibility_negative': 'orange'}
        
        for condition in self.data['condition'].unique():
            cond_data = self.data[self.data['condition'] == condition]
            ax1.scatter(cond_data['consumer_allocation'], cond_data['industrial_allocation'], 
                       c=colors[condition], label=condition.replace('_', ' ').title(), alpha=0.6)
        
        ax1.plot([0, 20000000], [20000000, 0], 'k--', alpha=0.5, label='Total = $20M')
        ax1.set_xlabel('Consumer Allocation ($)')
        ax1.set_ylabel('Industrial Allocation ($)')
        ax1.set_title('Allocation Patterns by Condition')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.0f}M'))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.0f}M'))
        
        # Commitment distribution by condition
        for i, condition in enumerate(self.data['condition'].unique()):
            cond_data = self.data[self.data['condition'] == condition]['commitment_amount']
            ax2.hist(cond_data, bins=15, alpha=0.5, label=condition.replace('_', ' ').title())
        
        ax2.set_xlabel('Commitment Amount ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Commitment Amount Distributions')
        ax2.legend()
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.0f}M'))
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / 'allocation_patterns.png', dpi=300, bbox_inches='tight')
            print(f"Allocation patterns saved to {save_dir / 'allocation_patterns.png'}")
        
        plt.show()
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive analysis report"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"analysis_report_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            # Redirect print statements to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            print("ESCALATION OF COMMITMENT EXPERIMENT ANALYSIS REPORT")
            print("=" * 60)
            print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Data file: {self.results_dir}")
            print(f"Total subjects: {len(self.data)}")
            print()
            
            # Run all analyses
            self.descriptive_stats()
            self.main_effects_analysis()
            self.pairwise_comparisons()
            self.effect_sizes()
            self.choice_analysis()
            self.allocation_patterns()
            
            # Restore stdout
            sys.stdout = original_stdout
        
        print(f"Analysis report saved to: {output_file}")
        return output_file
    
    def run_full_analysis(self, create_plots=True, save_plots=True):
        """Run complete analysis pipeline"""
        print("Starting comprehensive analysis...")
        
        # Run statistical analyses
        desc_stats = self.descriptive_stats()
        main_effects = self.main_effects_analysis()
        pairwise = self.pairwise_comparisons()
        effect_sizes = self.effect_sizes()
        self.choice_analysis()
        self.allocation_patterns()
        
        # Create visualizations
        if create_plots:
            save_dir = self.results_dir / "plots" if save_plots else None
            self.create_visualizations(save_dir)
        
        # Generate report
        report_file = self.generate_report()
        
        print(f"\nAnalysis complete! Report saved to: {report_file}")
        
        return {
            'descriptive_stats': desc_stats,
            'main_effects': main_effects,
            'pairwise_comparisons': pairwise,
            'effect_sizes': effect_sizes
        }

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Analyze Escalation of Commitment experiment results')
    parser.add_argument('--results_dir', default='escalation_results', 
                       help='Directory containing result files')
    parser.add_argument('--file', help='Specific file to analyze (optional)')
    parser.add_argument('--no_plots', action='store_true', 
                       help='Skip creating visualizations')
    parser.add_argument('--no_save_plots', action='store_true',
                       help='Don\'t save plots to file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EscalationAnalyzer(args.results_dir)
    
    try:
        # Load data
        analyzer.load_data(args.file)
        
        # Run analysis
        results = analyzer.run_full_analysis(
            create_plots=not args.no_plots,
            save_plots=not args.no_save_plots
        )
        
        print("Analysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the results directory exists and contains data files.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()