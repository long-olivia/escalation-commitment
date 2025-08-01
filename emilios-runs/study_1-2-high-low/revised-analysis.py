import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EscalationAnalysis:
    """
    Statistical analysis for escalation of commitment experiment.
    2x2 between-subjects design: Responsibility (High/Low) × Outcome (Positive/Negative)
    """
    
    def __init__(self, json_file_path):
        """Load and prepare data from JSON file."""
        self.json_file_path = json_file_path
        self.df = self.load_and_prepare_data()
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load JSON data and convert to structured DataFrame."""
        print("Loading data from JSON file...")
        
        with open(self.json_file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list of objects and single object formats
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])  # Single object case
        
        # Handle different condition naming formats
        if 'condition' in df.columns:
            # Extract responsibility and outcome from condition string
            df[['responsibility', 'outcome']] = df['condition'].str.extract(r'(\w+)_responsibility_(\w+)')
            # Fill any missing values by parsing the full condition string
            mask = df['responsibility'].isna()
            if mask.any():
                condition_parts = df.loc[mask, 'condition'].str.split('_')
                df.loc[mask, 'responsibility'] = condition_parts.str[0]
                df.loc[mask, 'outcome'] = condition_parts.str[-1]
        
        # Ensure we have the required columns
        if 'responsibility' not in df.columns or 'outcome' not in df.columns:
            raise ValueError("Could not extract 'responsibility' and 'outcome' from data. Please check JSON structure.")
        
        # Create combined condition variable first (before converting to categorical)
        df['condition_combined'] = df['responsibility'].astype(str) + '_' + df['outcome'].astype(str)
        
        # Now create categorical variables for analysis
        df['responsibility'] = df['responsibility'].astype('category')
        df['outcome'] = df['outcome'].astype('category')
        df['condition_combined'] = df['condition_combined'].astype('category')
        
        # Ensure commitment_amount is numeric and convert to millions if needed
        df['commitment_amount'] = pd.to_numeric(df['commitment_amount'])
        
        # Convert from dollars to millions if values are too large (> 1000)
        if df['commitment_amount'].mean() > 1000:
            df['commitment_amount'] = df['commitment_amount'] / 1_000_000
            print("Note: Converted commitment amounts from dollars to millions")
        
        print(f"Data loaded successfully: {len(df)} observations")
        print(f"Conditions: {df['condition_combined'].value_counts().to_dict()}")
        print(f"Sample commitment amounts: {df['commitment_amount'].head().tolist()}")
        
        # Verify data structure
        print(f"\nData structure verification:")
        print(f"- Responsibility levels: {df['responsibility'].unique()}")
        print(f"- Outcome levels: {df['outcome'].unique()}")
        print(f"- Commitment amount range: ${df['commitment_amount'].min():.2f}M - ${df['commitment_amount'].max():.2f}M")
        
        return df
    
    def descriptive_statistics(self):
        """Generate descriptive statistics."""
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        
        # Overall descriptives
        print(f"Total N: {len(self.df)}")
        print(f"Overall Mean Commitment: ${self.df['commitment_amount'].mean():,.2f}M")
        print(f"Overall SD: ${self.df['commitment_amount'].std():,.2f}M")
        
        # By condition
        desc_stats = self.df.groupby(['responsibility', 'outcome'])['commitment_amount'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print("\nDescriptive Statistics by Condition:")
        print(desc_stats)
        
        # Store for later use
        self.descriptives = desc_stats
        
        return desc_stats
    
    def two_way_anova(self):
        """Conduct 2x2 between-subjects ANOVA."""
        print("\n" + "="*60)
        print("TWO-WAY ANOVA")
        print("="*60)
        
        # Using pingouin for cleaner output and effect sizes
        anova_results = pg.anova(data=self.df, 
                                dv='commitment_amount',
                                between=['responsibility', 'outcome'],
                                detailed=True)
        
        print("ANOVA Results:")
        print(anova_results.round(4))
        
        # Store results
        self.results['anova'] = anova_results
        
        # Interpret results
        print("\nInterpretation:")
        for idx, row in anova_results.iterrows():
            source = row['Source']
            p_val = row['p-unc']
            eta_sq = row['np2']  # partial eta squared
            
            sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            effect_size = "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small"
            
            print(f"  {source}: F = {row['F']:.3f}, p = {p_val:.4f} {sig_level}, η²p = {eta_sq:.4f} ({effect_size} effect)")
        
        return anova_results
    
    def simple_main_effects(self):
        """Conduct simple main effects analysis if interaction is significant."""
        print("\n" + "="*60)
        print("SIMPLE MAIN EFFECTS ANALYSIS")
        print("="*60)
        
        # Check if interaction is significant
        interaction_p = self.results['anova'][self.results['anova']['Source'] == 'responsibility * outcome']['p-unc'].iloc[0]
        
        if interaction_p < 0.05:
            print("Interaction is significant (p < 0.05). Conducting simple main effects...")
            
            # Effect of outcome at each level of responsibility
            print("\nEffect of Outcome at each level of Responsibility:")
            
            for resp_level in ['high', 'low']:
                subset = self.df[self.df['responsibility'] == resp_level]
                pos_group = subset[subset['outcome'] == 'positive']['commitment_amount']
                neg_group = subset[subset['outcome'] == 'negative']['commitment_amount']
                
                t_stat, p_val = ttest_ind(neg_group, pos_group)
                cohens_d = self.cohens_d(neg_group, pos_group)
                
                sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                print(f"  {resp_level.title()} Responsibility: t = {t_stat:.3f}, p = {p_val:.4f} {sig_level}, d = {cohens_d:.3f}")
                print(f"    Negative M = ${neg_group.mean():.2f}M, Positive M = ${pos_group.mean():.2f}M")
            
            # Effect of responsibility at each level of outcome
            print("\nEffect of Responsibility at each level of Outcome:")
            
            for outcome_level in ['positive', 'negative']:
                subset = self.df[self.df['outcome'] == outcome_level]
                high_group = subset[subset['responsibility'] == 'high']['commitment_amount']
                low_group = subset[subset['responsibility'] == 'low']['commitment_amount']
                
                t_stat, p_val = ttest_ind(high_group, low_group)
                cohens_d = self.cohens_d(high_group, low_group)
                
                sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                print(f"  {outcome_level.title()} Outcome: t = {t_stat:.3f}, p = {p_val:.4f} {sig_level}, d = {cohens_d:.3f}")
                print(f"    High M = ${high_group.mean():.2f}M, Low M = ${low_group.mean():.2f}M")
                
        else:
            print(f"Interaction is not significant (p = {interaction_p:.4f}). Simple main effects not warranted.")
    
    def pairwise_comparisons(self):
        """Conduct all pairwise comparisons with multiple comparison correction."""
        print("\n" + "="*60)
        print("PAIRWISE COMPARISONS")
        print("="*60)
        
        # Get all four groups
        conditions = ['high_positive', 'high_negative', 'low_positive', 'low_negative']
        groups = {}
        
        for condition in conditions:
            resp, outcome = condition.split('_')
            groups[condition] = self.df[
                (self.df['responsibility'] == resp) & 
                (self.df['outcome'] == outcome)
            ]['commitment_amount']
        
        # Conduct all pairwise t-tests
        comparisons = []
        t_stats = []
        p_values = []
        effect_sizes = []
        
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions):
                if i < j:  # Only unique pairs
                    t_stat, p_val = ttest_ind(groups[cond1], groups[cond2])
                    d = self.cohens_d(groups[cond1], groups[cond2])
                    
                    comparisons.append(f"{cond1} vs {cond2}")
                    t_stats.append(t_stat)
                    p_values.append(p_val)
                    effect_sizes.append(d)
        
        # Apply multiple comparison correction
        rejected_bonf, p_bonf, _, _ = multipletests(p_values, method='bonferroni')
        rejected_holm, p_holm, _, _ = multipletests(p_values, method='holm')
        
        # Create results DataFrame
        pairwise_results = pd.DataFrame({
            'Comparison': comparisons,
            't-statistic': t_stats,
            'p-value': p_values,
            'p-bonferroni': p_bonf,
            'p-holm': p_holm,
            'sig-bonferroni': rejected_bonf,
            'sig-holm': rejected_holm,
            'Cohens-d': effect_sizes
        })
        
        print("Pairwise Comparisons (with multiple comparison corrections):")
        print(pairwise_results.round(4))
        
        # Store results
        self.results['pairwise'] = pairwise_results
        
        # Highlight significant comparisons
        print("\nSignificant comparisons (Bonferroni corrected):")
        sig_comparisons = pairwise_results[pairwise_results['sig-bonferroni']]
        if len(sig_comparisons) > 0:
            for _, row in sig_comparisons.iterrows():
                print(f"  {row['Comparison']}: t = {row['t-statistic']:.3f}, "
                      f"p_bonf = {row['p-bonferroni']:.4f}, d = {row['Cohens-d']:.3f}")
        else:
            print("  No significant pairwise differences after Bonferroni correction.")
        
        return pairwise_results
    
    def escalation_analysis(self):
        """Analyze escalation of commitment effects specifically."""
        print("\n" + "="*60)
        print("ESCALATION OF COMMITMENT ANALYSIS")
        print("="*60)
        
        # Key comparison: High responsibility, negative vs positive outcomes
        high_neg = self.df[(self.df['responsibility'] == 'high') & 
                          (self.df['outcome'] == 'negative')]['commitment_amount']
        high_pos = self.df[(self.df['responsibility'] == 'high') & 
                          (self.df['outcome'] == 'positive')]['commitment_amount']
        
        escalation_effect = high_neg.mean() - high_pos.mean()
        t_stat, p_val = ttest_ind(high_neg, high_pos)
        cohens_d = self.cohens_d(high_neg, high_pos)
        
        print("PRIMARY ESCALATION EFFECT (High Responsibility):")
        print(f"  Negative Outcome M = ${high_neg.mean():.2f}M (SD = ${high_neg.std():.2f}M)")
        print(f"  Positive Outcome M = ${high_pos.mean():.2f}M (SD = ${high_pos.std():.2f}M)")
        print(f"  Difference = ${escalation_effect:.2f}M")
        print(f"  t({len(high_neg) + len(high_pos) - 2}) = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Cohen's d = {cohens_d:.3f}")
        
        sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  Result: {sig_level}")
        
        if escalation_effect > 0:
            print("  ✓ ESCALATION EFFECT CONFIRMED: Higher investment after negative outcomes")
        else:
            print("  ✗ No escalation effect: Lower investment after negative outcomes")
        
        # Compare to low responsibility condition
        low_neg = self.df[(self.df['responsibility'] == 'low') & 
                         (self.df['outcome'] == 'negative')]['commitment_amount']
        low_pos = self.df[(self.df['responsibility'] == 'low') & 
                         (self.df['outcome'] == 'positive')]['commitment_amount']
        
        low_effect = low_neg.mean() - low_pos.mean()
        
        print(f"\nCOMPARISON - Low Responsibility Effect:")
        print(f"  Difference = ${low_effect:.2f}M")
        print(f"  High Responsibility Effect = ${escalation_effect:.2f}M")
        print(f"  Difference between conditions = ${escalation_effect - low_effect:.2f}M")
        
        if escalation_effect > low_effect:
            print("  ✓ RESPONSIBILITY MODERATION CONFIRMED: Stronger escalation under high responsibility")
        else:
            print("  ✗ No responsibility moderation effect")
    
    def cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        return (group1.mean() - group2.mean()) / pooled_std
    
    def create_visualizations(self):
        """Create publication-ready visualizations."""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Escalation of Commitment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Main interaction plot
        ax1 = axes[0, 0]
        
        # Calculate means and SEs for each condition
        summary_stats = self.df.groupby(['responsibility', 'outcome'])['commitment_amount'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        summary_stats['se'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
        
        # Pivot for easier plotting
        pivot_means = summary_stats.pivot(index='responsibility', columns='outcome', values='mean')
        pivot_ses = summary_stats.pivot(index='responsibility', columns='outcome', values='se')
        
        x_pos = np.arange(len(pivot_means.index))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, pivot_means['negative'], width, 
                       yerr=pivot_ses['negative'], label='Negative Outcome', 
                       alpha=0.8, capsize=5)
        bars2 = ax1.bar(x_pos + width/2, pivot_means['positive'], width,
                       yerr=pivot_ses['positive'], label='Positive Outcome', 
                       alpha=0.8, capsize=5)
        
        ax1.set_xlabel('Responsibility Level')
        ax1.set_ylabel('Investment Amount (Millions $)')
        ax1.set_title('Mean Investment by Responsibility and Outcome')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['High', 'Low'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'${height:.1f}M', ha='center', va='bottom', fontsize=9)
        
        # 2. Box plot
        ax2 = axes[0, 1]
        sns.boxplot(data=self.df, x='responsibility', y='commitment_amount', 
                   hue='outcome', ax=ax2)
        ax2.set_title('Distribution of Investment Amounts')
        ax2.set_ylabel('Investment Amount (Millions $)')
        ax2.set_xlabel('Responsibility Level')
        
        # 3. Individual data points with jitter
        ax3 = axes[1, 0]
        sns.stripplot(data=self.df, x='responsibility', y='commitment_amount', 
                     hue='outcome', dodge=True, alpha=0.6, ax=ax3)
        ax3.set_title('Individual Data Points')
        ax3.set_ylabel('Investment Amount (Millions $)')
        ax3.set_xlabel('Responsibility Level')
        
        # 4. Effect sizes visualization
        ax4 = axes[1, 1]
        
        # Calculate key effect sizes
        effects = {
            'High Resp:\nNeg vs Pos': self.cohens_d(
                self.df[(self.df['responsibility'] == 'high') & (self.df['outcome'] == 'negative')]['commitment_amount'],
                self.df[(self.df['responsibility'] == 'high') & (self.df['outcome'] == 'positive')]['commitment_amount']
            ),
            'Low Resp:\nNeg vs Pos': self.cohens_d(
                self.df[(self.df['responsibility'] == 'low') & (self.df['outcome'] == 'negative')]['commitment_amount'],
                self.df[(self.df['responsibility'] == 'low') & (self.df['outcome'] == 'positive')]['commitment_amount']
            ),
            'Neg Outcome:\nHigh vs Low': self.cohens_d(
                self.df[(self.df['responsibility'] == 'high') & (self.df['outcome'] == 'negative')]['commitment_amount'],
                self.df[(self.df['responsibility'] == 'low') & (self.df['outcome'] == 'negative')]['commitment_amount']
            ),
            'Pos Outcome:\nHigh vs Low': self.cohens_d(
                self.df[(self.df['responsibility'] == 'high') & (self.df['outcome'] == 'positive')]['commitment_amount'],
                self.df[(self.df['responsibility'] == 'low') & (self.df['outcome'] == 'positive')]['commitment_amount']
            )
        }
        
        comparison_names = list(effects.keys())
        effect_sizes = list(effects.values())
        colors = ['red' if d > 0 else 'blue' for d in effect_sizes]
        
        bars = ax4.bar(comparison_names, effect_sizes, color=colors, alpha=0.7)
        ax4.set_title("Effect Sizes (Cohen's d)")
        ax4.set_ylabel("Cohen's d")
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
        ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.9, label='Large effect')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, d in zip(bars, effect_sizes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.1),
                    f'{d:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(self.json_file_path).parent / 'escalation_analysis_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {output_path}")
        
        plt.show()
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive statistical report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL REPORT")
        print("="*80)
        
        print("\nSTUDY DESIGN:")
        print("- 2×2 between-subjects factorial design")
        print("- Independent Variables: Responsibility (High/Low) × Outcome (Positive/Negative)")
        print("- Dependent Variable: Investment allocation to originally funded division (millions $)")
        print(f"- Sample Size: N = {len(self.df)} (n ≈ {len(self.df)//4} per condition)")
        
        # Key findings summary
        anova_results = self.results['anova']
        
        main_resp_p = anova_results[anova_results['Source'] == 'responsibility']['p-unc'].iloc[0]
        main_outcome_p = anova_results[anova_results['Source'] == 'outcome']['p-unc'].iloc[0]
        interaction_p = anova_results[anova_results['Source'] == 'responsibility * outcome']['p-unc'].iloc[0]
        
        print(f"\nKEY STATISTICAL RESULTS:")
        print(f"- Main effect of Responsibility: p = {main_resp_p:.4f}")
        print(f"- Main effect of Outcome: p = {main_outcome_p:.4f}")
        print(f"- Interaction effect: p = {interaction_p:.4f}")
        
        # Practical conclusions
        print(f"\nPRACTICAL CONCLUSIONS:")
        
        high_neg_mean = self.df[(self.df['responsibility'] == 'high') & 
                               (self.df['outcome'] == 'negative')]['commitment_amount'].mean()
        high_pos_mean = self.df[(self.df['responsibility'] == 'high') & 
                               (self.df['outcome'] == 'positive')]['commitment_amount'].mean()
        
        escalation_effect = high_neg_mean - high_pos_mean
        
        if escalation_effect > 0 and interaction_p < 0.05:
            print("✓ ESCALATION OF COMMITMENT CONFIRMED:")
            print(f"  - LLMs show escalation behavior similar to humans")
            print(f"  - Under high responsibility, negative outcomes lead to ${escalation_effect:.2f}M more investment")
            print(f"  - This supports the psychological theory of escalation of commitment")
        elif escalation_effect > 0:
            print("? PARTIAL ESCALATION EVIDENCE:")
            print(f"  - Escalation pattern observed (${escalation_effect:.2f}M difference)")
            print(f"  - But interaction not statistically significant (p = {interaction_p:.4f})")
        else:
            print("✗ NO ESCALATION OF COMMITMENT:")
            print(f"  - LLMs do not show expected escalation behavior")
            print(f"  - Negative outcomes led to less investment (${escalation_effect:.2f}M)")
    
    def run_full_analysis(self):
        """Run the complete statistical analysis."""
        print("Starting comprehensive escalation of commitment analysis...")
        
        # Run all analyses
        self.descriptive_statistics()
        self.two_way_anova()
        self.simple_main_effects()
        self.pairwise_comparisons()
        self.escalation_analysis()
        self.create_visualizations()
        self.generate_report()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        return self.results

def main():
    """Main function to run the analysis."""
    # You'll need to update this path to your JSON file
    json_file_path = input("Enter the path to your JSON results file: ").strip()
    
    # Remove quotes if user included them
    json_file_path = json_file_path.strip('"\'')
    
    # Alternatively, set a default path:
    # json_file_path = "/path/to/your/all_results_o4-mini-2025-04-16.json"
    
    try:
        # Initialize and run analysis
        analyzer = EscalationAnalysis(json_file_path)
        results = analyzer.run_full_analysis()
        
        # Optionally save results to file
        output_dir = Path(json_file_path).parent
        results_file = output_dir / 'statistical_analysis_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
                
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file at {json_file_path}")
        print("Please check the file path and try again.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file: {e}")
        print("Please check that your JSON file is properly formatted.")
    except ValueError as e:
        print(f"Error: Data structure issue: {e}")
        print("Please check that your data contains the required fields.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        print("Please check your data file and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()