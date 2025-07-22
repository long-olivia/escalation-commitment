import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import os

class EscalationAnalysis:
    """
    Analyzes escalation of commitment from experimental data.
    
    Escalation of commitment occurs when people continue investing in a 
    previously chosen alternative despite negative feedback, especially 
    when they feel personally responsible for the initial decision.
    """
    
    def __init__(self, json_file_path: str = None, data: List[Dict] = None):
        """Initialize with either a JSON file path or data directly."""
        if json_file_path:
            with open(json_file_path, 'r') as f:
                self.data = json.load(f)
            # Store the filename for use in visualizations
            self.filename = os.path.basename(json_file_path)
        elif data:
            self.data = data
            self.filename = "Direct Data Input"
        else:
            raise ValueError("Must provide either json_file_path or data")
        
        self.df = pd.DataFrame(self.data)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and clean the data for analysis."""
        # Convert monetary values to millions for easier interpretation
        monetary_cols = ['consumer_allocation', 'industrial_allocation', 'commitment_amount']
        for col in monetary_cols:
            self.df[col] = self.df[col] / 1_000_000
        
        # Create a unified 'previous_choice' column
        self.df['previous_choice'] = self.df['initial_choice'].fillna(self.df['assigned_previous_choice'])
        
        # Calculate commitment to previously chosen alternative
        self.df['commitment_to_previous'] = self.df.apply(self._calculate_commitment, axis=1)
        
        # Create categorical variables for analysis
        self.df['responsibility'] = pd.Categorical(self.df['responsibility_condition'], 
                                                 categories=['low', 'high'], ordered=True)
        self.df['outcome'] = pd.Categorical(self.df['outcome_condition'], 
                                          categories=['positive', 'negative'], ordered=True)
    
    def _calculate_commitment(self, row):
        """Calculate how much money was allocated to the previously chosen alternative."""
        if row['previous_choice'] == 'consumer':
            return row['consumer_allocation']
        else:  # 'industrial'
            return row['industrial_allocation']
    
    def descriptive_statistics(self) -> pd.DataFrame:
        """Generate descriptive statistics by condition."""
        print("=== ESCALATION OF COMMITMENT ANALYSIS ===\n")
        print(f"Analyzing file: {self.filename}\n")
        print("Descriptive Statistics by Condition:")
        print("-" * 50)
        
        desc_stats = self.df.groupby(['responsibility_condition', 'outcome_condition']).agg({
            'commitment_to_previous': ['count', 'mean', 'std', 'min', 'max'],
            'commitment_amount': ['mean', 'std']
        }).round(3)
        
        print(desc_stats)
        return desc_stats
    
    def escalation_analysis(self) -> Dict:
        """
        Analyze escalation of commitment patterns.
        
        Key hypotheses:
        1. High responsibility + negative outcomes = highest escalation
        2. Low responsibility conditions show less escalation
        3. Positive outcomes should show less variation between conditions
        
        CORRECTED: Escalation means MORE investment after negative outcomes,
        not less investment.
        """
        print("\n=== ESCALATION PATTERNS ===")
        print("-" * 40)
        
        results = {}
        
        # Group means for each condition
        condition_means = self.df.groupby(['responsibility_condition', 'outcome_condition'])['commitment_to_previous'].agg(['mean', 'count', 'std'])
        
        print("Mean commitment to previously chosen alternative (in millions):")
        for (resp, outcome), stats in condition_means.iterrows():
            print(f"  {resp.title()} Responsibility + {outcome.title()} Outcome: "
                  f"${stats['mean']:.2f}M (n={stats['count']}, std=${stats['std']:.2f}M)")
        
        results['condition_means'] = condition_means
        
        # CORRECTED: Calculate escalation effect (negative - positive outcomes)
        # Positive values = escalation (more money after bad outcomes)
        # Negative values = rational behavior (less money after bad outcomes)
        high_resp_escalation = (condition_means.loc[('high', 'negative'), 'mean'] - 
                               condition_means.loc[('high', 'positive'), 'mean'])
        low_resp_escalation = (condition_means.loc[('low', 'negative'), 'mean'] - 
                              condition_means.loc[('low', 'positive'), 'mean'])
        
        print(f"\nEscalation Effects (Negative Outcome Investment - Positive Outcome Investment):")
        print(f"  High Responsibility: ${high_resp_escalation:.2f}M")
        print(f"  Low Responsibility: ${low_resp_escalation:.2f}M")
        print(f"  Responsibility Difference: ${high_resp_escalation - low_resp_escalation:.2f}M")
        
        print(f"\nInterpretation:")
        if high_resp_escalation > 0:
            print(f"  • High responsibility shows ESCALATION (${high_resp_escalation:.2f}M more after bad outcomes)")
        else:
            print(f"  • High responsibility shows RATIONAL behavior (${abs(high_resp_escalation):.2f}M less after bad outcomes)")
            
        if low_resp_escalation > 0:
            print(f"  • Low responsibility shows ESCALATION (${low_resp_escalation:.2f}M more after bad outcomes)")
        else:
            print(f"  • Low responsibility shows RATIONAL behavior (${abs(low_resp_escalation):.2f}M less after bad outcomes)")
        
        results['escalation_effects'] = {
            'high_responsibility': high_resp_escalation,
            'low_responsibility': low_resp_escalation,
            'difference': high_resp_escalation - low_resp_escalation
        }
        
        return results
    
    def statistical_tests(self) -> Dict:
        """Perform statistical tests for escalation of commitment."""
        print("\n=== STATISTICAL TESTS ===")
        print("-" * 30)
        
        results = {}
        
        # 2-way ANOVA
        from scipy.stats import f_oneway
        
        # Separate groups for ANOVA
        high_pos = self.df[(self.df['responsibility_condition'] == 'high') & 
                          (self.df['outcome_condition'] == 'positive')]['commitment_to_previous']
        high_neg = self.df[(self.df['responsibility_condition'] == 'high') & 
                          (self.df['outcome_condition'] == 'negative')]['commitment_to_previous']
        low_pos = self.df[(self.df['responsibility_condition'] == 'low') & 
                         (self.df['outcome_condition'] == 'positive')]['commitment_to_previous']
        low_neg = self.df[(self.df['responsibility_condition'] == 'low') & 
                         (self.df['outcome_condition'] == 'negative')]['commitment_to_previous']
        
        # Overall F-test
        f_stat, p_value = f_oneway(high_pos, high_neg, low_pos, low_neg)
        print(f"Overall F-test: F = {f_stat:.3f}, p = {p_value:.3f}")
        results['overall_f_test'] = {'f_stat': f_stat, 'p_value': p_value}
        
        # Specific comparisons for escalation of commitment
        # High responsibility: negative vs positive (main escalation effect)
        high_t_stat, high_p = stats.ttest_ind(high_neg, high_pos)
        print(f"\nHigh Responsibility (Negative vs Positive): t = {high_t_stat:.3f}, p = {high_p:.3f}")
        
        # Low responsibility: negative vs positive
        low_t_stat, low_p = stats.ttest_ind(low_neg, low_pos)
        print(f"Low Responsibility (Negative vs Positive): t = {low_t_stat:.3f}, p = {low_p:.3f}")
        
        # Interaction effect: compare escalation between responsibility conditions
        high_escalation_scores = high_neg.values - high_pos.mean()
        low_escalation_scores = low_neg.values - low_pos.mean()
        
        interaction_t, interaction_p = stats.ttest_ind(high_escalation_scores, low_escalation_scores)
        print(f"\nResponsibility × Outcome Interaction: t = {interaction_t:.3f}, p = {interaction_p:.3f}")
        
        results['pairwise_tests'] = {
            'high_responsibility': {'t_stat': high_t_stat, 'p_value': high_p},
            'low_responsibility': {'t_stat': low_t_stat, 'p_value': low_p},
            'interaction': {'t_stat': interaction_t, 'p_value': interaction_p}
        }
        
        return results
    
    def create_visualization(self, save_path: str = None, auto_save: bool = True):
        """Create the main interaction plot showing escalation of commitment patterns."""
        plt.figure(figsize=(10, 8))
        
        # Main interaction plot (replicating your figure)
        condition_means = self.df.groupby(['responsibility_condition', 'outcome_condition'])['commitment_to_previous'].mean()
        
        outcomes = ['positive', 'negative']
        high_resp_means = [condition_means[('high', outcome)] for outcome in outcomes]
        low_resp_means = [condition_means[('low', outcome)] for outcome in outcomes]
        
        plt.plot(outcomes, high_resp_means, 'o-', linewidth=3, markersize=10, 
                label='High Personal\nResponsibility Conditions', color='red')
        plt.plot(outcomes, low_resp_means, 'o-', linewidth=3, markersize=10, 
                label='Low Personal\nResponsibility Conditions', color='blue')
        
        plt.ylabel('Amount of Money Allocated to\nPreviously Chosen Alternative\n(in millions)', fontsize=12)
        plt.xlabel('Decision Consequences', fontsize=12)
        
        # Updated title to include the filename
        title_main = 'Amount of money allocated to previously chosen alternative by personal\nresponsibility and decision consequences'
        title_with_file = f'{title_main}\n(Data from: {self.filename})'
        plt.title(title_with_file, fontsize=14, pad=20)
        
        # Position legend like in your figure
        plt.legend(bbox_to_anchor=(0.7, 0.8), fontsize=11)
        
        # Set y-axis limits based on the actual data range with some padding
        all_means = high_resp_means + low_resp_means
        y_min = min(all_means) * 0.95  # 5% padding below
        y_max = max(all_means) * 1.05  # 5% padding above
        plt.ylim(y_min, y_max)
        
        # Clean up the plot
        plt.tight_layout()
        
        # Save the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {save_path}")
        elif auto_save:
            # Auto-generate filename based on the data source
            auto_filename = f"escalation_plot_{self.filename.replace('.json', '.png')}"
            plt.savefig(auto_filename, dpi=300, bbox_inches='tight')
            print(f"Plot automatically saved as: {auto_filename}")
        
        plt.show()
    
    def interpret_results(self, results: Dict):
        """
        CORRECTED: Provide accurate interpretation of escalation of commitment findings.
        
        Escalation = MORE investment after negative outcomes (positive difference)
        Rational behavior = LESS investment after negative outcomes (negative difference)
        """
        print("\n=== INTERPRETATION ===")
        print("-" * 25)
        
        high_escalation = results['escalation_effects']['high_responsibility']
        low_escalation = results['escalation_effects']['low_responsibility']
        escalation_diff = results['escalation_effects']['difference']
        
        # Check if there's any actual escalation (positive values)
        has_escalation = high_escalation > 0 or low_escalation > 0
        
        if has_escalation:
            print("⚠️ ESCALATION OF COMMITMENT DETECTED")
            if high_escalation > 0:
                print(f"  • High responsibility participants showed escalation: ${high_escalation:.2f}M more after negative outcomes")
            if low_escalation > 0:
                print(f"  • Low responsibility participants showed escalation: ${low_escalation:.2f}M more after negative outcomes")
            
            if escalation_diff > 0:
                print(f"  • High responsibility showed ${escalation_diff:.2f}M more escalation than low responsibility")
            elif escalation_diff < 0:
                print(f"  • Low responsibility showed ${abs(escalation_diff):.2f}M more escalation than high responsibility")
        else:
            print("✓ NO ESCALATION OF COMMITMENT DETECTED")
            print("  Participants showed RATIONAL DECISION-MAKING")
            print(f"  • High responsibility: ${abs(high_escalation):.2f}M LESS after negative outcomes")
            print(f"  • Low responsibility: ${abs(low_escalation):.2f}M LESS after negative outcomes")
        
        print(f"\nKey Findings:")
        print(f"  • High responsibility effect: ${high_escalation:.2f}M {'(escalation)' if high_escalation > 0 else '(rational)'}")
        print(f"  • Low responsibility effect: ${low_escalation:.2f}M {'(escalation)' if low_escalation > 0 else '(rational)'}")
        
        if has_escalation:
            print(f"\nThis supports the escalation of commitment theory,")
            print("which predicts that people will throw good money after bad when they")
            print("feel personally responsible for the initial decision, especially after negative feedback.")
        else:
            print(f"\nThis does NOT support the escalation of commitment theory.")
            print("Instead, participants showed rational behavior by investing LESS money")
            print("in alternatives that performed poorly, regardless of their responsibility level.")
            print("This suggests good decision-making rather than the escalation bias.")

def main():
    """Run the complete escalation of commitment analysis."""
    # Sample data from your JSON
    sample_data = [
        {
            "subject_id": 1, "responsibility_condition": "high", "outcome_condition": "positive",
            "initial_choice": "consumer", "consumer_allocation": 14000000, "industrial_allocation": 6000000,
            "commitment_amount": 14000000, "total_allocated": 20000000
        },
        # ... (rest of your data would go here)
    ]
    
    try:
        # Initialize analyzer with the Staw replication data
        analyzer = EscalationAnalysis(json_file_path='staw_replication_results_v3(exact).json') # Update this path as needed to reflect which JSON file you want to analyze

        # Run complete analysis
        desc_stats = analyzer.descriptive_statistics()
        escalation_results = analyzer.escalation_analysis()
        statistical_results = analyzer.statistical_tests()
        
        # Create visualizations (and save them)
        save_filename = f"escalation_plot_{analyzer.filename.replace('.json', '.png')}"
        analyzer.create_visualization(save_path=save_filename)
        
        # Interpret findings
        analyzer.interpret_results(escalation_results)
        
        return {
            'descriptive_stats': desc_stats,
            'escalation_results': escalation_results,
            'statistical_results': statistical_results
        }
        
    except FileNotFoundError:
        print("JSON file not found. Please update the file path in the main() function.")
        print("Alternatively, you can pass your data directly to EscalationAnalysis(data=your_data)")

if __name__ == "__main__":
    results = main()