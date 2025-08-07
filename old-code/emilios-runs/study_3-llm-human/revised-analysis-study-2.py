import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

class Study2Analyzer:
    """
    Statistical analysis for Study 2: Advisory Role
    Escalation of Commitment Experiment with LLMs
    """
    
    def __init__(self, json_file_path):
        """
        Initialize analyzer with JSON data file
        
        Args:
            json_file_path (str): Path to the JSON results file
        """
        self.json_file_path = json_file_path
        self.data = None
        self.df = None
        
    def load_data(self):
        """Load and parse JSON data"""
        print("Loading JSON data...")
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Extract trials data
            trials = self.data.get('trials', [])
            if not trials:
                raise ValueError("No trials data found in JSON file")
            
            print(f"✅ Loaded {len(trials)} trials")
            return trials
            
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
    
    def preprocess_data(self):
        """
        Preprocess and encode experimental data
        """
        print("\nPreprocessing data...")
        
        trials = self.load_data()
        
        # Convert to DataFrame
        self.df = pd.DataFrame(trials)
        
        # Parse condition into outcome valence and investment plan
        def parse_condition(condition):
            """Parse condition string into outcome valence and investment plan"""
            condition_lower = condition.lower()
            
            # Determine outcome valence
            if 'success' in condition_lower:
                outcome_valence = 'Positive'
            elif 'failure' in condition_lower:
                outcome_valence = 'Negative'
            else:
                outcome_valence = 'Unknown'
            
            # Determine investment plan (what the VP is proposing)
            if 'continue' in condition_lower or 'escalation' in condition_lower:
                investment_plan = 'Escalation'
            elif 'pivot' in condition_lower or 'rational' in condition_lower:
                investment_plan = 'Rational'
            else:
                investment_plan = 'Unknown'
            
            return outcome_valence, investment_plan
        
        # Apply parsing
        parsed_conditions = self.df['condition'].apply(parse_condition)
        self.df['outcome_valence'] = [x[0] for x in parsed_conditions]
        self.df['investment_plan'] = [x[1] for x in parsed_conditions]
        
        # Create binary escalation support variable
        # supports_vp_proposal indicates whether model supports VP's proposal
        # When VP proposes escalation, supporting = escalation
        # When VP proposes rational, supporting = rational (not escalation)
        def determine_escalation_support(row):
            """Determine if model supports escalation based on VP proposal and model support"""
            if pd.isna(row['supports_vp_proposal']):
                return np.nan
            
            # If VP proposed escalation and model supports, then model supports escalation
            # If VP proposed rational and model supports, then model supports rational (not escalation)
            # If VP proposed escalation and model doesn't support, then model opposes escalation
            # If VP proposed rational and model doesn't support, then model supports escalation
            
            if row['investment_plan'] == 'Escalation':
                # VP proposed escalation (continue/escalation conditions)
                return row['supports_vp_proposal']  # True = supports escalation, False = opposes escalation
            else:
                # VP proposed rational reallocation (pivot/rational conditions)
                return not row['supports_vp_proposal']  # True = opposes rational (supports escalation), False = supports rational
        
        self.df['supports_escalation'] = self.df.apply(determine_escalation_support, axis=1)
        
        # Create numeric encoding for ANOVA
        self.df['outcome_valence_numeric'] = self.df['outcome_valence'].map({'Positive': 1, 'Negative': 0})
        self.df['investment_plan_numeric'] = self.df['investment_plan'].map({'Escalation': 1, 'Rational': 0})
        self.df['supports_escalation_numeric'] = self.df['supports_escalation'].astype(float)
        
        # Remove rows with missing key variables
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['supports_escalation', 'outcome_valence', 'investment_plan'])
        final_count = len(self.df)
        
        if initial_count != final_count:
            print(f"⚠️  Removed {initial_count - final_count} trials with missing data")
        
        # Print summary of conditions and their interpretation
        print("\nCondition Summary:")
        condition_summary = self.df.groupby(['outcome_valence', 'investment_plan']).size().unstack(fill_value=0)
        print(condition_summary)
        
        print(f"\nCondition Interpretation:")
        print(f"- success_continue: Positive outcome, VP proposes escalation (continue same strategy)")
        print(f"- success_pivot: Positive outcome, VP proposes rational shift")  
        print(f"- failure_rational: Negative outcome, VP proposes rational shift")
        print(f"- failure_escalation: Negative outcome, VP proposes escalation (KEY CONDITION)")
        
        print(f"\nEscalation Support Logic:")
        print(f"- When VP proposes escalation → model support = escalation support")
        print(f"- When VP proposes rational → model support = rational support (NOT escalation)")
        
        # Show a few examples
        print(f"\nSample interpretations:")
        for condition in self.df['condition'].unique():
            sample = self.df[self.df['condition'] == condition].iloc[0]
            escalation_meaning = "supports escalation" if sample['supports_escalation'] else "opposes escalation"
            print(f"- {condition}: VP proposes {sample['investment_plan']}, model {'supports' if sample['supports_vp_proposal'] else 'opposes'} → model {escalation_meaning}")
        
        print(f"\n✅ Preprocessed {len(self.df)} trials")
        
        return self.df
    
    def check_assumptions(self):
        """Check assumptions for statistical tests"""
        print("\n" + "="*60)
        print("ASSUMPTION CHECKS")
        print("="*60)
        
        # Check sample sizes per condition
        print("\n1. Sample Size Check:")
        crosstab = pd.crosstab(self.df['outcome_valence'], self.df['investment_plan'], margins=True)
        print(crosstab)
        
        # Check for balance
        min_cell = crosstab.iloc[:-1, :-1].min().min()
        print(f"\nMinimum cell size: {min_cell}")
        if min_cell < 5:
            print("⚠️  Warning: Some cells have fewer than 5 observations")
        
        # Check distribution of dependent variable
        print(f"\n2. Dependent Variable Distribution:")
        print(f"Support for Escalation:")
        print(self.df['supports_escalation'].value_counts().sort_index())
        
        escalation_rate = self.df['supports_escalation'].mean()
        print(f"Overall escalation support rate: {escalation_rate:.2%}")
        
        return crosstab
    
    def run_two_way_anova(self):
        """
        Run two-way ANOVA on escalation support
        """
        print("\n" + "="*60)
        print("TWO-WAY ANOVA")
        print("="*60)
        
        # Prepare data for ANOVA (using binary outcome as continuous for ANOVA)
        # Note: This is appropriate for binary outcomes with sufficient sample size
        
        # Create formula for ANOVA
        formula = 'supports_escalation_numeric ~ C(outcome_valence) * C(investment_plan)'
        
        # Fit model
        model = ols(formula, data=self.df).fit()
        
        # Run ANOVA
        anova_results = anova_lm(model, typ=2)
        print("\nTwo-Way ANOVA Results:")
        print(anova_results)
        
        # Extract effect sizes (eta-squared)
        ss_total = anova_results['sum_sq'].sum()
        anova_results['eta_squared'] = anova_results['sum_sq'] / ss_total
        
        print("\nEffect Sizes (η²):")
        for factor in anova_results.index[:-1]:  # Exclude residual
            eta_sq = anova_results.loc[factor, 'eta_squared']
            print(f"{factor}: η² = {eta_sq:.4f}")
        
        # Store results for simple effects
        self.anova_model = model
        self.anova_results = anova_results
        
        return anova_results
    
    def run_simple_effects(self):
        """
        Run simple effects tests to unpack interactions
        """
        print("\n" + "="*60)
        print("SIMPLE EFFECTS TESTS")
        print("="*60)
        
        # Check if interaction is significant
        interaction_p = self.anova_results.loc['C(outcome_valence):C(investment_plan)', 'PR(>F)']
        
        if interaction_p < 0.05:
            print(f"Significant interaction found (p = {interaction_p:.4f})")
            print("Running simple effects tests...\n")
        else:
            print(f"No significant interaction (p = {interaction_p:.4f})")
            print("Simple effects tests may not be necessary, but running for completeness...\n")
        
        # Simple effects of Outcome Valence within each Investment Plan
        print("1. Effect of Outcome Valence within each Investment Plan:")
        
        simple_effects_results = []
        
        for plan in ['Escalation', 'Rational']:
            subset = self.df[self.df['investment_plan'] == plan]
            
            if len(subset) > 0:
                # T-test comparing positive vs negative outcomes within this investment plan
                pos_group = subset[subset['outcome_valence'] == 'Positive']['supports_escalation_numeric']
                neg_group = subset[subset['outcome_valence'] == 'Negative']['supports_escalation_numeric']
                
                if len(pos_group) > 0 and len(neg_group) > 0:
                    t_stat, p_val = stats.ttest_ind(pos_group, neg_group)
                    
                    pos_mean = pos_group.mean()
                    neg_mean = neg_group.mean()
                    
                    print(f"   {plan} Plan:")
                    print(f"      Positive outcome: M = {pos_mean:.3f} (n = {len(pos_group)})")
                    print(f"      Negative outcome: M = {neg_mean:.3f} (n = {len(neg_group)})")
                    print(f"      t({len(pos_group) + len(neg_group) - 2}) = {t_stat:.3f}, p = {p_val:.4f}")
                    
                    simple_effects_results.append({
                        'test': f'Outcome Valence within {plan}',
                        't_stat': t_stat,
                        'p_value': p_val,
                        'positive_mean': pos_mean,
                        'negative_mean': neg_mean
                    })
        
        print("\n2. Effect of Investment Plan within each Outcome Valence:")
        
        for outcome in ['Positive', 'Negative']:
            subset = self.df[self.df['outcome_valence'] == outcome]
            
            if len(subset) > 0:
                # T-test comparing escalation vs rational plans within this outcome
                esc_group = subset[subset['investment_plan'] == 'Escalation']['supports_escalation_numeric']
                rat_group = subset[subset['investment_plan'] == 'Rational']['supports_escalation_numeric']
                
                if len(esc_group) > 0 and len(rat_group) > 0:
                    t_stat, p_val = stats.ttest_ind(esc_group, rat_group)
                    
                    esc_mean = esc_group.mean()
                    rat_mean = rat_group.mean()
                    
                    print(f"   {outcome} Outcome:")
                    print(f"      Escalation plan: M = {esc_mean:.3f} (n = {len(esc_group)})")
                    print(f"      Rational plan: M = {rat_mean:.3f} (n = {len(rat_group)})")
                    print(f"      t({len(esc_group) + len(rat_group) - 2}) = {t_stat:.3f}, p = {p_val:.4f}")
                    
                    simple_effects_results.append({
                        'test': f'Investment Plan within {outcome}',
                        't_stat': t_stat,
                        'p_value': p_val,
                        'escalation_mean': esc_mean,
                        'rational_mean': rat_mean
                    })
        
        # Apply Bonferroni correction
        if simple_effects_results:
            p_values = [result['p_value'] for result in simple_effects_results]
            corrected_p = multipletests(p_values, method='bonferroni')[1]
            
            print(f"\n3. Bonferroni-corrected p-values:")
            for i, result in enumerate(simple_effects_results):
                print(f"   {result['test']}: p_corrected = {corrected_p[i]:.4f}")
        
        return simple_effects_results
    
    def run_chi_square_tests(self):
        """
        Run Chi-square tests for categorical analyses
        """
        print("\n" + "="*60)
        print("CHI-SQUARE TESTS")
        print("="*60)
        
        chi_square_results = {}
        
        # 1. Test relationship between Outcome Valence and Escalation Support
        print("1. Outcome Valence × Escalation Support:")
        
        contingency_outcome = pd.crosstab(
            self.df['outcome_valence'], 
            self.df['supports_escalation'],
            margins=True
        )
        print(contingency_outcome)
        
        # Remove margins for chi-square test
        contingency_clean = contingency_outcome.iloc[:-1, :-1]
        chi2_outcome, p_outcome, dof_outcome, expected_outcome = chi2_contingency(contingency_clean)
        
        print(f"\nχ²({dof_outcome}) = {chi2_outcome:.3f}, p = {p_outcome:.4f}")
        
        # Calculate Cramer's V (effect size)
        n = contingency_clean.sum().sum()
        cramers_v_outcome = np.sqrt(chi2_outcome / (n * (min(contingency_clean.shape) - 1)))
        print(f"Cramer's V = {cramers_v_outcome:.3f}")
        
        chi_square_results['outcome_valence'] = {
            'chi2': chi2_outcome,
            'p_value': p_outcome,
            'cramers_v': cramers_v_outcome,
            'contingency_table': contingency_outcome
        }
        
        # 2. Test relationship between Investment Plan and Escalation Support
        print("\n2. Investment Plan × Escalation Support:")
        
        contingency_plan = pd.crosstab(
            self.df['investment_plan'], 
            self.df['supports_escalation'],
            margins=True
        )
        print(contingency_plan)
        
        contingency_clean = contingency_plan.iloc[:-1, :-1]
        chi2_plan, p_plan, dof_plan, expected_plan = chi2_contingency(contingency_clean)
        
        print(f"\nχ²({dof_plan}) = {chi2_plan:.3f}, p = {p_plan:.4f}")
        
        cramers_v_plan = np.sqrt(chi2_plan / (n * (min(contingency_clean.shape) - 1)))
        print(f"Cramer's V = {cramers_v_plan:.3f}")
        
        chi_square_results['investment_plan'] = {
            'chi2': chi2_plan,
            'p_value': p_plan,
            'cramers_v': cramers_v_plan,
            'contingency_table': contingency_plan
        }
        
        # 3. Three-way contingency table for interaction
        print("\n3. Three-way Analysis (Outcome × Plan × Escalation):")
        
        # Create a more detailed contingency table
        detailed_contingency = pd.crosstab(
            [self.df['outcome_valence'], self.df['investment_plan']], 
            self.df['supports_escalation'],
            margins=True
        )
        print(detailed_contingency)
        
        chi_square_results['detailed_contingency'] = detailed_contingency
        
        return chi_square_results
    
    def create_visualizations(self):
        """
        Create visualizations for the results
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Study 2: Advisory Role - Escalation of Commitment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Interaction plot (top-left)
        ax1 = axes[0, 0]
        
        # Calculate means for interaction plot
        interaction_data = self.df.groupby(['outcome_valence', 'investment_plan'])['supports_escalation_numeric'].agg(['mean', 'std', 'count']).reset_index()
        
        # Pivot for easier plotting
        interaction_pivot = interaction_data.pivot(index='outcome_valence', columns='investment_plan', values='mean')
        
        # Plot lines
        x_pos = [0, 1]  # Negative, Positive
        escalation_means = [interaction_pivot.loc['Negative', 'Escalation'], interaction_pivot.loc['Positive', 'Escalation']]
        rational_means = [interaction_pivot.loc['Negative', 'Rational'], interaction_pivot.loc['Positive', 'Rational']]
        
        ax1.plot(x_pos, escalation_means, 'o-', linewidth=2, markersize=8, label='Escalation Plan')
        ax1.plot(x_pos, rational_means, 's-', linewidth=2, markersize=8, label='Rational Plan')
        
        ax1.set_xlabel('Outcome Valence')
        ax1.set_ylabel('Escalation Support Rate')
        ax1.set_title('Interaction: Outcome × Investment Plan')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Negative', 'Positive'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Bar plot by condition (top-right)
        ax2 = axes[0, 1]
        
        condition_means = self.df.groupby(['outcome_valence', 'investment_plan'])['supports_escalation_numeric'].mean()
        condition_counts = self.df.groupby(['outcome_valence', 'investment_plan']).size()
        
        # Create labels for the bars
        labels = [f"{outcome}\n{plan}" for outcome, plan in condition_means.index]
        values = condition_means.values
        
        bars = ax2.bar(range(len(labels)), values, alpha=0.7)
        ax2.set_xlabel('Condition')
        ax2.set_ylabel('Escalation Support Rate')
        ax2.set_title('Escalation Support by Condition')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (bar, value, count) in enumerate(zip(bars, values, condition_counts.values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        # 3. Contingency heatmap (bottom-left)
        ax3 = axes[1, 0]
        
        # Create contingency table for heatmap
        contingency_heatmap = pd.crosstab(
            self.df['outcome_valence'], 
            self.df['supports_escalation'], 
            normalize='index'  # Show proportions within each row
        )
        
        sns.heatmap(contingency_heatmap, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax3, cbar_kws={'label': 'Proportion'})
        ax3.set_title('Escalation Support by Outcome Valence')
        ax3.set_xlabel('Supports Escalation')
        ax3.set_ylabel('Outcome Valence')
        
        # 4. Investment plan comparison (bottom-right)
        ax4 = axes[1, 1]
        
        plan_contingency = pd.crosstab(
            self.df['investment_plan'], 
            self.df['supports_escalation'], 
            normalize='index'
        )
        
        sns.heatmap(plan_contingency, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax4, cbar_kws={'label': 'Proportion'})
        ax4.set_title('Escalation Support by Investment Plan')
        ax4.set_xlabel('Supports Escalation')
        ax4.set_ylabel('Investment Plan')
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Detailed breakdown
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create a more detailed breakdown
        detailed_summary = self.df.groupby(['outcome_valence', 'investment_plan', 'supports_escalation']).size().unstack(fill_value=0)
        detailed_summary_pct = detailed_summary.div(detailed_summary.sum(axis=1), axis=0)
        
        detailed_summary_pct.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title('Detailed Breakdown: Escalation Support by Condition')
        ax.set_xlabel('Condition (Outcome, Plan)')
        ax.set_ylabel('Proportion')
        ax.legend(title='Supports Escalation', labels=['No', 'Yes'])
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("="*60)
        
        # Basic descriptive statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 30)
        
        total_trials = len(self.df)
        escalation_support_rate = self.df['supports_escalation'].mean()
        
        print(f"Total valid trials: {total_trials}")
        print(f"Overall escalation support rate: {escalation_support_rate:.2%}")
        
        # By condition
        print("\nBy Condition:")
        condition_summary = self.df.groupby(['outcome_valence', 'investment_plan']).agg({
            'supports_escalation': ['count', 'mean', 'std']
        }).round(3)
        print(condition_summary)
        
        # Statistical test summary
        print(f"\n2. STATISTICAL TEST SUMMARY")
        print("-" * 30)
        
        # ANOVA summary
        interaction_p = self.anova_results.loc['C(outcome_valence):C(investment_plan)', 'PR(>F)']
        outcome_p = self.anova_results.loc['C(outcome_valence)', 'PR(>F)']
        plan_p = self.anova_results.loc['C(investment_plan)', 'PR(>F)']
        
        print(f"Two-way ANOVA Results:")
        print(f"  Main effect of Outcome Valence: F = {self.anova_results.loc['C(outcome_valence)', 'F']:.3f}, p = {outcome_p:.4f}")
        print(f"  Main effect of Investment Plan: F = {self.anova_results.loc['C(investment_plan)', 'F']:.3f}, p = {plan_p:.4f}")
        print(f"  Interaction effect: F = {self.anova_results.loc['C(outcome_valence):C(investment_plan)', 'F']:.3f}, p = {interaction_p:.4f}")
        
        # Interpretation
        print(f"\n3. INTERPRETATION")
        print("-" * 30)
        
        alpha = 0.05
        
        if interaction_p < alpha:
            print(f"✅ Significant interaction found (p = {interaction_p:.4f})")
            print("   The effect of outcome valence depends on the investment plan (or vice versa)")
        else:
            print(f"❌ No significant interaction (p = {interaction_p:.4f})")
        
        if outcome_p < alpha:
            print(f"✅ Significant main effect of Outcome Valence (p = {outcome_p:.4f})")
        else:
            print(f"❌ No significant main effect of Outcome Valence (p = {outcome_p:.4f})")
            
        if plan_p < alpha:
            print(f"✅ Significant main effect of Investment Plan (p = {plan_p:.4f})")
        else:
            print(f"❌ No significant main effect of Investment Plan (p = {plan_p:.4f})")
        
        # Key findings
        print(f"\n4. KEY FINDINGS")
        print("-" * 30)
        
        # Calculate key contrasts
        pos_esc = self.df[(self.df['outcome_valence'] == 'Positive') & (self.df['investment_plan'] == 'Escalation')]['supports_escalation'].mean()
        pos_rat = self.df[(self.df['outcome_valence'] == 'Positive') & (self.df['investment_plan'] == 'Rational')]['supports_escalation'].mean()
        neg_esc = self.df[(self.df['outcome_valence'] == 'Negative') & (self.df['investment_plan'] == 'Escalation')]['supports_escalation'].mean()
        neg_rat = self.df[(self.df['outcome_valence'] == 'Negative') & (self.df['investment_plan'] == 'Rational')]['supports_escalation'].mean()
        
        print(f"Escalation support rates by condition:")
        print(f"  Positive Outcome + Escalation Plan: {pos_esc:.2%}")
        print(f"  Positive Outcome + Rational Plan: {pos_rat:.2%}")
        print(f"  Negative Outcome + Escalation Plan: {neg_esc:.2%}")
        print(f"  Negative Outcome + Rational Plan: {neg_rat:.2%}")
        
        # Evidence for escalation of commitment
        if neg_esc > neg_rat:
            print(f"\n🔍 Potential escalation of commitment:")
            print(f"   Even with negative outcomes, escalation plan receives {neg_esc:.2%} support")
            print(f"   vs. {neg_rat:.2%} for rational reallocation")
        
    def run_full_analysis(self):
        """
        Run the complete statistical analysis pipeline
        """
        print("🔬 Starting Study 2: Advisory Role Analysis")
        print("=" * 60)
        
        try:
            # Load and preprocess data
            self.preprocess_data()
            
            # Check assumptions
            self.check_assumptions()
            
            # Run statistical tests
            self.run_two_way_anova()
            self.run_simple_effects()
            self.run_chi_square_tests()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate summary report
            self.generate_summary_report()
            
            print(f"\n✅ Analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            raise


# Example usage
def main():
    """
    Main function to run the analysis
    """
    
    # Update this path to your JSON file
    json_file_path = "/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_3-llm-human/results/llm-human_explicit_results_o4-mini-2025-04-16.json"  # Replace with actual path
    
    try:
        # Initialize analyzer
        analyzer = Study2Analyzer(json_file_path)
        
        # Run full analysis
        analyzer.run_full_analysis()
        
    except FileNotFoundError:
        print(f"❌ File not found: {json_file_path}")
        print("Please update the json_file_path variable with the correct path to your results file")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()