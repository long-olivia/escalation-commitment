import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class StawReplicationAnalyzer:
    def __init__(self, results_file: str = None):
        """
        Initialize the Staw (1976) replication analyzer
        
        Args:
            results_file: Path to the JSON results file from the experiment
        """
        self.results_file = results_file
        self.data = None
        self.df = None
        self.analysis_results = {}
        
    def load_data(self, file_path: str = None) -> bool:
        """Load experiment results from JSON file"""
        if file_path:
            self.results_file = file_path
        
        if not self.results_file:
            print("❌ No results file specified")
            return False
            
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Convert to DataFrame for easier analysis
            self.df = pd.DataFrame(self.data['trials'])
            
            print(f"✅ Loaded {len(self.df)} trials from {self.results_file}")
            return True
            
        except FileNotFoundError:
            print(f"❌ File not found: {self.results_file}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def map_conditions_to_staw_format(self) -> pd.DataFrame:
        """
        Map experimental conditions to Staw (1976) format:
        - High/Low Personal Responsibility
        - Positive/Negative Decision Consequences
        """
        if self.df is None:
            print("❌ No data loaded")
            return None
        
        # Create mapping based on your experimental conditions
        condition_mapping = {
            'success_continue': ('High', 'Positive'),     # VP chose, positive outcome, proposes continue
            'success_pivot': ('Low', 'Positive'),         # VP chose, positive outcome, proposes pivot  
            'failure_rational': ('Low', 'Negative'),      # VP chose, negative outcome, proposes pivot
            'failure_escalation': ('High', 'Negative')    # VP chose, negative outcome, proposes continue
        }
        
        # Map conditions
        mapped_df = self.df.copy()
        mapped_df['responsibility'] = mapped_df['condition'].map(lambda x: condition_mapping.get(x, ('Unknown', 'Unknown'))[0])
        mapped_df['consequences'] = mapped_df['condition'].map(lambda x: condition_mapping.get(x, ('Unknown', 'Unknown'))[1])
        
        # Extract allocation amounts from responses
        mapped_df['allocation_amount'] = mapped_df.apply(self._extract_allocation_amount, axis=1)
        
        return mapped_df
    
    def _extract_allocation_amount(self, row) -> float:
        """
        Extract the monetary allocation amount from LLM responses
        This assumes the VP proposes specific dollar amounts in the experimental setup
        """
        # For the Staw replication, we need to convert support/oppose to allocation amounts
        # Based on the experimental design where VP proposes specific allocations
        
        condition = row['condition']
        supports_vp = row['supports_vp_proposal']
        
        # Default allocations based on VP proposals in your experiment
        # Adjust these based on your actual experimental setup
        if condition in ['success_continue', 'failure_escalation']:
            # VP proposes continuing with higher allocation to original choice
            if supports_vp:
                return 15.0  # $15M to original choice (escalation)
            else:
                return 5.0   # $5M to original choice (rational)
        else:  # success_pivot, failure_rational
            # VP proposes switching to lower allocation to original choice
            if supports_vp:
                return 5.0   # $5M to original choice (rational)
            else:
                return 15.0  # $15M to original choice (against VP's rational advice)
        
        # Default if unclear
        return 10.0
    
    def calculate_descriptive_statistics(self) -> Dict:
        """Calculate descriptive statistics in Staw format"""
        mapped_df = self.map_conditions_to_staw_format()
        
        if mapped_df is None:
            return {}
        
        # Group by responsibility and consequences
        grouped = mapped_df.groupby(['responsibility', 'consequences'])['allocation_amount']
        
        stats_dict = {}
        for (resp, cons), group in grouped:
            condition_name = f"{resp} Responsibility + {cons}"
            stats_dict[condition_name] = {
                'N': len(group),
                'Mean': group.mean(),
                'SD': group.std(),
                'SE': group.std() / np.sqrt(len(group))
            }
        
        self.analysis_results['descriptive_stats'] = stats_dict
        return stats_dict
    
    def test_main_effects(self) -> Dict:
        """Test main effects of responsibility and consequences"""
        mapped_df = self.map_conditions_to_staw_format()
        
        if mapped_df is None:
            return {}
        
        results = {}
        
        # Main effect of Personal Responsibility
        high_resp = mapped_df[mapped_df['responsibility'] == 'High']['allocation_amount']
        low_resp = mapped_df[mapped_df['responsibility'] == 'Low']['allocation_amount']
        
        t_stat_resp, p_val_resp = stats.ttest_ind(high_resp, low_resp)
        
        results['responsibility_main_effect'] = {
            'high_resp_mean': high_resp.mean(),
            'low_resp_mean': low_resp.mean(),
            'high_resp_n': len(high_resp),
            'low_resp_n': len(low_resp),
            'difference': high_resp.mean() - low_resp.mean(),
            't_statistic': t_stat_resp,
            'p_value': p_val_resp,
            'significant': p_val_resp < 0.05,
            'cohen_d': (high_resp.mean() - low_resp.mean()) / np.sqrt(((len(high_resp)-1)*high_resp.var() + (len(low_resp)-1)*low_resp.var()) / (len(high_resp) + len(low_resp) - 2))
        }
        
        # Main effect of Decision Consequences
        positive_cons = mapped_df[mapped_df['consequences'] == 'Positive']['allocation_amount']
        negative_cons = mapped_df[mapped_df['consequences'] == 'Negative']['allocation_amount']
        
        t_stat_cons, p_val_cons = stats.ttest_ind(positive_cons, negative_cons)
        
        results['consequences_main_effect'] = {
            'positive_cons_mean': positive_cons.mean(),
            'negative_cons_mean': negative_cons.mean(),
            'positive_cons_n': len(positive_cons),
            'negative_cons_n': len(negative_cons),
            'difference': positive_cons.mean() - negative_cons.mean(),
            't_statistic': t_stat_cons,
            'p_value': p_val_cons,
            'significant': p_val_cons < 0.05,
            'cohen_d': (positive_cons.mean() - negative_cons.mean()) / np.sqrt(((len(positive_cons)-1)*positive_cons.var() + (len(negative_cons)-1)*negative_cons.var()) / (len(positive_cons) + len(negative_cons) - 2))
        }
        
        self.analysis_results['main_effects'] = results
        return results
    
    def test_interaction_effect(self) -> Dict:
        """Test the critical interaction effect (High Responsibility + Negative vs others)"""
        mapped_df = self.map_conditions_to_staw_format()
        
        if mapped_df is None:
            return {}
        
        # Critical condition: High Responsibility + Negative
        critical_condition = mapped_df[
            (mapped_df['responsibility'] == 'High') & 
            (mapped_df['consequences'] == 'Negative')
        ]['allocation_amount']
        
        # All other conditions
        other_conditions = mapped_df[
            ~((mapped_df['responsibility'] == 'High') & 
              (mapped_df['consequences'] == 'Negative'))
        ]['allocation_amount']
        
        t_stat, p_val = stats.ttest_ind(critical_condition, other_conditions)
        
        # Calculate cell means for full interaction
        cell_means = {}
        for resp in ['High', 'Low']:
            for cons in ['Positive', 'Negative']:
                cell_data = mapped_df[
                    (mapped_df['responsibility'] == resp) & 
                    (mapped_df['consequences'] == cons)
                ]['allocation_amount']
                cell_means[f"{resp} Responsibility + {cons}"] = {
                    'mean': cell_data.mean(),
                    'n': len(cell_data),
                    'sd': cell_data.std()
                }
        
        interaction_results = {
            'critical_condition_mean': critical_condition.mean(),
            'critical_condition_n': len(critical_condition),
            'other_conditions_mean': other_conditions.mean(),
            'other_conditions_n': len(other_conditions),
            'difference': critical_condition.mean() - other_conditions.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'cohen_d': (critical_condition.mean() - other_conditions.mean()) / np.sqrt(((len(critical_condition)-1)*critical_condition.var() + (len(other_conditions)-1)*other_conditions.var()) / (len(critical_condition) + len(other_conditions) - 2)),
            'cell_means': cell_means
        }
        
        self.analysis_results['interaction_effect'] = interaction_results
        return interaction_results
    
    def test_escalation_hypothesis(self) -> Dict:
        """Test the primary escalation of commitment hypothesis"""
        mapped_df = self.map_conditions_to_staw_format()
        
        if mapped_df is None:
            return {}
        
        # High responsibility conditions only
        high_resp_data = mapped_df[mapped_df['responsibility'] == 'High']
        
        # Compare positive vs negative outcomes within high responsibility
        high_resp_positive = high_resp_data[high_resp_data['consequences'] == 'Positive']['allocation_amount']
        high_resp_negative = high_resp_data[high_resp_data['consequences'] == 'Negative']['allocation_amount']
        
        t_stat, p_val = stats.ttest_ind(high_resp_negative, high_resp_positive)
        
        escalation_results = {
            'high_resp_positive_mean': high_resp_positive.mean(),
            'high_resp_negative_mean': high_resp_negative.mean(),
            'high_resp_positive_n': len(high_resp_positive),
            'high_resp_negative_n': len(high_resp_negative),
            'escalation_effect': high_resp_negative.mean() - high_resp_positive.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'direction_confirmed': high_resp_negative.mean() > high_resp_positive.mean(),
            'cohen_d': (high_resp_negative.mean() - high_resp_positive.mean()) / np.sqrt(((len(high_resp_negative)-1)*high_resp_negative.var() + (len(high_resp_positive)-1)*high_resp_positive.var()) / (len(high_resp_negative) + len(high_resp_positive) - 2))
        }
        
        self.analysis_results['escalation_hypothesis'] = escalation_results
        return escalation_results
    
    def generate_staw_format_report(self) -> str:
        """Generate a report in the exact format of Staw (1976) analysis"""
        if self.df is None:
            return "❌ No data loaded. Cannot generate report."
        
        # Run all analyses
        descriptive_stats = self.calculate_descriptive_statistics()
        main_effects = self.test_main_effects()
        interaction = self.test_interaction_effect()
        escalation = self.test_escalation_hypothesis()
        
        report = []
        report.append("=" * 80)
        report.append("ESCALATION OF COMMITMENT ANALYSIS - STAW (1976) REPLICATION")
        report.append("=" * 80)
        report.append(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total subjects analyzed: {len(self.df)}")
        report.append("")
        
        # Descriptive Statistics
        report.append("=" * 80)
        report.append("DESCRIPTIVE STATISTICS")
        report.append("=" * 80)
        report.append(f"{'Condition':<35} {'N':<8} {'Mean ($M)':<12} {'SD ($M)':<12} {'SE ($M)':<12}")
        report.append("-" * 80)
        
        for condition_name, stats in descriptive_stats.items():
            report.append(f"{condition_name:<35} {stats['N']:<8} {stats['Mean']:<12.2f} {stats['SD']:<12.2f} {stats['SE']:<12.2f}")
        
        report.append("")
        
        # Main Effects Analysis
        report.append("=" * 80)
        report.append("MAIN EFFECTS ANALYSIS")
        report.append("=" * 80)
        
        # Personal Responsibility Main Effect
        resp_effect = main_effects['responsibility_main_effect']
        report.append("MAIN EFFECT OF PERSONAL RESPONSIBILITY:")
        report.append(f"  High Personal Responsibility: M = ${resp_effect['high_resp_mean']:.2f}M (n = {resp_effect['high_resp_n']})")
        report.append(f"  Low Personal Responsibility:  M = ${resp_effect['low_resp_mean']:.2f}M (n = {resp_effect['low_resp_n']})")
        report.append(f"  Difference: ${resp_effect['difference']:+.2f}M")
        report.append(f"  t({resp_effect['high_resp_n'] + resp_effect['low_resp_n'] - 2}) = {resp_effect['t_statistic']:.3f}, p = {resp_effect['p_value']:.3f}")
        report.append(f"  Result: {'✓ Significant' if resp_effect['significant'] else '✗ Not significant'}")
        report.append("")
        
        # Decision Consequences Main Effect
        cons_effect = main_effects['consequences_main_effect']
        report.append("MAIN EFFECT OF DECISION CONSEQUENCES:")
        report.append(f"  Positive Consequences: M = ${cons_effect['positive_cons_mean']:.2f}M (n = {cons_effect['positive_cons_n']})")
        report.append(f"  Negative Consequences: M = ${cons_effect['negative_cons_mean']:.2f}M (n = {cons_effect['negative_cons_n']})")
        report.append(f"  Difference: ${cons_effect['difference']:+.2f}M")
        report.append(f"  t({cons_effect['positive_cons_n'] + cons_effect['negative_cons_n'] - 2}) = {cons_effect['t_statistic']:.3f}, p = {cons_effect['p_value']:.3f}")
        report.append(f"  Result: {'✓ Significant main effect' if cons_effect['significant'] else '✗ Not significant'}")
        report.append("")
        
        # Interaction Effect
        report.append("=" * 80)
        report.append("INTERACTION OF PERSONAL RESPONSIBILITY AND DECISION CONSEQUENCES")
        report.append("=" * 80)
        report.append("CELL MEANS:")
        
        for cell_name, cell_data in interaction['cell_means'].items():
            report.append(f"  {cell_name:<35}: M = ${cell_data['mean']:.2f}M (n = {cell_data['n']})")
        
        report.append("")
        report.append("CRITICAL INTERACTION TEST:")
        report.append("High Personal Responsibility + Negative Consequences vs. All Other Conditions")
        report.append(f"  High Responsibility + Negative: M = ${interaction['critical_condition_mean']:.2f}M (n = {interaction['critical_condition_n']})")
        report.append(f"  All Other Conditions:           M = ${interaction['other_conditions_mean']:.2f}M (n = {interaction['other_conditions_n']})")
        report.append(f"  Difference: ${interaction['difference']:+.2f}M")
        report.append(f"  t(df = {interaction['critical_condition_n'] + interaction['other_conditions_n'] - 2}) = {interaction['t_statistic']:.3f}, p = {interaction['p_value']:.3f}")
        report.append(f"  Result: {'✓ Significant interaction effect' if interaction['significant'] else '✗ Not significant'}")
        report.append(f"  Effect size (Cohen's d): {interaction['cohen_d']:.3f}")
        report.append("")
        
        # Escalation Hypothesis Test
        report.append("=" * 80)
        report.append("ESCALATION OF COMMITMENT - PRIMARY HYPOTHESIS TEST")
        report.append("=" * 80)
        report.append("ESCALATION HYPOTHESIS: High Responsibility subjects will allocate more")
        report.append("money to previously chosen alternatives after NEGATIVE consequences")
        report.append("")
        report.append("High Responsibility Conditions:")
        report.append(f"  After Positive Outcomes: M = ${escalation['high_resp_positive_mean']:.2f}M (n = {escalation['high_resp_positive_n']})")
        report.append(f"  After Negative Outcomes: M = ${escalation['high_resp_negative_mean']:.2f}M (n = {escalation['high_resp_negative_n']})")
        report.append(f"  Escalation Effect: ${escalation['escalation_effect']:+.2f}M")
        report.append(f"  t(df = {escalation['high_resp_positive_n'] + escalation['high_resp_negative_n'] - 2}) = {escalation['t_statistic']:.3f}, p = {escalation['p_value']:.3f}")
        report.append("")
        report.append("HYPOTHESIS TEST RESULTS:")
        report.append(f"  Direction: {'✓ Confirmed' if escalation['direction_confirmed'] else '✗ Not confirmed'}")
        report.append(f"  Statistical Significance: {'✓ Significant' if escalation['significant'] else '✗ Not significant'}")
        report.append(f"  Expected: Positive escalation effect (negative > positive)")
        report.append(f"  Effect Size (Cohen's d): {escalation['cohen_d']:.3f}")
        report.append("")
        
        # Effect Sizes Summary
        report.append("=" * 80)
        report.append("EFFECT SIZES SUMMARY (Cohen's d)")
        report.append("=" * 80)
        report.append(f"  Personal Responsibility main effect: d = {resp_effect['cohen_d']:.3f}")
        report.append(f"  Decision Consequences main effect:   d = {cons_effect['cohen_d']:.3f}")
        report.append(f"  Interaction effect (HR+Neg vs others): d = {interaction['cohen_d']:.3f}")
        report.append(f"  Escalation hypothesis test: d = {escalation['cohen_d']:.3f}")
        report.append("")
        report.append("Effect Size Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large")
        report.append("")
        
        # LLM-Specific Summary
        report.append("=" * 80)
        report.append("LLM ESCALATION BIAS ASSESSMENT")
        report.append("=" * 80)
        
        if escalation['direction_confirmed'] and escalation['significant']:
            bias_assessment = "✓ ESCALATION BIAS DETECTED: LLM shows escalation of commitment"
        elif escalation['significant'] and not escalation['direction_confirmed']:
            bias_assessment = "⚠️ REVERSE PATTERN: LLM shows rational de-escalation behavior"
        else:
            bias_assessment = "✗ NO CLEAR BIAS: Results not statistically significant"
        
        report.append(f"Overall Assessment: {bias_assessment}")
        report.append("")
        report.append("Key Findings:")
        report.append(f"- LLM responds to personal responsibility manipulation: {'Yes' if resp_effect['significant'] else 'No'}")
        report.append(f"- LLM responds to outcome feedback: {'Yes' if cons_effect['significant'] else 'No'}")
        report.append(f"- LLM shows escalation after failure: {'Yes' if escalation['direction_confirmed'] else 'No'}")
        report.append(f"- Effect size compared to humans: {abs(escalation['cohen_d']):.1f}x {'stronger' if abs(escalation['cohen_d']) > 0.8 else 'weaker'}")
        
        return "\n".join(report)
    
    def create_staw_visualizations(self, save_path: str = None):
        """Create visualizations matching Staw (1976) analysis format"""
        if self.df is None:
            print("❌ No data loaded. Cannot create visualizations.")
            return
        
        mapped_df = self.map_conditions_to_staw_format()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Staw (1976) Replication - Escalation of Commitment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Main Effects
        ax1 = axes[0, 0]
        resp_means = mapped_df.groupby('responsibility')['allocation_amount'].mean()
        bars1 = ax1.bar(resp_means.index, resp_means.values, color=['lightblue', 'lightcoral'])
        ax1.set_xlabel('Personal Responsibility')
        ax1.set_ylabel('Mean Allocation ($M)')
        ax1.set_title('Main Effect of Personal Responsibility')
        ax1.set_ylim(0, max(resp_means.values) * 1.2)
        
        for i, v in enumerate(resp_means.values):
            ax1.text(i, v + 0.5, f'${v:.1f}M', ha='center', va='bottom')
        
        # 2. Consequences Main Effect
        ax2 = axes[0, 1]
        cons_means = mapped_df.groupby('consequences')['allocation_amount'].mean()
        bars2 = ax2.bar(cons_means.index, cons_means.values, color=['lightgreen', 'salmon'])
        ax2.set_xlabel('Decision Consequences')
        ax2.set_ylabel('Mean Allocation ($M)')
        ax2.set_title('Main Effect of Decision Consequences')
        ax2.set_ylim(0, max(cons_means.values) * 1.2)
        
        for i, v in enumerate(cons_means.values):
            ax2.text(i, v + 0.5, f'${v:.1f}M', ha='center', va='bottom')
        
        # 3. Interaction Plot (2x2 design)
        ax3 = axes[1, 0]
        interaction_data = mapped_df.groupby(['responsibility', 'consequences'])['allocation_amount'].mean().unstack()
        
        x = np.arange(len(interaction_data.index))
        width = 0.35
        
        bars_pos = ax3.bar(x - width/2, interaction_data['Positive'], width, label='Positive Consequences', color='lightgreen', alpha=0.7)
        bars_neg = ax3.bar(x + width/2, interaction_data['Negative'], width, label='Negative Consequences', color='salmon', alpha=0.7)
        
        ax3.set_xlabel('Personal Responsibility')
        ax3.set_ylabel('Mean Allocation ($M)')
        ax3.set_title('Interaction: Responsibility × Consequences')
        ax3.set_xticks(x)
        ax3.set_xticklabels(interaction_data.index)
        ax3.legend()
        
        # Add value labels
        for bars in [bars_pos, bars_neg]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f'${height:.1f}M', ha='center', va='bottom')
        
        # 4. Escalation Effect (High Responsibility Only)
        ax4 = axes[1, 1]
        high_resp_data = mapped_df[mapped_df['responsibility'] == 'High']
        escalation_means = high_resp_data.groupby('consequences')['allocation_amount'].mean()
        
        bars4 = ax4.bar(escalation_means.index, escalation_means.values, 
                       color=['lightgreen', 'red'], alpha=0.7)
        ax4.set_xlabel('Outcome (High Responsibility Only)')
        ax4.set_ylabel('Mean Allocation ($M)')
        ax4.set_title('Escalation Effect Test')
        ax4.set_ylim(0, max(escalation_means.values) * 1.2)
        
        # Highlight the critical comparison
        if escalation_means['Negative'] > escalation_means['Positive']:
            ax4.annotate('Escalation Effect!', 
                        xy=(1, escalation_means['Negative']), 
                        xytext=(0.5, escalation_means['Negative'] + 2),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=12, color='red', weight='bold')
        
        for i, v in enumerate(escalation_means.values):
            ax4.text(i, v + 0.5, f'${v:.1f}M', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {save_path}")
        
        plt.show()

def main():
    """Main function to run the Staw replication analysis"""
    # Configuration - Update these paths
    RESULTS_DIR = "/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_3-llm-human/results"
    RESULTS_FILE = "/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_3-llm-human/results/llm-human_explicit_results_o4-mini-2025-04-16.json"  # Change this to your actual filename
    
    print("🔬 Staw (1976) Escalation of Commitment Replication Analysis")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = StawReplicationAnalyzer()
    
    # Try to find results file
    full_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    
    if not os.path.exists(full_path):
        # Look for any JSON files in the results directory
        json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
        if json_files:
            print(f"Found JSON files: {json_files}")
            full_path = os.path.join(RESULTS_DIR, json_files[0])
            print(f"Using: {json_files[0]}")
        else:
            print(f"❌ No JSON files found in {RESULTS_DIR}")
            return
    
    # Load and analyze data
    if analyzer.load_data(full_path):
        # Generate Staw-format report
        report = analyzer.generate_staw_format_report()
        print("\n" + report)
        
        # Save report to file
        report_file = full_path.replace('.json', '_staw_replication_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Staw replication report saved to: {report_file}")
        
        # Create visualizations
        viz_file = full_path.replace('.json', '_staw_visualizations.png')
        try:
            analyzer.create_staw_visualizations(viz_file)
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {str(e)}")
            print("Install matplotlib and seaborn for visualization support")
    
    else:
        print("❌ Failed to load data. Check file path and format.")

if __name__ == "__main__":
    main()