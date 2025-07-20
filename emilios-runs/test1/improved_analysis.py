import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

class EscalationAnalyzer:
    def __init__(self, results_dir="experiment_results_improved"):
        self.results_dir = Path(results_dir)
        self.results = []
        self.df = None
        
    def load_results(self):
        """Load all experiment results from JSON files"""
        all_results_file = self.results_dir / "all_results.json"
        
        if all_results_file.exists():
            with open(all_results_file, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded {len(self.results)} results from {all_results_file}")
        else:
            # Load from individual files
            json_files = list(self.results_dir.glob("*.json"))
            for file in json_files:
                if file.name != "memory_validation_stats.json":
                    with open(file, 'r') as f:
                        data = json.load(f)
                        self.results.extend(data)
            print(f"Loaded {len(self.results)} results from {len(json_files)} files")
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.results)
        
        # Add derived columns
        if not self.df.empty:
            self.df['commitment_percentage'] = self.df['commitment'] / 20000000 * 100
            self.df['escalation_indicator'] = self.df['commitment'] > 10000000  # More than 50%
            
    def analyze_escalation_effect(self):
        """Analyze the main escalation of commitment effect"""
        print("\n" + "="*60)
        print("ESCALATION OF COMMITMENT ANALYSIS")
        print("="*60)
        
        # Filter for high responsibility conditions only
        high_resp = self.df[self.df['responsibility'] == 'high'].copy()
        
        if high_resp.empty:
            print("No high responsibility data found!")
            return
            
        # Group by condition
        positive_condition = high_resp[high_resp['condition'] == 'positive']
        negative_condition = high_resp[high_resp['condition'] == 'negative']
        
        print(f"Sample sizes:")
        print(f"  Positive outcomes: {len(positive_condition)}")
        print(f"  Negative outcomes: {len(negative_condition)}")
        
        if len(positive_condition) == 0 or len(negative_condition) == 0:
            print("Insufficient data for comparison!")
            return
            
        # Calculate key metrics
        pos_commitment_avg = positive_condition['commitment'].mean()
        neg_commitment_avg = negative_condition['commitment'].mean()
        pos_commitment_pct = positive_condition['commitment_percentage'].mean()
        neg_commitment_pct = negative_condition['commitment_percentage'].mean()
        
        print(f"\nCommitment Analysis:")
        print(f"  Positive outcomes: ${pos_commitment_avg:,.0f} ({pos_commitment_pct:.1f}%)")
        print(f"  Negative outcomes: ${neg_commitment_avg:,.0f} ({neg_commitment_pct:.1f}%)")
        print(f"  Difference: ${neg_commitment_avg - pos_commitment_avg:,.0f}")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(negative_condition['commitment'], 
                                        positive_condition['commitment'])
        
        print(f"\nStatistical Test (Independent t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.3f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Escalation interpretation
        escalation_detected = neg_commitment_avg > pos_commitment_avg
        print(f"\nEscalation of Commitment:")
        print(f"  Expected: Higher commitment after negative outcomes")
        print(f"  Observed: {'Higher' if escalation_detected else 'Lower'} commitment after negative outcomes")
        print(f"  Effect: {'DETECTED' if escalation_detected and p_value < 0.05 else 'NOT DETECTED'}")
        
        # Count high commitment cases
        pos_high_commitment = (positive_condition['commitment'] > 10000000).sum()
        neg_high_commitment = (negative_condition['commitment'] > 10000000).sum()
        
        print(f"\nHigh Commitment Cases (>50% to original choice):")
        print(f"  Positive outcomes: {pos_high_commitment}/{len(positive_condition)} ({pos_high_commitment/len(positive_condition)*100:.1f}%)")
        print(f"  Negative outcomes: {neg_high_commitment}/{len(negative_condition)} ({neg_high_commitment/len(negative_condition)*100:.1f}%)")
        
        return {
            'positive_avg': pos_commitment_avg,
            'negative_avg': neg_commitment_avg,
            'difference': neg_commitment_avg - pos_commitment_avg,
            'p_value': p_value,
            'escalation_detected': escalation_detected and p_value < 0.05
        }
    
    def analyze_by_choice(self):
        """Analyze results by initial choice (Division A vs B)"""
        print("\n" + "="*60)
        print("ANALYSIS BY INITIAL CHOICE")
        print("="*60)
        
        high_resp = self.df[self.df['responsibility'] == 'high'].copy()
        
        if 'first_choice' not in high_resp.columns:
            print("No first_choice data available!")
            return
            
        choice_analysis = high_resp.groupby(['first_choice', 'condition']).agg({
            'commitment': ['mean', 'std', 'count'],
            'commitment_percentage': 'mean'
        }).round(0)
        
        print("Average commitment by initial choice and condition:")
        print(choice_analysis)
        
        # Test for choice balance
        choice_counts = high_resp['first_choice'].value_counts()
        print(f"\nChoice Balance:")
        for choice, count in choice_counts.items():
            print(f"  Division {choice.upper()}: {count} subjects")
            
        # Chi-square test for choice balance
        if len(choice_counts) == 2:
            chi2, p_chi = stats.chisquare(choice_counts.values)
            print(f"  Balance test p-value: {p_chi:.3f}")
            print(f"  Balanced: {'Yes' if p_chi > 0.05 else 'No'}")
    
    def create_visualizations(self):
        """Create visualizations of the results"""
        if self.df.empty:
            print("No data to visualize!")
            return
            
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Escalation of Commitment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Main escalation effect - bar chart
        ax1 = axes[0, 0]
        high_resp = self.df[self.df['responsibility'] == 'high']
        
        if not high_resp.empty:
            condition_means = high_resp.groupby('condition')['commitment'].mean()
            condition_stds = high_resp.groupby('condition')['commitment'].std()
            
            bars = ax1.bar(condition_means.index, condition_means.values, 
                          yerr=condition_stds.values, capsize=5, 
                          color=['lightblue', 'lightcoral'], alpha=0.7)
            ax1.set_title('Average Commitment by Condition\n(High Responsibility)', fontweight='bold')
            ax1.set_ylabel('Commitment ($)')
            ax1.set_xlabel('Condition')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
            
            # Add value labels on bars
            for bar, value in zip(bars, condition_means.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200000,
                        f'${value/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # 2. Commitment distribution by condition
        ax2 = axes[0, 1]
        if not high_resp.empty:
            sns.boxplot(data=high_resp, x='condition', y='commitment', ax=ax2)
            ax2.set_title('Commitment Distribution by Condition', fontweight='bold')
            ax2.set_ylabel('Commitment ($)')
            ax2.set_xlabel('Condition')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
            
            # Add horizontal line at 50%
            ax2.axhline(y=10000000, color='red', linestyle='--', alpha=0.7, label='50% allocation')
            ax2.legend()
        
        # 3. Allocation patterns by all conditions
        ax3 = axes[1, 0]
        
        # Create a comprehensive view of all conditions
        plot_data = []
        for _, row in self.df.iterrows():
            if row['responsibility'] == 'high':
                condition_label = f"High-{row['condition']}"
            else:
                condition_label = f"Low-{row['condition']}"
            
            plot_data.append({
                'condition': condition_label,
                'Division A': row['division_a_allocation'],
                'Division B': row['division_b_allocation']
            })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            condition_order = ['High-positive', 'High-negative', 'Low-positive', 'Low-negative']
            condition_order = [c for c in condition_order if c in plot_df['condition'].unique()]
            
            # Group by condition and calculate means
            grouped = plot_df.groupby('condition')[['Division A', 'Division B']].mean()
            
            # Create stacked bar chart
            bottom = np.zeros(len(grouped))
            width = 0.6
            
            colors = ['skyblue', 'lightgreen']
            for i, (division, color) in enumerate(zip(['Division A', 'Division B'], colors)):
                values = grouped[division].values
                ax3.bar(range(len(grouped)), values, bottom=bottom, 
                       label=division, color=color, width=width, alpha=0.8)
                
                # Add value labels
                for j, value in enumerate(values):
                    if value > 500000:  # Only label if allocation is significant
                        ax3.text(j, bottom[j] + value/2, f'${value/1e6:.1f}M', 
                               ha='center', va='center', fontweight='bold')
                
                bottom += values
            
            ax3.set_title('Average Allocation by Condition', fontweight='bold')
            ax3.set_ylabel('Allocation ($)')
            ax3.set_xlabel('Condition')
            ax3.set_xticks(range(len(grouped)))
            ax3.set_xticklabels(grouped.index, rotation=45)
            ax3.legend()
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # 4. Escalation indicator by condition
        ax4 = axes[1, 1]
        if not high_resp.empty:
            escalation_counts = high_resp.groupby('condition')['escalation_indicator'].agg(['sum', 'count'])
            escalation_rates = (escalation_counts['sum'] / escalation_counts['count'] * 100)
            
            bars = ax4.bar(escalation_rates.index, escalation_rates.values, 
                          color=['lightblue', 'lightcoral'], alpha=0.7)
            ax4.set_title('High Commitment Rate by Condition\n(>50% to original choice)', fontweight='bold')
            ax4.set_ylabel('Percentage (%)')
            ax4.set_xlabel('Condition')
            ax4.set_ylim(0, 100)
            
            # Add value labels
            for bar, value in zip(bars, escalation_rates.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'escalation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualization saved to: {self.results_dir / 'escalation_analysis.png'}")
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        report_path = self.results_dir / 'escalation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("ESCALATION OF COMMITMENT EXPERIMENT REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total subjects: {len(self.df)}\n")
            f.write(f"High responsibility: {len(self.df[self.df['responsibility'] == 'high'])}\n")
            f.write(f"Low responsibility: {len(self.df[self.df['responsibility'] == 'low'])}\n\n")
            
            # Summary statistics
            if not self.df.empty:
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 20 + "\n")
                summary = self.df.groupby(['responsibility', 'condition']).agg({
                    'commitment': ['mean', 'std', 'count'],
                    'commitment_percentage': 'mean'
                }).round(2)
                f.write(str(summary))
                f.write("\n\n")
            
            # Main findings
            escalation_results = self.analyze_escalation_effect()
            if escalation_results:
                f.write("MAIN FINDINGS\n")
                f.write("-" * 15 + "\n")
                f.write(f"Escalation effect detected: {escalation_results['escalation_detected']}\n")
                f.write(f"P-value: {escalation_results['p_value']:.3f}\n")
                f.write(f"Effect size: ${escalation_results['difference']:,.0f}\n")
        
        print(f"Report saved to: {report_path}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Escalation of Commitment Analysis...")
        
        # Load data
        self.load_results()
        
        if self.df.empty:
            print("No data loaded! Please check the results directory.")
            return
        
        # Run analyses
        self.analyze_escalation_effect()
        self.analyze_by_choice()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        print("\nAnalysis complete!")

def main():
    """Main function to run the analysis"""
    # Initialize analyzer
    analyzer = EscalationAnalyzer()
    
    # Run full analysis
    analyzer.run_full_analysis()
    
    # Optional: Print raw data summary
    if not analyzer.df.empty:
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(analyzer.df.describe())

if __name__ == "__main__":
    main()