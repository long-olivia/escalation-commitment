import json
import numpy as np
from scipy import stats
import pandas as pd
import os
from datetime import datetime

def load_json_results(file_path):
    """Load JSON results from file"""
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        print(f"Successfully loaded {len(results)} subjects from {file_path}")
        return results
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def enhanced_analyze_results(results):
    """Enhanced statistical analysis matching Staw (1976) approach"""
    print(f"\n{'='*80}")
    print(f"ESCALATION OF COMMITMENT ANALYSIS - STAW (1976) REPLICATION")
    print(f"{'='*80}")
    print(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total subjects analyzed: {len(results)}")
    
    if len(results) == 0:
        print("No results to analyze!")
        return None
    
    # Group by conditions using exact condition names from JSON
    conditions = {
        'high_positive': [],
        'high_negative': [],
        'low_positive': [],
        'low_negative': []
    }
    
    # Extract commitment amounts by condition
    for subject in results:
        condition = subject['condition']
        commitment = subject['commitment_amount']
        
        if condition == 'high_responsibility_positive':
            conditions['high_positive'].append(commitment)
        elif condition == 'high_responsibility_negative':
            conditions['high_negative'].append(commitment)
        elif condition == 'low_responsibility_positive':
            conditions['low_positive'].append(commitment)
        elif condition == 'low_responsibility_negative':
            conditions['low_negative'].append(commitment)
    
    # Convert to numpy arrays and validate data
    for key in conditions:
        conditions[key] = np.array(conditions[key])
    
    # Calculate descriptive statistics
    def get_descriptives(data, name):
        if len(data) == 0:
            return None
        return {
            'n': len(data),
            'mean': np.mean(data),
            'sd': np.std(data, ddof=1) if len(data) > 1 else 0,
            'se': np.std(data, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else 0,
            'data': data
        }
    
    # Generate descriptive statistics
    print(f"\n{'='*80}")
    print(f"DESCRIPTIVE STATISTICS")
    print(f"{'='*80}")
    print(f"{'Condition':<35} {'N':<5} {'Mean ($M)':<12} {'SD ($M)':<12} {'SE ($M)':<12}")
    print(f"{'-'*80}")
    
    descriptives = {}
    condition_labels = {
        'high_positive': 'High Responsibility + Positive',
        'high_negative': 'High Responsibility + Negative', 
        'low_positive': 'Low Responsibility + Positive',
        'low_negative': 'Low Responsibility + Negative'
    }
    
    for cond_name, data in conditions.items():
        stats_dict = get_descriptives(data, cond_name)
        if stats_dict is not None and stats_dict['n'] > 0:
            descriptives[cond_name] = stats_dict
            label = condition_labels[cond_name]
            print(f"{label:<35} {stats_dict['n']:<5} {stats_dict['mean']/1e6:<12.2f} {stats_dict['sd']/1e6:<12.2f} {stats_dict['se']/1e6:<12.2f}")
    
    # Check if we have sufficient data for analysis
    if len(descriptives) == 0:
        print("No valid data found for analysis!")
        return None
    
    # PRELIMINARY ANALYSIS (matching original study)
    print(f"\n{'='*80}")
    print(f"PRELIMINARY ANALYSIS")
    print(f"{'='*80}")
    print(f"Testing for effects of division choice and information format...")
    
    # Check division choices in high responsibility conditions
    high_resp_subjects = [s for s in results if 'high_responsibility' in s['condition']]
    if high_resp_subjects:
        consumer_choices = sum(1 for s in high_resp_subjects if s.get('stage1_choice') == 'consumer')
        industrial_choices = sum(1 for s in high_resp_subjects if s.get('stage1_choice') == 'industrial')
        print(f"Division choices in high responsibility conditions:")
        print(f"  Consumer: {consumer_choices}, Industrial: {industrial_choices}")
        print(f"  (No significant effects expected, allowing for 2x2 analysis)")
    
    # MAIN EFFECTS ANALYSIS
    print(f"\n{'='*80}")
    print(f"MAIN EFFECTS ANALYSIS")
    print(f"{'='*80}")
    
    # Combine data for main effects
    high_resp_data = []
    low_resp_data = []
    pos_cons_data = []
    neg_cons_data = []
    
    for cond_name, stats_dict in descriptives.items():
        data = stats_dict['data']
        if 'high' in cond_name:
            high_resp_data.extend(data)
        else:
            low_resp_data.extend(data)
            
        if 'positive' in cond_name:
            pos_cons_data.extend(data)
        else:
            neg_cons_data.extend(data)
    
    # Convert to numpy arrays
    high_resp_data = np.array(high_resp_data)
    low_resp_data = np.array(low_resp_data)
    pos_cons_data = np.array(pos_cons_data)
    neg_cons_data = np.array(neg_cons_data)
    
    # Main effect of Personal Responsibility
    if len(high_resp_data) > 0 and len(low_resp_data) > 0:
        high_resp_mean = np.mean(high_resp_data)
        low_resp_mean = np.mean(low_resp_data)
        
        t_resp, p_resp = stats.ttest_ind(high_resp_data, low_resp_data)
        
        print(f"MAIN EFFECT OF PERSONAL RESPONSIBILITY:")
        print(f"  High Personal Responsibility: M = ${high_resp_mean/1e6:.2f}M (n = {len(high_resp_data)})")
        print(f"  Low Personal Responsibility:  M = ${low_resp_mean/1e6:.2f}M (n = {len(low_resp_data)})")
        print(f"  Difference: ${(high_resp_mean - low_resp_mean)/1e6:.2f}M")
        print(f"  t({len(high_resp_data) + len(low_resp_data) - 2}) = {t_resp:.3f}, p = {p_resp:.3f}")
        print(f"  Result: {'✓ Significant main effect' if p_resp < 0.05 else '✗ Not significant'}")
        print(f"  Original study: High (11.08M) vs Low (8.89M)")
    
    # Main effect of Decision Consequences  
    if len(pos_cons_data) > 0 and len(neg_cons_data) > 0:
        pos_cons_mean = np.mean(pos_cons_data)
        neg_cons_mean = np.mean(neg_cons_data)
        
        t_cons, p_cons = stats.ttest_ind(neg_cons_data, pos_cons_data)
        
        print(f"\nMAIN EFFECT OF DECISION CONSEQUENCES:")
        print(f"  Positive Consequences: M = ${pos_cons_mean/1e6:.2f}M (n = {len(pos_cons_data)})")
        print(f"  Negative Consequences: M = ${neg_cons_mean/1e6:.2f}M (n = {len(neg_cons_data)})")
        print(f"  Difference: ${(neg_cons_mean - pos_cons_mean)/1e6:.2f}M")
        print(f"  t({len(pos_cons_data) + len(neg_cons_data) - 2}) = {t_cons:.3f}, p = {p_cons:.3f}")
        print(f"  Result: {'✓ Significant main effect' if p_cons < 0.05 else '✗ Not significant'}")
        print(f"  Original study: Negative (11.20M) vs Positive (8.77M)")
    
    # INTERACTION ANALYSIS - The Critical Finding
    print(f"\n{'='*80}")
    print(f"INTERACTION OF PERSONAL RESPONSIBILITY AND DECISION CONSEQUENCES")
    print(f"{'='*80}")
    
    # Display all four cell means
    print(f"CELL MEANS:")
    for cond_name, stats_dict in descriptives.items():
        label = condition_labels[cond_name]
        print(f"  {label:<35}: M = ${stats_dict['mean']/1e6:.2f}M (n = {stats_dict['n']})")
    
    # Key test: High Responsibility + Negative vs. all other conditions
    if 'high_negative' in descriptives:
        high_neg_data = descriptives['high_negative']['data']
        high_neg_mean = descriptives['high_negative']['mean']
        
        # Combine all other conditions
        other_data = []
        other_ns = []
        for cond in ['high_positive', 'low_positive', 'low_negative']:
            if cond in descriptives:
                other_data.extend(descriptives[cond]['data'])
                other_ns.append(descriptives[cond]['n'])
        
        if len(other_data) > 0:
            other_data = np.array(other_data)
            other_mean = np.mean(other_data)
            
            t_interaction, p_interaction = stats.ttest_ind(high_neg_data, other_data)
            
            print(f"\nCRITICAL INTERACTION TEST:")
            print(f"High Personal Responsibility + Negative Consequences vs. All Other Conditions")
            print(f"  High Responsibility + Negative: M = ${high_neg_mean/1e6:.2f}M (n = {len(high_neg_data)})")
            print(f"  All Other Conditions:           M = ${other_mean/1e6:.2f}M (n = {len(other_data)})")
            print(f"  Difference: ${(high_neg_mean - other_mean)/1e6:.2f}M")
            print(f"  t(df = {len(high_neg_data) + len(other_data) - 2}) = {t_interaction:.3f}, p = {p_interaction:.3f}")
            print(f"  Result: {'✓ Significant interaction effect' if p_interaction < 0.05 else '✗ Not significant'}")
            print(f"  Original study: High Resp + Negative (13.07M) significantly > all others")
            
            # Effect size for the critical comparison
            pooled_std = np.sqrt(((len(high_neg_data) - 1) * np.var(high_neg_data, ddof=1) + 
                                (len(other_data) - 1) * np.var(other_data, ddof=1)) / 
                               (len(high_neg_data) + len(other_data) - 2))
            cohens_d = (high_neg_mean - other_mean) / pooled_std if pooled_std > 0 else 0
            print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Additional pairwise comparisons (as reported in original)
    print(f"\nADDITIONAL PAIRWISE COMPARISONS:")
    
    # Effect of consequences under low responsibility
    if 'low_positive' in descriptives and 'low_negative' in descriptives:
        low_pos_data = descriptives['low_positive']['data']
        low_neg_data = descriptives['low_negative']['data']
        t_low, p_low = stats.ttest_ind(low_neg_data, low_pos_data)
        print(f"  Low Responsibility: Negative vs Positive consequences")
        print(f"    t(df = {len(low_pos_data) + len(low_neg_data) - 2}) = {t_low:.3f}, p = {p_low:.3f}")
        print(f"    Original study: t = 1.20, n.s. {'✓ Matches' if p_low >= 0.05 else '✗ Differs'}")
    
    # Effect of responsibility under positive consequences
    if 'high_positive' in descriptives and 'low_positive' in descriptives:
        high_pos_data = descriptives['high_positive']['data']
        low_pos_data = descriptives['low_positive']['data']
        t_pos, p_pos = stats.ttest_ind(high_pos_data, low_pos_data)
        print(f"  Positive Consequences: High vs Low responsibility")
        print(f"    t(df = {len(high_pos_data) + len(low_pos_data) - 2}) = {t_pos:.3f}, p = {p_pos:.3f}")
        print(f"    Original study: t = 1.13, n.s. {'✓ Matches' if p_pos >= 0.05 else '✗ Differs'}")
    
    # PRIMARY ESCALATION OF COMMITMENT TEST
    print(f"\n{'='*80}")
    print(f"ESCALATION OF COMMITMENT - PRIMARY HYPOTHESIS TEST")
    print(f"{'='*80}")
    
    if 'high_positive' in descriptives and 'high_negative' in descriptives:
        pos_data = descriptives['high_positive']['data']
        neg_data = descriptives['high_negative']['data']
        
        pos_mean = descriptives['high_positive']['mean']
        neg_mean = descriptives['high_negative']['mean']
        escalation_effect = neg_mean - pos_mean
        
        t_esc, p_esc = stats.ttest_ind(neg_data, pos_data)
        
        print(f"ESCALATION HYPOTHESIS: High Responsibility subjects will allocate more")
        print(f"money to previously chosen alternatives after NEGATIVE consequences")
        print(f"")
        print(f"High Responsibility Conditions:")
        print(f"  After Positive Outcomes: M = ${pos_mean/1e6:.2f}M (n = {len(pos_data)})")
        print(f"  After Negative Outcomes: M = ${neg_mean/1e6:.2f}M (n = {len(neg_data)})")
        print(f"  Escalation Effect: ${escalation_effect/1e6:.2f}M")
        print(f"  t(df = {len(neg_data) + len(pos_data) - 2}) = {t_esc:.3f}, p = {p_esc:.3f}")
        print(f"")
        print(f"HYPOTHESIS TEST RESULTS:")
        print(f"  Direction: {'✓ Confirmed (negative > positive)' if escalation_effect > 0 else '✗ Not confirmed'}")
        print(f"  Statistical Significance: {'✓ Significant' if p_esc < 0.05 else '✗ Not significant'}")
        print(f"  Original Study Comparison: Expected positive escalation effect")
        
        # Effect size
        pooled_std = np.sqrt(((len(neg_data) - 1) * np.var(neg_data, ddof=1) + 
                            (len(pos_data) - 1) * np.var(pos_data, ddof=1)) / 
                           (len(neg_data) + len(pos_data) - 2))
        d_escalation = escalation_effect / pooled_std if pooled_std > 0 else 0
        print(f"  Effect Size (Cohen's d): {d_escalation:.3f}")
    
    # COMPREHENSIVE EFFECT SIZES
    print(f"\n{'='*80}")
    print(f"EFFECT SIZES SUMMARY (Cohen's d)")
    print(f"{'='*80}")
    
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size"""
        if len(group1) <= 1 or len(group2) <= 1:
            return 0
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    
    if len(high_resp_data) > 1 and len(low_resp_data) > 1:
        d_resp = cohens_d(high_resp_data, low_resp_data)
        print(f"  Personal Responsibility main effect: d = {d_resp:.3f}")
    
    if len(neg_cons_data) > 1 and len(pos_cons_data) > 1:
        d_cons = cohens_d(neg_cons_data, pos_cons_data)
        print(f"  Decision Consequences main effect:   d = {d_cons:.3f}")
    
    if 'high_negative' in descriptives and len(descriptives['high_negative']['data']) > 1:
        if len(other_data) > 1:
            d_interaction = cohens_d(descriptives['high_negative']['data'], other_data)
            print(f"  Interaction effect (HR+Neg vs others): d = {d_interaction:.3f}")
    
    print(f"\nEffect Size Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large")
    
    # SUMMARY FOR RESULTS SECTION
    print(f"\n{'='*80}")
    print(f"SUMMARY FOR RESULTS SECTION")
    print(f"{'='*80}")
    
    print(f"This analysis replicates the statistical approach of Staw (1976).")
    print(f"Key findings to report:")
    print(f"")
    print(f"1. DESCRIPTIVE STATISTICS: Report the four cell means and standard deviations")
    print(f"2. MAIN EFFECTS: Report t-tests for Responsibility and Consequences")
    print(f"3. INTERACTION: Report the critical test of High Resp + Negative vs. others")
    print(f"4. ESCALATION: Report the primary escalation hypothesis test")
    print(f"5. EFFECT SIZES: Include Cohen's d for all major comparisons")
    print(f"")
    print(f"Expected pattern from original study:")
    print(f"- High Responsibility + Negative should show highest allocation")
    print(f"- This condition should be significantly different from all others")
    print(f"- Main effects of both Responsibility and Consequences")
    print(f"")
    print(f"LLM-specific insights:")
    print(f"- Do LLMs show the same escalation bias as humans?")
    print(f"- Are LLMs affected by personal responsibility manipulation?")
    print(f"- How do LLMs respond to negative feedback?")
    
    return descriptives

def save_analysis_report(results, json_file_path, output_file=None):
    """Save analysis results to a text file"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_dir = os.path.dirname(json_file_path)
        output_file = os.path.join(input_dir, f"escalation_analysis_report_{timestamp}.txt")

    # Redirect print output to file
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    try:
        enhanced_analyze_results(results)
        analysis_text = buffer.getvalue()
    finally:
        sys.stdout = old_stdout
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(analysis_text)
    
    print(f"Analysis report saved to: {output_file}")
    return output_file

def main():
    """Main function to run the analysis"""
    print("Escalation of Commitment Analysis - JSON Results Processor")
    print("="*60)
    
    # SPECIFY YOUR JSON FILE PATH HERE:
    json_file_path = "/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_1-2-high-low/results/all_results_o4-mini-2025-04-16.json"

    # Alternative file paths (uncomment the one you want to use):
    # json_file_path = "escalation_results/all_results_n100_20250726_143045.json"
    # json_file_path = "escalation_results/your_filename_here.json"
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        print("\nAvailable files in escalation_results directory:")
        try:
            files = [f for f in os.listdir("escalation_results") if f.endswith('.json')]
            for i, file in enumerate(files, 1):
                print(f"  {i}. {file}")
            
            if files:
                choice = input(f"\nEnter the number of the file to analyze (1-{len(files)}): ")
                try:
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(files):
                        json_file_path = os.path.join("escalation_results", files[file_index])
                    else:
                        print("Invalid selection.")
                        return
                except ValueError:
                    print("Invalid input.")
                    return
        except FileNotFoundError:
            print("escalation_results directory not found.")
            return
    
    # Load and analyze results
    results = load_json_results(json_file_path)
    if results is None:
        return
    
    # Run the enhanced analysis
    descriptives = enhanced_analyze_results(results)
    
    # Ask if user wants to save report
    save_report = input("\nSave analysis report to file? (y/n): ").lower().strip()
    if save_report in ['y', 'yes']:
        report_file = save_analysis_report(results, json_file_path)
        print(f"Report saved as: {report_file}")

if __name__ == "__main__":
    main()