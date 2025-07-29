import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

# ========== CONFIGURATION ==========
JSON_PATH = "/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_3-llm-human/results/llm-human_explicit_results_o4-mini-2025-04-16.json"  # Update if needed
PLOT = True  # Set to False if running headless

# ========== LOAD JSON DATA ==========
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['trials'])

# ========== DATA CLEANING ==========
# Clean up the explicit decision data
def parse_explicit_decision(row):
    """Parse the explicit decision from the response"""
    if row['supports_vp_proposal'] is None:
        return None
    return row['supports_vp_proposal']

df['supports_vp_proposal'] = df.apply(parse_explicit_decision, axis=1)

# Parse condition into meaningful categories based on your experimental design
def parse_condition_components(condition):
    """Parse condition string into outcome and VP proposal type"""
    if 'success' in condition.lower():
        outcome = 'Positive'
    elif 'failure' in condition.lower():
        outcome = 'Negative'
    else:
        outcome = 'Unknown'
    
    if 'continue' in condition.lower() or 'escalation' in condition.lower():
        vp_behavior = 'Escalation'  # VP proposes doubling down
    elif 'pivot' in condition.lower() or 'rational' in condition.lower():
        vp_behavior = 'Pivot'  # VP proposes rational shift
    else:
        vp_behavior = 'Unknown'
    
    return outcome, vp_behavior

df[['outcome', 'vp_behavior']] = df['condition'].apply(
    lambda x: pd.Series(parse_condition_components(x))
)

# Convert boolean to numeric for analysis
df['support_numeric'] = df['supports_vp_proposal'].astype(float)
df['escalation_numeric'] = df['mentions_escalation_concepts'].astype(float)

# ========== DATA VALIDATION ==========
print("✅ Dataset Summary")
print("Experiment metadata:")
if 'experiment_metadata' in data:
    for key, value in data['experiment_metadata'].items():
        print(f"  {key}: {value}")

print(f"\nCondition breakdown:")
print(df['condition'].value_counts())
print(f"\nOutcome levels: {df['outcome'].unique()}")
print(f"VP Behavior levels: {df['vp_behavior'].unique()}")

print(f"\nDecision clarity breakdown:")
if 'decision_clarity' in df.columns:
    print(df['decision_clarity'].value_counts())

print(f"\nSupport decisions:")
support_breakdown = df['supports_vp_proposal'].value_counts(dropna=False)
print(support_breakdown)

# Check for full experimental design
has_full_design = (df['outcome'].nunique() == 2 and 
                   df['vp_behavior'].nunique() == 2 and
                   len(df) >= 4)

print(f"\nFull 2x2 design present: {has_full_design}")

# ========== KEY ESCALATION ANALYSIS ==========
print("\n" + "="*60)
print("🔬 ESCALATION OF COMMITMENT ANALYSIS")
print("="*60)

# The key comparison: How do LLMs respond when VP proposes escalation after failure?
failure_escalation = df[(df['outcome'] == 'Negative') & (df['vp_behavior'] == 'Escalation')]
failure_rational = df[(df['outcome'] == 'Negative') & (df['vp_behavior'] == 'Pivot')]

print(f"\n=== KEY CONDITION: Failure + VP Proposes Escalation ===")
if len(failure_escalation) > 0:
    escalation_support_rate = failure_escalation['support_numeric'].mean()
    print(f"Support rate for VP escalation after failure: {escalation_support_rate:.2%}")
    print(f"Number of trials: {len(failure_escalation)}")
    
    if len(failure_escalation) > 1:
        print(f"Individual decisions: {failure_escalation['supports_vp_proposal'].tolist()}")
else:
    print("⚠️ No failure_escalation trials found")

print(f"\n=== COMPARISON: Failure + VP Proposes Rational Pivot ===")
if len(failure_rational) > 0:
    rational_support_rate = failure_rational['support_numeric'].mean()
    print(f"Support rate for VP pivot after failure: {rational_support_rate:.2%}")
    print(f"Number of trials: {len(failure_rational)}")
    
    if len(failure_rational) > 1:
        print(f"Individual decisions: {failure_rational['supports_vp_proposal'].tolist()}")
else:
    print("⚠️ No failure_rational trials found")

# ========== STATISTICAL TESTS ==========
print("\n" + "="*60)
print("📊 STATISTICAL ANALYSIS")
print("="*60)

# Chi-square test for association between outcome/VP behavior and support
print("\n=== CHI-SQUARE TEST: Support ~ Outcome * VP_Behavior ===")
if has_full_design and df['supports_vp_proposal'].notna().sum() > 0:
    # Create contingency table
    contingency = pd.crosstab([df['outcome'], df['vp_behavior']], 
                             df['supports_vp_proposal'], 
                             margins=True)
    print("Contingency Table:")
    print(contingency)
    
    # Chi-square test (if we have enough data)
    if contingency.shape[0] > 2 and contingency.shape[1] > 2:
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency.iloc[:-1, :-1])
        print(f"\nChi-square test: χ² = {chi2:.3f}, p = {p_val:.4f}, df = {dof}")
else:
    print("⚠️ Insufficient data for chi-square test")

# Two-way ANOVA if we have enough data
print("\n=== TWO-WAY ANOVA: Support ~ Outcome * VP_Behavior ===")
if has_full_design and len(df) >= 8:  # Need reasonable sample size
    try:
        model = ols('support_numeric ~ C(outcome) * C(vp_behavior)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
    except Exception as e:
        print(f"⚠️ ANOVA failed: {str(e)}")
else:
    print("⚠️ Insufficient data for ANOVA (need full 2x2 design with adequate sample size)")

# ========== ESCALATION LANGUAGE ANALYSIS ==========
print("\n=== ESCALATION LANGUAGE ANALYSIS ===")
if 'mentions_escalation_concepts' in df.columns:
    escalation_by_condition = df.groupby(['outcome', 'vp_behavior'])['escalation_numeric'].agg(['count', 'sum', 'mean'])
    print("Escalation language mentions by condition:")
    print(escalation_by_condition)
else:
    print("⚠️ No escalation language analysis available")

# ========== DECISION CLARITY ANALYSIS ==========
print("\n=== DECISION CLARITY ANALYSIS ===")
if 'decision_clarity' in df.columns:
    clarity_by_condition = df.groupby(['outcome', 'vp_behavior'])['decision_clarity'].value_counts()
    print("Decision clarity by condition:")
    print(clarity_by_condition)
    
    # Check how many decisions were unclear
    unclear_decisions = df[df['decision_clarity'] == 'unclear']
    if len(unclear_decisions) > 0:
        print(f"\n⚠️ Found {len(unclear_decisions)} unclear decisions out of {len(df)} total")
        print("Unclear decision conditions:")
        print(unclear_decisions[['condition', 'decision_clarity']].to_string())

# ========== VISUALIZATION ==========
if PLOT and has_full_design and len(df) > 0:
    print("\n📈 Generating visualizations...")
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Support rate by condition
    if df['supports_vp_proposal'].notna().sum() > 0:
        condition_support = df.groupby('condition')['support_numeric'].mean()
        ax1.bar(range(len(condition_support)), condition_support.values)
        ax1.set_xticks(range(len(condition_support)))
        ax1.set_xticklabels(condition_support.index, rotation=45, ha='right')
        ax1.set_ylabel('Support Rate')
        ax1.set_title('Support Rate by Condition')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(condition_support.values):
            ax1.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')
    
    # Plot 2: 2x2 heatmap if we have full design
    if df['outcome'].nunique() == 2 and df['vp_behavior'].nunique() == 2:
        pivot_table = df.pivot_table(values='support_numeric', 
                                   index='outcome', 
                                   columns='vp_behavior', 
                                   aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlBu_r', 
                   ax=ax2, vmin=0, vmax=1)
        ax2.set_title('Support Rate: Outcome × VP Behavior')
    
    # Plot 3: Decision clarity
    if 'decision_clarity' in df.columns:
        clarity_counts = df['decision_clarity'].value_counts()
        ax3.pie(clarity_counts.values, labels=clarity_counts.index, autopct='%1.1f%%')
        ax3.set_title('Decision Clarity Distribution')
    
    # Plot 4: Response length distribution
    if 'response_length' in df.columns:
        ax4.hist(df['response_length'], bins=10, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Response Length (words)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Response Length Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Support rate comparison for key conditions
    if len(failure_escalation) > 0 and len(failure_rational) > 0:
        plt.figure(figsize=(10, 6))
        
        key_conditions = ['Failure + Escalation', 'Failure + Rational']
        support_rates = [
            failure_escalation['support_numeric'].mean(),
            failure_rational['support_numeric'].mean()
        ]
        
        bars = plt.bar(key_conditions, support_rates, 
                      color=['red', 'blue'], alpha=0.7)
        plt.ylabel('Support Rate')
        plt.title('Key Comparison: LLM Support for VP Proposals After Failure')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars, support_rates):
            plt.text(bar.get_x() + bar.get_width()/2, rate + 0.02, 
                    f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Add sample sizes
        sample_sizes = [len(failure_escalation), len(failure_rational)]
        for i, (bar, n) in enumerate(zip(bars, sample_sizes)):
            plt.text(bar.get_x() + bar.get_width()/2, 0.05, 
                    f'n={n}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()

# ========== SUMMARY REPORT ==========
print("\n" + "="*60)
print("📋 EXPERIMENT SUMMARY REPORT")
print("="*60)

print(f"Model: {data.get('experiment_metadata', {}).get('model_name', 'Unknown')}")
print(f"Total trials: {len(df)}")
print(f"Conditions tested: {df['condition'].nunique()}")

if len(failure_escalation) > 0:
    escalation_support = failure_escalation['support_numeric'].mean()
    print(f"\n🔍 KEY FINDING:")
    print(f"When VP proposes escalation after failure, LLM supports it {escalation_support:.1%} of the time")
    
    if escalation_support > 0.5:
        print("⚠️  This suggests potential escalation of commitment bias")
    else:
        print("✅ This suggests rational decision making (resistance to escalation)")

if len(failure_rational) > 0:
    rational_support = failure_rational['support_numeric'].mean()
    print(f"When VP proposes rational pivot after failure, LLM supports it {rational_support:.1%} of the time")

# Decision quality assessment
unclear_rate = (df['decision_clarity'] == 'unclear').sum() / len(df) if 'decision_clarity' in df.columns else 0
print(f"\nDecision clarity: {(1-unclear_rate):.1%} of responses were clear")

if unclear_rate > 0.1:
    print("⚠️  High rate of unclear decisions - consider refining prompt format")

print(f"\nAverage response length: {df['response_length'].mean():.0f} words")

# Export summary for further analysis
summary_stats = {
    'total_trials': len(df),
    'conditions': df['condition'].unique().tolist(),
    'escalation_after_failure_support_rate': failure_escalation['support_numeric'].mean() if len(failure_escalation) > 0 else None,
    'rational_after_failure_support_rate': failure_rational['support_numeric'].mean() if len(failure_rational) > 0 else None,
    'decision_clarity_rate': 1 - unclear_rate,
    'avg_response_length': df['response_length'].mean()
}

print(f"\n📊 Summary statistics:")
for key, value in summary_stats.items():
    if value is not None:
        if isinstance(value, float) and 'rate' in key:
            print(f"  {key}: {value:.2%}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

print("\n✅ Analysis complete!")
print(f"Data loaded from: {JSON_PATH}")
if PLOT:
    print("📈 Visualizations displayed")