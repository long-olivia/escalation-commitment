import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ========== CONFIGURATION ==========
CSV_PATH = "/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_3-llm-human/results/llm-human_results_o4-mini-2025-04-16.csv"  # Update if needed
PLOT = True  # Set to False if running headless

# ========== LOAD DATA ==========
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# ========== DATA CLEANING ==========
# Normalize boolean fields
df['supports_vp_proposal'] = df['supports_vp_proposal'].astype(str).str.lower() == 'true'
df['mentions_escalation_concepts'] = df['mentions_escalation_concepts'].astype(str).str.lower() == 'true'

# Parse condition into Responsibility and Outcome
df['responsibility'] = df['condition'].apply(lambda x: 'High' if 'high' in x.lower() else 'Low')
df['outcome'] = df['condition'].apply(lambda x: 'Positive' if 'positive' in x.lower() else 'Negative')

# Convert binary DV to numeric
df['support_numeric'] = df['supports_vp_proposal'].astype(float)
df['escalation_numeric'] = df['mentions_escalation_concepts'].astype(float)

# ========== DATA VALIDATION ==========
print("✅ Dataset Summary")
print(df['condition'].value_counts())
print("\nResponsibility levels:", df['responsibility'].unique())
print("Outcome levels:", df['outcome'].unique())

has_full_design = df['responsibility'].nunique() == 2 and df['outcome'].nunique() == 2
has_enough_high = df[df['responsibility'] == 'High'].shape[0] >= 2
has_enough_low = df[df['responsibility'] == 'Low'].shape[0] >= 2

# ========== T-TEST ==========
print("\n=== T-TEST: Support VP Proposal (High vs Low Responsibility) ===")
if has_enough_high and has_enough_low:
    high = df[df['responsibility'] == 'High']['support_numeric']
    low = df[df['responsibility'] == 'Low']['support_numeric']
    t_stat, p_val = stats.ttest_ind(high, low)
    print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
else:
    print("⚠️ Not enough data in both responsibility groups for t-test.")

# ========== TWO-WAY ANOVA ==========
print("\n=== TWO-WAY ANOVA: Support VP Proposal ~ Responsibility * Outcome ===")
if has_full_design:
    model = ols('support_numeric ~ C(responsibility) * C(outcome)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
else:
    print("⚠️ Full 2x2 design (High/Low × Pos/Neg) not present. Skipping ANOVA.")

# ========== ESCALATION LANGUAGE ANALYSIS ==========
print("\n=== TWO-WAY ANOVA: Mentions Escalation Concepts ===")
if has_full_design:
    model2 = ols('escalation_numeric ~ C(responsibility) * C(outcome)', data=df).fit()
    anova_table2 = sm.stats.anova_lm(model2, typ=2)
    print(anova_table2)
else:
    print("⚠️ Full 2x2 design not present for escalation concept ANOVA.")

# ========== PLOT ==========
if PLOT and has_full_design:
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="responsibility", y="support_numeric", hue="outcome", ci=95)
    plt.title("Support Rate by Responsibility and Outcome")
    plt.ylabel("Support Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
