import pandas as pd
import numpy as np
import json
import os
import glob
from scipy import stats
from scipy.stats import chi2_contingency, levene, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class DeliberationAnalyzer:
    """
    Comprehensive statistical analysis for multi-agent deliberation experiment.
    
    Analyzes a 2x2x2 between-subjects design:
    - Responsibility (high vs. low)
    - Outcome Valence (positive vs. negative) 
    - Hierarchy Structure (symmetrical vs. asymmetrical)
    """
    
    def __init__(self, asymm_path=None, symm_path=None, single_path=None):
        """
        Initialize analyzer with data paths.
        
        Parameters:
        -----------
        asymm_path : str, optional
            Path to directory containing asymmetrical condition JSON files
        symm_path : str, optional
            Path to directory containing symmetrical condition JSON files
        single_path : str, optional
            Path to single directory or file (for backward compatibility)
        """
        self.asymm_path = asymm_path
        self.symm_path = symm_path
        self.single_path = single_path
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load and preprocess data from JSON files."""
        print("Loading data...")
        
        all_data = []
        
        # Handle different input methods
        if self.single_path:
            # Backward compatibility - single path
            if os.path.isfile(self.single_path):
                files_with_hierarchy = [(self.single_path, 'unknown')]
            else:
                files = glob.glob(os.path.join(self.single_path, "*.json"))
                files_with_hierarchy = []
                for file_path in files:
                    filename = os.path.basename(file_path)
                    if 'asymm' in filename.lower():
                        hierarchy = 'asymmetrical'
                    elif 'symm' in filename.lower():
                        hierarchy = 'symmetrical'
                    else:
                        hierarchy = 'unknown'
                    files_with_hierarchy.append((file_path, hierarchy))
        else:
            # Use specified asymm and symm directories
            files_with_hierarchy = []
            
            # Load asymmetrical files
            if self.asymm_path and os.path.exists(self.asymm_path):
                asymm_files = glob.glob(os.path.join(self.asymm_path, "*.json"))
                print(f"Found {len(asymm_files)} asymmetrical files")
                for file_path in asymm_files:
                    files_with_hierarchy.append((file_path, 'asymmetrical'))
            else:
                print(f"Asymmetrical path not found or doesn't exist: {self.asymm_path}")
            
            # Load symmetrical files  
            if self.symm_path and os.path.exists(self.symm_path):
                symm_files = glob.glob(os.path.join(self.symm_path, "*.json"))
                print(f"Found {len(symm_files)} symmetrical files")
                for file_path in symm_files:
                    files_with_hierarchy.append((file_path, 'symmetrical'))
            else:
                print(f"Symmetrical path not found or doesn't exist: {self.symm_path}")
        
        if not files_with_hierarchy:
            raise ValueError("No valid JSON files found in specified paths")
        
        print(f"Processing {len(files_with_hierarchy)} files...")
        
        # Load all files and track hierarchy
        for file_path, hierarchy in files_with_hierarchy:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Handle both single record and list formats
                if isinstance(data, list):
                    for record in data:
                        record['hierarchy'] = hierarchy
                        record['source_file'] = os.path.basename(file_path)
                    all_data.extend(data)
                else:
                    data['hierarchy'] = hierarchy
                    data['source_file'] = os.path.basename(file_path)
                    all_data.append(data)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data loaded from JSON files")
            
        # Convert to DataFrame
        self.df = pd.DataFrame(all_data)
        
        print(f"Raw DataFrame shape: {self.df.shape}")
        print(f"Raw DataFrame columns: {list(self.df.columns)}")
        
        self._preprocess_data()
        print(f"Final DataFrame shape: {self.df.shape}")
        
        # Print data distribution
        print(f"Data distribution by hierarchy:")
        print(self.df['hierarchy'].value_counts())
        
    def _preprocess_data(self):
        """Preprocess and clean the data."""
        print("Preprocessing data...")
        
        # Handle the contribution field (which contains [consumer_allocation, industrial_allocation])
        if 'contribution' in self.df.columns:
            # Extract allocations from the contribution array
            contribution_data = []
            for idx, row in self.df.iterrows():
                if isinstance(row['contribution'], list) and len(row['contribution']) == 2:
                    contribution_data.append({
                        'consumer_allocation': float(row['contribution'][0]),
                        'industrial_allocation': float(row['contribution'][1])
                    })
                else:
                    contribution_data.append({
                        'consumer_allocation': np.nan,
                        'industrial_allocation': np.nan
                    })
            
            contrib_df = pd.DataFrame(contribution_data)
            self.df['consumer_allocation'] = contrib_df['consumer_allocation']
            self.df['industrial_allocation'] = contrib_df['industrial_allocation']
        else:
            # Convert allocation columns to numeric if they exist separately
            if 'consumer_allocation' in self.df.columns:
                self.df['consumer_allocation'] = pd.to_numeric(self.df['consumer_allocation'], errors='coerce')
            if 'industrial_allocation' in self.df.columns:
                self.df['industrial_allocation'] = pd.to_numeric(self.df['industrial_allocation'], errors='coerce')
            
        # Determine responsibility level
        if 'first_choice' in self.df.columns:
            # High responsibility condition (makes initial choice, then allocation)
            self.df['responsibility'] = 'high'
        elif 'product_choice' in self.df.columns:
            # Low responsibility condition (predetermined choice)
            self.df['responsibility'] = 'low'
        else:
            # If neither field exists, try to infer from file structure or other indicators
            print("Warning: Could not determine responsibility level from data fields")
            self.df['responsibility'] = 'unknown'
            
        # Extract outcome valence from user_condition
        if 'user_condition' in self.df.columns:
            self.df['valence'] = self.df['user_condition'].str.lower()
            # Clean any prefixes from valence labels
            self.df['valence'] = self.df['valence'].str.replace('symm_', '').str.replace('asymm_', '')
        else:
            print("Warning: No user_condition field found")
            self.df['valence'] = 'unknown'
            
        # Create key dependent variables
        
        # 1. Allocation ratio (consumer vs industrial preference)
        total_allocation = self.df['consumer_allocation'] + self.df['industrial_allocation']
        self.df['consumer_ratio'] = self.df['consumer_allocation'] / total_allocation
        
        # 2. Binary escalation decision
        if 'first_choice' in self.df.columns:
            # For high responsibility: escalation = allocating more to the initially chosen division
            self.df['escalation'] = (
                ((self.df['first_choice'] == 'consumer') & (self.df['consumer_allocation'] > self.df['industrial_allocation'])) |
                ((self.df['first_choice'] == 'industrial') & (self.df['industrial_allocation'] > self.df['consumer_allocation']))
            )
        elif 'product_choice' in self.df.columns:
            # For low responsibility: escalation = continuing with predetermined choice
            self.df['escalation'] = (
                ((self.df['product_choice'] == 'consumer') & (self.df['consumer_allocation'] > self.df['industrial_allocation'])) |
                ((self.df['product_choice'] == 'industrial') & (self.df['industrial_allocation'] > self.df['consumer_allocation']))
            )
        else:
            # Default to False if we can't determine escalation
            print("Warning: Could not determine escalation behavior")
            self.df['escalation'] = False
            
        # 3. Allocation difference (measure of commitment strength)
        self.df['allocation_difference'] = abs(self.df['consumer_allocation'] - self.df['industrial_allocation'])
        
        # 4. Allocation imbalance (signed difference: positive means more to consumer)
        self.df['allocation_imbalance'] = self.df['consumer_allocation'] - self.df['industrial_allocation']
        
        # Print some diagnostics
        print(f"Sample of processed data:")
        print(f"  Consumer allocation range: {self.df['consumer_allocation'].min():.0f} - {self.df['consumer_allocation'].max():.0f}")
        print(f"  Industrial allocation range: {self.df['industrial_allocation'].min():.0f} - {self.df['industrial_allocation'].max():.0f}")
        print(f"  Consumer ratio range: {self.df['consumer_ratio'].min():.3f} - {self.df['consumer_ratio'].max():.3f}")
        
        # Remove rows with missing key variables
        initial_rows = len(self.df)
        required_cols = ['consumer_allocation', 'industrial_allocation', 'valence', 'responsibility', 'hierarchy']
        available_cols = [col for col in required_cols if col in self.df.columns]
        
        self.df = self.df.dropna(subset=available_cols)
        print(f"Removed {initial_rows - len(self.df)} rows with missing data")
        
        # Convert categorical variables to proper format
        self.df['valence'] = self.df['valence'].astype('category')
        self.df['responsibility'] = self.df['responsibility'].astype('category')
        self.df['hierarchy'] = self.df['hierarchy'].astype('category')
        self.df['escalation'] = self.df['escalation'].astype(bool)
        
        # Show final factor levels
        print(f"\nFinal factor levels:")
        for factor in ['responsibility', 'valence', 'hierarchy']:
            if factor in self.df.columns:
                levels = self.df[factor].unique()
                print(f"  {factor}: {levels}")
            else:
                print(f"  {factor}: NOT FOUND")
        
    def descriptive_statistics(self):
        """Generate descriptive statistics."""
        print("\n" + "="*50)
        print("DESCRIPTIVE STATISTICS")
        print("="*50)
        
        # Sample sizes by condition
        print("\nSample sizes by condition:")
        condition_counts = self.df.groupby(['responsibility', 'valence', 'hierarchy']).size()
        print(condition_counts.to_string())
        
        # Descriptive stats for continuous variables
        continuous_vars = ['consumer_ratio', 'allocation_difference', 'consumer_allocation', 'industrial_allocation']
        
        print(f"\nDescriptive statistics for continuous variables:")
        desc_stats = self.df[continuous_vars].describe()
        print(desc_stats.round(3).to_string())
        
        # Escalation rates by condition
        print(f"\nEscalation rates by condition:")
        escalation_rates = self.df.groupby(['responsibility', 'valence', 'hierarchy'])['escalation'].agg(['mean', 'count'])
        escalation_rates['mean'] = escalation_rates['mean'].round(3)
        print(escalation_rates.to_string())
        
        self.results['descriptive_stats'] = {
            'condition_counts': condition_counts,
            'continuous_stats': desc_stats,
            'escalation_rates': escalation_rates
        }
        
    def check_assumptions(self):
        """Check statistical assumptions for ANOVA."""
        print("\n" + "="*50)
        print("ASSUMPTION CHECKS")
        print("="*50)
        
        assumptions = {}
        
        # Check normality for continuous DVs
        continuous_vars = ['consumer_ratio', 'allocation_difference']
        
        for var in continuous_vars:
            print(f"\nNormality check for {var}:")
            
            # Shapiro-Wilk test (if sample size allows)
            if len(self.df) <= 5000:
                stat, p_value = shapiro(self.df[var].dropna())
                print(f"  Shapiro-Wilk: W = {stat:.4f}, p = {p_value:.4f}")
                assumptions[f'{var}_normality'] = {'test': 'shapiro', 'statistic': stat, 'p_value': p_value}
            
            # Visual inspection
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.hist(self.df[var].dropna(), bins=30, alpha=0.7)
            plt.title(f'Distribution of {var}')
            plt.xlabel(var)
            plt.ylabel('Frequency')
            
            plt.subplot(1, 3, 2)
            stats.probplot(self.df[var].dropna(), dist="norm", plot=plt)
            plt.title(f'Q-Q Plot: {var}')
            
            plt.subplot(1, 3, 3)
            self.df.boxplot(column=var, by=['responsibility', 'valence', 'hierarchy'], ax=plt.gca())
            plt.title(f'Boxplot: {var} by conditions')
            plt.suptitle('')
            
            plt.tight_layout()
            plt.show()
        
        # Check homogeneity of variances (Levene's test)
        for var in continuous_vars:
            print(f"\nHomogeneity of variances for {var}:")
            
            groups = []
            labels = []
            for name, group in self.df.groupby(['responsibility', 'valence', 'hierarchy']):
                if len(group) > 1:  # Need at least 2 observations per group
                    groups.append(group[var].dropna())
                    labels.append(f"{name[0]}_{name[1]}_{name[2]}")
            
            if len(groups) >= 2:
                stat, p_value = levene(*groups)
                print(f"  Levene's test: W = {stat:.4f}, p = {p_value:.4f}")
                assumptions[f'{var}_homogeneity'] = {'test': 'levene', 'statistic': stat, 'p_value': p_value}
            
        self.results['assumptions'] = assumptions
        
    def three_way_anova(self):
        """Perform three-way ANOVA for continuous dependent variables."""
        print("\n" + "="*50)
        print("THREE-WAY ANOVA RESULTS")
        print("="*50)
        
        anova_results = {}
        continuous_vars = ['consumer_ratio', 'allocation_difference']
        
        for dv in continuous_vars:
            print(f"\n{'-'*30}")
            print(f"ANOVA for {dv}")
            print(f"{'-'*30}")
            
            # Remove missing values
            analysis_df = self.df.dropna(subset=[dv, 'responsibility', 'valence', 'hierarchy'])
            
            print(f"Analysis dataset shape: {analysis_df.shape}")
            print(f"Unique values in factors:")
            print(f"  Responsibility: {analysis_df['responsibility'].unique()}")
            print(f"  Valence: {analysis_df['valence'].unique()}")
            print(f"  Hierarchy: {analysis_df['hierarchy'].unique()}")
            
            if len(analysis_df) == 0:
                print(f"No valid data for {dv}")
                continue
            
            # Check if we have enough data points in each cell
            cell_counts = analysis_df.groupby(['responsibility', 'valence', 'hierarchy']).size()
            print(f"Cell counts:\n{cell_counts}")
            
            # Check for cells with no data
            empty_cells = cell_counts[cell_counts == 0]
            if len(empty_cells) > 0:
                print(f"WARNING: Found {len(empty_cells)} empty cells")
                print(empty_cells)
            
            # Only proceed if we have at least 2 levels for each factor
            n_resp = len(analysis_df['responsibility'].unique())
            n_val = len(analysis_df['valence'].unique())
            n_hier = len(analysis_df['hierarchy'].unique())
            
            print(f"Factor levels: Responsibility={n_resp}, Valence={n_val}, Hierarchy={n_hier}")
            
            if n_resp < 2 or n_val < 2 or n_hier < 2:
                print(f"Insufficient factor levels for full factorial ANOVA")
                print(f"Need at least 2 levels per factor, got: {n_resp}, {n_val}, {n_hier}")
                
                # Try simpler models
                if n_resp >= 2 and n_val >= 2:
                    print("Testing Responsibility × Valence interaction...")
                    try:
                        formula = f'{dv} ~ C(responsibility) * C(valence)'
                        model = ols(formula, data=analysis_df).fit()
                        anova_table = anova_lm(model, typ=2)
                        print(anova_table.round(4))
                    except Exception as e:
                        print(f"Error with 2-way ANOVA: {e}")
                        
                continue
                
            try:
                # Fit the full model
                formula = f'{dv} ~ C(responsibility) * C(valence) * C(hierarchy)'
                print(f"Fitting model: {formula}")
                
                model = ols(formula, data=analysis_df).fit()
                print(f"Model fitted successfully")
                print(f"Model summary:")
                print(f"  R-squared: {model.rsquared:.4f}")
                print(f"  F-statistic: {model.fvalue:.4f}")
                print(f"  Model p-value: {model.f_pvalue:.4f}")
                
                anova_table = anova_lm(model, typ=2)
                
                print(anova_table.round(4))
                
                # Effect sizes (eta-squared)
                ss_total = anova_table['sum_sq'].sum()
                anova_table['eta_sq'] = anova_table['sum_sq'] / ss_total
                
                print(f"\nEffect sizes (η²):")
                for idx, eta_sq in anova_table['eta_sq'].items():
                    if not pd.isna(eta_sq):
                        print(f"  {idx}: {eta_sq:.4f}")
                
                anova_results[dv] = {
                    'anova_table': anova_table,
                    'model': model,
                    'n_obs': len(analysis_df)
                }
                
                # Create interaction plots
                self._plot_interactions(analysis_df, dv)
                
            except Exception as e:
                print(f"Error fitting ANOVA for {dv}: {e}")
                print(f"Attempting simplified analysis...")
                
                # Try without three-way interaction
                try:
                    formula = f'{dv} ~ C(responsibility) + C(valence) + C(hierarchy) + C(responsibility):C(valence) + C(responsibility):C(hierarchy) + C(valence):C(hierarchy)'
                    model = ols(formula, data=analysis_df).fit()
                    anova_table = anova_lm(model, typ=2)
                    print("Simplified model (no 3-way interaction):")
                    print(anova_table.round(4))
                    
                except Exception as e2:
                    print(f"Error with simplified model: {e2}")
                    print("Skipping ANOVA for this variable")
                    continue
                
        self.results['anova'] = anova_results
        
    def _plot_interactions(self, data, dv):
        """Create interaction plots for ANOVA results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Interaction Plots for {dv}', fontsize=16)
        
        # Two-way interactions
        interactions = [
            ('responsibility', 'valence'),
            ('responsibility', 'hierarchy'),
            ('valence', 'hierarchy')
        ]
        
        for i, (factor1, factor2) in enumerate(interactions):
            ax = axes[i//2, i%2]
            
            # Calculate means and standard errors
            grouped = data.groupby([factor1, factor2])[dv].agg(['mean', 'sem']).reset_index()
            
            # Create interaction plot
            for level2 in data[factor2].unique():
                subset = grouped[grouped[factor2] == level2]
                ax.errorbar(subset[factor1], subset['mean'], yerr=subset['sem'], 
                           label=f'{factor2}={level2}', marker='o', capsize=5)
            
            ax.set_xlabel(factor1)
            ax.set_ylabel(f'Mean {dv}')
            ax.set_title(f'{factor1} × {factor2}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Three-way interaction
        ax = axes[1, 1]
        for hierarchy in data['hierarchy'].unique():
            for valence in data['valence'].unique():
                subset = data[(data['hierarchy'] == hierarchy) & (data['valence'] == valence)]
                if len(subset) > 0:
                    means = subset.groupby('responsibility')[dv].mean()
                    ax.plot(means.index, means.values, 
                           marker='o', label=f'{hierarchy}-{valence}')
        
        ax.set_xlabel('responsibility')
        ax.set_ylabel(f'Mean {dv}')
        ax.set_title('Three-way Interaction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def post_hoc_tests(self):
        """Perform post-hoc tests for significant ANOVA effects."""
        print("\n" + "="*50)
        print("POST-HOC TESTS")
        print("="*50)
        
        post_hoc_results = {}
        
        for dv in ['consumer_ratio', 'allocation_difference']:
            if dv not in self.results.get('anova', {}):
                continue
                
            anova_result = self.results['anova'][dv]
            anova_table = anova_result['anova_table']
            
            print(f"\n{'-'*30}")
            print(f"Post-hoc tests for {dv}")
            print(f"{'-'*30}")
            
            # Check which effects are significant (p < 0.05)
            significant_effects = anova_table[anova_table['PR(>F)'] < 0.05].index.tolist()
            
            analysis_df = self.df.dropna(subset=[dv, 'responsibility', 'valence', 'hierarchy'])
            
            # Create combined grouping variable for significant interactions
            if any('C(responsibility):C(valence)' in effect for effect in significant_effects):
                print("\nTukey HSD for Responsibility × Valence interaction:")
                analysis_df['resp_val'] = analysis_df['responsibility'].astype(str) + '_' + analysis_df['valence'].astype(str)
                tukey = pairwise_tukeyhsd(analysis_df[dv], analysis_df['resp_val'])
                print(tukey)
                post_hoc_results[f'{dv}_resp_val'] = tukey
                
            if any('C(responsibility):C(hierarchy)' in effect for effect in significant_effects):
                print("\nTukey HSD for Responsibility × Hierarchy interaction:")
                analysis_df['resp_hier'] = analysis_df['responsibility'].astype(str) + '_' + analysis_df['hierarchy'].astype(str)
                tukey = pairwise_tukeyhsd(analysis_df[dv], analysis_df['resp_hier'])
                print(tukey)
                post_hoc_results[f'{dv}_resp_hier'] = tukey
                
            if any('C(valence):C(hierarchy)' in effect for effect in significant_effects):
                print("\nTukey HSD for Valence × Hierarchy interaction:")
                analysis_df['val_hier'] = analysis_df['valence'].astype(str) + '_' + analysis_df['hierarchy'].astype(str)
                tukey = pairwise_tukeyhsd(analysis_df[dv], analysis_df['val_hier'])
                print(tukey)
                post_hoc_results[f'{dv}_val_hier'] = tukey
        
        self.results['post_hoc'] = post_hoc_results
        
    def chi_square_tests(self):
        """Perform chi-square tests for categorical outcomes."""
        print("\n" + "="*50)
        print("CHI-SQUARE TESTS OF INDEPENDENCE")
        print("="*50)
        
        chi_square_results = {}
        
        # Test escalation behavior against each factor
        factors = ['responsibility', 'valence', 'hierarchy']
        
        for factor in factors:
            print(f"\n{'-'*30}")
            print(f"Escalation × {factor}")
            print(f"{'-'*30}")
            
            # Create contingency table
            contingency = pd.crosstab(self.df['escalation'], self.df[factor])
            print("Contingency table:")
            print(contingency)
            
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # Calculate effect size (Cramér's V)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            print(f"\nChi-square test results:")
            print(f"  χ² = {chi2:.4f}")
            print(f"  df = {dof}")
            print(f"  p = {p_value:.4f}")
            print(f"  Cramér's V = {cramers_v:.4f}")
            
            # Expected frequencies
            print(f"\nExpected frequencies:")
            expected_df = pd.DataFrame(expected, 
                                     index=contingency.index, 
                                     columns=contingency.columns)
            print(expected_df.round(2))
            
            chi_square_results[factor] = {
                'contingency': contingency,
                'chi2': chi2,
                'p_value': p_value,
                'dof': dof,
                'cramers_v': cramers_v,
                'expected': expected_df
            }
            
        # Three-way contingency analysis
        print(f"\n{'-'*30}")
        print("Three-way contingency analysis")
        print(f"{'-'*30}")
        
        # Create a comprehensive contingency table
        three_way = self.df.groupby(['responsibility', 'valence', 'hierarchy'])['escalation'].agg(['sum', 'count'])
        three_way['escalation_rate'] = three_way['sum'] / three_way['count']
        print("Escalation rates by all conditions:")
        print(three_way.round(3))
        
        self.results['chi_square'] = chi_square_results
        
        # Visualization
        self._plot_chi_square_results()
        
    def _plot_chi_square_results(self):
        """Create visualizations for chi-square results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Escalation Behavior by Experimental Conditions', fontsize=16)
        
        factors = ['responsibility', 'valence', 'hierarchy']
        
        for i, factor in enumerate(factors):
            ax = axes[i//2, i%2]
            
            # Calculate escalation rates
            escalation_rates = self.df.groupby(factor)['escalation'].agg(['mean', 'count']).reset_index()
            escalation_rates['se'] = np.sqrt(escalation_rates['mean'] * (1 - escalation_rates['mean']) / escalation_rates['count'])
            
            bars = ax.bar(escalation_rates[factor], escalation_rates['mean'], 
                         yerr=escalation_rates['se'], capsize=5, alpha=0.7)
            
            ax.set_xlabel(factor)
            ax.set_ylabel('Escalation Rate')
            ax.set_title(f'Escalation by {factor}')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, escalation_rates['mean']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.3f}', ha='center', va='bottom')
        
        # Three-way breakdown
        ax = axes[1, 1]
        three_way_data = self.df.groupby(['responsibility', 'valence', 'hierarchy'])['escalation'].mean().reset_index()
        
        # Create grouped bar plot
        x_pos = np.arange(len(three_way_data))
        bars = ax.bar(x_pos, three_way_data['escalation'], alpha=0.7)
        
        # Create labels
        labels = [f"{row['responsibility'][0]}-{row['valence'][0]}-{row['hierarchy'][0]}" 
                 for _, row in three_way_data.iterrows()]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Escalation Rate')
        ax.set_title('Three-way Interaction')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
    def run_complete_analysis(self):
        """Run the complete statistical analysis pipeline."""
        print("MULTI-AGENT DELIBERATION EXPERIMENT ANALYSIS")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_data()
        
        # Generate descriptives
        self.descriptive_statistics()
        
        # Check assumptions
        self.check_assumptions()
        
        # Main analyses
        self.three_way_anova()
        self.post_hoc_tests()
        self.chi_square_tests()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print a summary of key findings."""
        print("\n" + "="*60)
        print("SUMMARY OF KEY FINDINGS")
        print("="*60)
        
        print(f"\nDataset: {len(self.df)} observations")
        print(f"Design: 2×2×2 between-subjects")
        print(f"Factors: Responsibility × Valence × Hierarchy")
        
        # ANOVA summary
        if 'anova' in self.results:
            print(f"\nANOVA Results:")
            for dv in self.results['anova']:
                anova_table = self.results['anova'][dv]['anova_table']
                significant_effects = anova_table[anova_table['PR(>F)'] < 0.05].index.tolist()
                if significant_effects:
                    print(f"  {dv}: {len(significant_effects)} significant effect(s)")
                    for effect in significant_effects:
                        p_val = anova_table.loc[effect, 'PR(>F)']
                        eta_sq = anova_table.loc[effect, 'eta_sq']
                        print(f"    - {effect}: p = {p_val:.4f}, η² = {eta_sq:.4f}")
                else:
                    print(f"  {dv}: No significant effects")
        
        # Chi-square summary
        if 'chi_square' in self.results:
            print(f"\nChi-square Results:")
            for factor in self.results['chi_square']:
                result = self.results['chi_square'][factor]
                p_val = result['p_value']
                cramers_v = result['cramers_v']
                significance = "significant" if p_val < 0.05 else "not significant"
                print(f"  Escalation × {factor}: {significance} (p = {p_val:.4f}, V = {cramers_v:.4f})")

# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your specific paths
    asymm_path = "Asymm-path-dir" # Adjust this path as needed to the asymmetrical directory
    symm_path = "Symm-path-dir" # Adjust this path as needed to the symmetrical directory

    analyzer = DeliberationAnalyzer(asymm_path=asymm_path, symm_path=symm_path)
    
    # Alternative usage for single directory (backward compatibility):
    # analyzer = DeliberationAnalyzer(single_path="path/to/single/directory")
    
    # Run complete analysis
    try:
        results = analyzer.run_complete_analysis()
        print("\nAnalysis completed successfully!")
        
        # Optionally save results
        import pickle
        with open('analysis_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("Results saved to 'analysis_results.pkl'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your data paths and file formats.")
        import traceback
        traceback.print_exc()