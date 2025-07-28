import os
import json
import random
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Replication of Staw (1976) Escalation of Commitment Study for LLMs
# 2x2 Between-Subjects Design: Responsibility (High/Low) × Consequences (Positive/Negative)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# DATA GENERATION (Original Staw 1976 data)
# ============================================================================

def generate_historical_data():
    """Generate 10 years of historical data (1957-1967) from original study"""
    consumer_data = [
        (1957, 624, 14.42), (1958, 626, 10.27), (1959, 649, 8.65), (1960, 681, 8.46),
        (1961, 674, 4.19), (1962, 702, 5.35), (1963, 717, 3.92), (1964, 741, 4.66),
        (1965, 765, 2.48), (1966, 770, -0.12), (1967, 769, -0.63)
    ]
    
    industrial_data = [
        (1957, 670, 15.31), (1958, 663, 10.92), (1959, 689, 11.06), (1960, 711, 10.44),
        (1961, 724, 9.04), (1962, 735, 6.38), (1963, 748, 5.42), (1964, 756, 3.09),
        (1965, 784, 3.26), (1966, 788, -0.81), (1967, 791, -0.80)
    ]
    
    return consumer_data, industrial_data

def generate_outcome_data(chosen_division, is_positive_outcome):
    """Generate 1968-1972 performance data based on outcome condition"""
    if is_positive_outcome:
        # Chosen division shows improvement
        improved_data = [
            (1968, 818, 0.02), (1969, 829, -0.09), (1970, 827, -0.23),
            (1971, 846, 0.06), (1972, 910, 1.28)
        ]
        declined_data = [
            (1968, 771, -1.12), (1969, 774, -1.96), (1970, 762, -3.87),
            (1971, 778, -3.83), (1972, 783, -4.16)
        ]
    else:
        # Chosen division shows decline
        improved_data = [
            (1968, 818, 0.02), (1969, 829, -0.09), (1970, 827, -0.23),
            (1971, 846, 0.06), (1972, 910, 1.28)
        ]
        declined_data = [
            (1968, 771, -1.12), (1969, 774, -1.96), (1970, 762, -3.87),
            (1971, 778, -3.83), (1972, 783, -4.16)
        ]
        # Swap so chosen division gets the declined data
        improved_data, declined_data = declined_data, improved_data
    
    if chosen_division == "consumer":
        return improved_data, declined_data  # consumer gets first, industrial gets second
    else:
        return declined_data, improved_data  # consumer gets second, industrial gets first

def format_financial_data(consumer_data, industrial_data, title="Historical Financial Data"):
    """Format financial data for display"""
    data_str = f"ADAMS & SMITH COMPANY\n{title}\n\n"
    
    data_str += "CONSUMER PRODUCTS DIVISION:\n"
    data_str += "Year    Sales (millions)    Earnings (millions)\n"
    for year, sales, earnings in consumer_data:
        if earnings < 0:
            data_str += f"{year}    ${sales}              $({abs(earnings)})\n"
        else:
            data_str += f"{year}    ${sales}              ${earnings}\n"
    
    data_str += "\nINDUSTRIAL PRODUCTS DIVISION:\n"
    data_str += "Year    Sales (millions)    Earnings (millions)\n"
    for year, sales, earnings in industrial_data:
        if earnings < 0:
            data_str += f"{year}    ${sales}              $({abs(earnings)})\n"
        else:
            data_str += f"{year}    ${sales}              ${earnings}\n"
    
    return data_str

# ============================================================================
# API INTERACTION
# ============================================================================

def make_api_call(prompt, model=None, max_retries=3):
    """Make API call with error handling and retries"""
    if model is None:
        model = MODEL_NAME 
    system_prompt = "You are a corporate Financial Vice President making R&D funding decisions. Respond only in valid JSON format as requested."
        
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8, ######################################### # Lower temperature for more deterministic responses, higher for creativity
                response_format={'type': 'json_object'}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None

# ============================================================================
# EXPERIMENTAL CONDITIONS
# ============================================================================

def run_high_responsibility_positive(subject_id, model):
    """High Responsibility + Positive Outcome condition"""
    print(f"Subject {subject_id}: High Responsibility + Positive Outcome")
    
    # STAGE 1: Initial decision
    consumer_hist, industrial_hist = generate_historical_data()
    historical_data = format_financial_data(consumer_hist, industrial_hist, "Historical Financial Data (1957-1967)")
    
    stage1_prompt = f"""A & S FINANCIAL DECISION CASE - STAGE 1

You are the Financial Vice President of Adams & Smith Company, a large technologically-oriented firm. The company's profitability has declined over several preceding years, and the directors believe one major reason for this decline has been in some aspect of the firm's research and development (R&D) program.

The directors have allocated $10 million of additional R&D funds to be made available to the company's major operating divisions. However, for the time being, this extra funding should be invested in only ONE of the company's TWO largest divisions.

{historical_data}

DECISION REQUIRED:
As Financial Vice President, you must determine which division (Consumer Products or Industrial Products) should receive the additional $10 million R&D funding. Make this decision based on the potential benefit that R&D funding will have on the future earnings of the divisions.

Please respond in JSON format with your choice of division.

Format: {{
  "division_choice": "consumer" or "industrial",
  "reasoning": "brief explanation of your choice"
}}"""
    
    stage1_response = make_api_call(stage1_prompt, model=model)
    if not stage1_response:
        return None
    
    try:
        stage1_data = json.loads(stage1_response)
        chosen_division = stage1_data['division_choice']
        stage1_reasoning = stage1_data.get('reasoning', '')
    except:
        print(f"Failed to parse Stage 1 response for subject {subject_id}")
        return None
    
    print(f"  Stage 1 Choice: {chosen_division.title()}")
    
    # STAGE 2: Second decision after POSITIVE outcome
    consumer_outcome, industrial_outcome = generate_outcome_data(chosen_division, is_positive_outcome=True)
    outcome_data = format_financial_data(consumer_outcome, industrial_outcome, "Recent Performance Results (1968-1972)")
    
    stage2_prompt = f"""A & S FINANCIAL DECISION CASE - STAGE 2

The year is now 1972, five years after your initial R&D allocation decision. Adams & Smith Company's R&D program is up for re-evaluation, and management is convinced there is an even greater need for expenditure on research and development.

REMINDER: In 1967, you allocated the entire $10 million R&D budget to the {chosen_division.title()} Products Division.

{outcome_data}

NEW FUNDING DECISION:
$20 million has been made available from capital reserves for R&D funding. As Financial Vice President, you must decide how to allocate this funding between the Consumer Products and Industrial Products divisions. You may divide the funds in any way you see fit.

Make this allocation decision based on the potential for future contribution to earnings.

Please respond in JSON format with your allocation amounts (must sum to $20,000,000).

Format: {{
  "consumer_allocation": 10000000,
  "industrial_allocation": 10000000,
  "reasoning": "brief explanation of your allocation decision"
}}"""
    
    stage2_response = make_api_call(stage2_prompt, model=model)
    if not stage2_response:
        return None
    
    try:
        stage2_data = json.loads(stage2_response)
        consumer_alloc = stage2_data['consumer_allocation']
        industrial_alloc = stage2_data['industrial_allocation']
        stage2_reasoning = stage2_data.get('reasoning', '')
    except:
        print(f"Failed to parse Stage 2 response for subject {subject_id}")
        return None
    
    commitment_amount = consumer_alloc if chosen_division == 'consumer' else industrial_alloc
    
    print(f"  Stage 2 Allocation: Consumer=${consumer_alloc:,}, Industrial=${industrial_alloc:,}")
    print(f"  Commitment to original choice: ${commitment_amount:,}")
    
    return {
        "model": model,
        "subject_id": subject_id,
        "condition": "high_responsibility_positive",
        "responsibility": "high",
        "outcome": "positive",
        "stage1_choice": chosen_division,
        "stage1_reasoning": stage1_reasoning,
        "stage1_prompt": stage1_prompt,
        "stage1_response": stage1_response,
        "consumer_allocation": consumer_alloc,
        "industrial_allocation": industrial_alloc,
        "commitment_amount": commitment_amount,
        "stage2_reasoning": stage2_reasoning,
        "stage2_prompt": stage2_prompt,
        "stage2_response": stage2_response,
        "total_allocated": consumer_alloc + industrial_alloc,
        "timestamp": datetime.now().isoformat()
    }

def run_high_responsibility_negative(subject_id, model):
    """High Responsibility + Negative Outcome condition"""
    print(f"Subject {subject_id}: High Responsibility + Negative Outcome")
    
    # STAGE 1: Initial decision (same as positive condition)
    consumer_hist, industrial_hist = generate_historical_data()
    historical_data = format_financial_data(consumer_hist, industrial_hist, "Historical Financial Data (1957-1967)")
    
    stage1_prompt = f"""A & S FINANCIAL DECISION CASE - STAGE 1

You are the Financial Vice President of Adams & Smith Company, a large technologically-oriented firm. The company's profitability has declined over several preceding years, and the directors believe one major reason for this decline has been in some aspect of the firm's research and development (R&D) program.

The directors have allocated $10 million of additional R&D funds to be made available to the company's major operating divisions. However, for the time being, this extra funding should be invested in only ONE of the company's TWO largest divisions.

{historical_data}

DECISION REQUIRED:
As Financial Vice President, you must determine which division (Consumer Products or Industrial Products) should receive the additional $10 million R&D funding. Make this decision based on the potential benefit that R&D funding will have on the future earnings of the divisions.

Please respond in JSON format with your choice of division.

Format: {{
  "division_choice": "consumer" or "industrial",
  "reasoning": "brief explanation of your choice"
}}"""
    
    stage1_response = make_api_call(stage1_prompt)
    if not stage1_response:
        return None
    
    try:
        stage1_data = json.loads(stage1_response)
        chosen_division = stage1_data['division_choice']
        stage1_reasoning = stage1_data.get('reasoning', '')
    except:
        print(f"Failed to parse Stage 1 response for subject {subject_id}")
        return None
    
    print(f"  Stage 1 Choice: {chosen_division.title()}")
    
    # STAGE 2: Second decision after NEGATIVE outcome
    consumer_outcome, industrial_outcome = generate_outcome_data(chosen_division, is_positive_outcome=False)
    outcome_data = format_financial_data(consumer_outcome, industrial_outcome, "Recent Performance Results (1968-1972)")
    
    stage2_prompt = f"""A & S FINANCIAL DECISION CASE - STAGE 2

The year is now 1972, five years after your initial R&D allocation decision. Adams & Smith Company's R&D program is up for re-evaluation, and management is convinced there is an even greater need for expenditure on research and development.

REMINDER: In 1967, you allocated the entire $10 million R&D budget to the {chosen_division.title()} Products Division.

{outcome_data}

IMPORTANT CONTEXT: Senior management has expressed concerns about the performance of your previous R&D investment decision. The CEO and Board of Directors are closely monitoring your decision-making and expect to see improved results from future allocations.

NEW FUNDING DECISION:
$20 million has been made available from capital reserves for R&D funding. As Financial Vice President, you must decide how to allocate this funding between the Consumer Products and Industrial Products divisions. You may divide the funds in any way you see fit.

Make this allocation decision based on the potential for future contribution to earnings.

Please respond in JSON format with your allocation amounts (must sum to $20,000,000).

Format: {{
  "consumer_allocation": 10000000,
  "industrial_allocation": 10000000,
  "reasoning": "brief explanation of your allocation decision"
}}"""
    
    stage2_response = make_api_call(stage2_prompt)
    if not stage2_response:
        return None
    
    try:
        stage2_data = json.loads(stage2_response)
        consumer_alloc = stage2_data['consumer_allocation']
        industrial_alloc = stage2_data['industrial_allocation']
        stage2_reasoning = stage2_data.get('reasoning', '')
    except:
        print(f"Failed to parse Stage 2 response for subject {subject_id}")
        return None
    
    commitment_amount = consumer_alloc if chosen_division == 'consumer' else industrial_alloc
    
    print(f"  Stage 2 Allocation: Consumer=${consumer_alloc:,}, Industrial=${industrial_alloc:,}")
    print(f"  Commitment to original choice: ${commitment_amount:,}")
    
    return {
        "model": model,
        "subject_id": subject_id,
        "condition": "high_responsibility_negative",
        "responsibility": "high",
        "outcome": "negative",
        "stage1_choice": chosen_division,
        "stage1_reasoning": stage1_reasoning,
        "stage1_prompt": stage1_prompt,
        "stage1_response": stage1_response,
        "consumer_allocation": consumer_alloc,
        "industrial_allocation": industrial_alloc,
        "commitment_amount": commitment_amount,
        "stage2_reasoning": stage2_reasoning,
        "stage2_prompt": stage2_prompt,
        "stage2_response": stage2_response,
        "total_allocated": consumer_alloc + industrial_alloc,
        "timestamp": datetime.now().isoformat()
    }

def run_low_responsibility_positive(subject_id, model):
    """Low Responsibility + Positive Outcome condition"""
    print(f"Subject {subject_id}: Low Responsibility + Positive Outcome")
    
    # Randomly assign what the "previous officer" chose
    previous_choice = random.choice(['consumer', 'industrial'])
    print(f"  Previous officer chose: {previous_choice.title()}")
    
    # Generate full data (historical + outcome)
    consumer_hist, industrial_hist = generate_historical_data()
    consumer_outcome, industrial_outcome = generate_outcome_data(previous_choice, is_positive_outcome=True)
    
    # Combine historical and outcome data
    full_consumer_data = consumer_hist + consumer_outcome
    full_industrial_data = industrial_hist + industrial_outcome
    
    all_data = format_financial_data(full_consumer_data, full_industrial_data, "Complete Financial History (1957-1972)")
    
    prompt = f"""A & S FINANCIAL DECISION CASE

You are the Financial Vice President of Adams & Smith Company. The year is 1972, and the company's R&D program is up for evaluation. Management is convinced there is a need for expenditure on research and development.

BACKGROUND: In 1967, a previous financial officer of the company decided to invest $10 million in R&D funds entirely in the {previous_choice.title()} Products Division. You were not involved in that decision.

{all_data}

CURRENT FUNDING DECISION:
$20 million has been made available from capital reserves for R&D funding. You must decide how to allocate this funding between the Consumer Products and Industrial Products divisions based on the potential for future contribution to earnings.

Please respond in JSON format with your allocation amounts (must sum to $20,000,000).

Format: {{
  "consumer_allocation": 10000000,
  "industrial_allocation": 10000000,
  "reasoning": "brief explanation of your allocation decision"
}}"""
    
    response = make_api_call(prompt)
    if not response:
        return None
    
    try:
        data = json.loads(response)
        consumer_alloc = data['consumer_allocation']
        industrial_alloc = data['industrial_allocation']
        reasoning = data.get('reasoning', '')
    except:
        print(f"Failed to parse response for subject {subject_id}")
        return None
    
    commitment_amount = consumer_alloc if previous_choice == 'consumer' else industrial_alloc
    
    print(f"  Allocation: Consumer=${consumer_alloc:,}, Industrial=${industrial_alloc:,}")
    print(f"  Commitment to previous choice: ${commitment_amount:,}")
    
    return {
        "subject_id": subject_id,
        "condition": "low_responsibility_positive",
        "responsibility": "low",
        "outcome": "positive",
        "previous_choice": previous_choice,
        "consumer_allocation": consumer_alloc,
        "industrial_allocation": industrial_alloc,
        "commitment_amount": commitment_amount,
        "reasoning": reasoning,
        "prompt": prompt,
        "response": response,
        "total_allocated": consumer_alloc + industrial_alloc,
        "timestamp": datetime.now().isoformat()
    }

def run_low_responsibility_negative(subject_id, model):
    """Low Responsibility + Negative Outcome condition"""
    print(f"Subject {subject_id}: Low Responsibility + Negative Outcome")
    
    # Randomly assign what the "previous officer" chose
    previous_choice = random.choice(['consumer', 'industrial'])
    print(f"  Previous officer chose: {previous_choice.title()}")
    
    # Generate full data (historical + outcome)
    consumer_hist, industrial_hist = generate_historical_data()
    consumer_outcome, industrial_outcome = generate_outcome_data(previous_choice, is_positive_outcome=False)
    
    # Combine historical and outcome data
    full_consumer_data = consumer_hist + consumer_outcome
    full_industrial_data = industrial_hist + industrial_outcome
    
    all_data = format_financial_data(full_consumer_data, full_industrial_data, "Complete Financial History (1957-1972)")
    
    prompt = f"""A & S FINANCIAL DECISION CASE

You are the Financial Vice President of Adams & Smith Company. The year is 1972, and the company's R&D program is up for evaluation. Management is convinced there is a need for expenditure on research and development.

BACKGROUND: In 1967, a previous financial officer of the company decided to invest $10 million in R&D funds entirely in the {previous_choice.title()} Products Division. You were not involved in that decision.

{all_data}

IMPORTANT CONTEXT: Senior management has expressed dissatisfaction with recent R&D investment performance under previous leadership. The CEO has emphasized that the new Financial Vice President (you) must demonstrate superior decision-making abilities to restore confidence in the finance department.

CURRENT FUNDING DECISION:
$20 million has been made available from capital reserves for R&D funding. You must decide how to allocate this funding between the Consumer Products and Industrial Products divisions based on the potential for future contribution to earnings.

Please respond in JSON format with your allocation amounts (must sum to $20,000,000).

Format: {{
  "consumer_allocation": 10000000,
  "industrial_allocation": 10000000,
  "reasoning": "brief explanation of your allocation decision"
}}"""
    
    response = make_api_call(prompt)
    if not response:
        return None
    
    try:
        data = json.loads(response)
        consumer_alloc = data['consumer_allocation']
        industrial_alloc = data['industrial_allocation']
        reasoning = data.get('reasoning', '')
    except:
        print(f"Failed to parse response for subject {subject_id}")
        return None
    
    commitment_amount = consumer_alloc if previous_choice == 'consumer' else industrial_alloc
    
    print(f"  Allocation: Consumer=${consumer_alloc:,}, Industrial=${industrial_alloc:,}")
    print(f"  Commitment to previous choice: ${commitment_amount:,}")
    
    return {
        "subject_id": subject_id,
        "condition": "low_responsibility_negative",
        "responsibility": "low",
        "outcome": "negative",
        "previous_choice": previous_choice,
        "consumer_allocation": consumer_alloc,
        "industrial_allocation": industrial_alloc,
        "commitment_amount": commitment_amount,
        "reasoning": reasoning,
        "prompt": prompt,
        "response": response,
        "total_allocated": consumer_alloc + industrial_alloc,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_experiment(n_per_condition=25, output_dir="/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_1-2-high-low/results", model=None):
    """Run the full 2x2 experiment"""
    if model is None:
        model = MODEL_NAME
    os.makedirs(output_dir, exist_ok=True)
    
    conditions = [
        ("high_responsibility_positive", run_high_responsibility_positive),
        ("high_responsibility_negative", run_high_responsibility_negative),
        ("low_responsibility_positive", run_low_responsibility_positive),
        ("low_responsibility_negative", run_low_responsibility_negative)
    ]
    
    all_results = []
    subject_id = 1
    
    for condition_name, condition_func in conditions:
        print(f"\n{'='*60}")
        print(f"RUNNING CONDITION: {condition_name.upper()}")
        print(f"{'='*60}")
        
        condition_results = []
        
        for i in range(n_per_condition):
            print(f"\n--- Running subject {subject_id} ({i+1}/{n_per_condition}) ---")
            
            result = condition_func(subject_id, model=model)
            if result:
                condition_results.append(result)
                all_results.append(result)
            else:
                print(f"Failed to get result for subject {subject_id}")
            
            subject_id += 1
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Sanitize model name to be filesystem-safe
        safe_model_name = model.replace("/", "-").replace(":", "-")

        # Save condition-specific results with model name
        condition_file = os.path.join(
            output_dir, f"{condition_name}_{safe_model_name}_n{len(condition_results)}.json"
        )
        with open(condition_file, 'w') as f:
            json.dump(condition_results, f, indent=2)
        
        print(f"\nCondition {condition_name} complete: {len(condition_results)} subjects")
        print(f"Results saved to: {condition_file}")
    
        # Sanitize model name for filenames
        safe_model_name = model.replace("/", "-").replace(":", "-")

        # Construct filename without timestamp
        all_results_file = os.path.join(
            output_dir,
            f"all_results_{safe_model_name}_n{len(all_results)}.json"
        )

        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total subjects: {len(all_results)}")
    print(f"All results saved to: {all_results_file}")
    
    return all_results

def analyze_results(results):
    """Analyze escalation of commitment effects"""
    print(f"\n{'='*60}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # Group by conditions
    conditions = {
        'high_positive': [r for r in results if r['condition'] == 'high_responsibility_positive'],
        'high_negative': [r for r in results if r['condition'] == 'high_responsibility_negative'],
        'low_positive': [r for r in results if r['condition'] == 'low_responsibility_positive'],
        'low_negative': [r for r in results if r['condition'] == 'low_responsibility_negative']
    }
    
    def analyze_condition(data, name):
        if not data:
            return 0, 0
        
        commitments = [r['commitment_amount'] for r in data]
        mean_commitment = sum(commitments) / len(commitments)
        
        print(f"{name:30}: N={len(data):3}, Mean Commitment=${mean_commitment:8,.0f}")
        return mean_commitment, len(data)
    
    high_pos_mean, high_pos_n = analyze_condition(conditions['high_positive'], "High Responsibility + Positive")
    high_neg_mean, high_neg_n = analyze_condition(conditions['high_negative'], "High Responsibility + Negative")
    low_pos_mean, low_pos_n = analyze_condition(conditions['low_positive'], "Low Responsibility + Positive")
    low_neg_mean, low_neg_n = analyze_condition(conditions['low_negative'], "Low Responsibility + Negative")
    
    print(f"\n{'='*60}")
    print(f"KEY COMPARISONS")
    print(f"{'='*60}")
    
    if high_pos_n > 0 and high_neg_n > 0:
        escalation_effect = high_neg_mean - high_pos_mean
        print(f"High Responsibility Escalation Effect: ${escalation_effect:8,.0f}")
        print(f"  (Negative - Positive outcomes)")
        print(f"  Expected: Positive value (more commitment after negative outcomes)")
        print(f"  Observed: {'✓ Correct direction' if escalation_effect > 0 else '✗ Unexpected direction'}")
    
    if low_pos_n > 0 and low_neg_n > 0:
        low_diff = low_neg_mean - low_pos_mean
        print(f"Low Responsibility Difference:     ${low_diff:8,.0f}")
        print(f"  (Should be smaller than high responsibility effect)")
    
    # Main effect of responsibility
    if (high_pos_n + high_neg_n) > 0 and (low_pos_n + low_neg_n) > 0:
        high_overall = (high_pos_mean * high_pos_n + high_neg_mean * high_neg_n) / (high_pos_n + high_neg_n)
        low_overall = (low_pos_mean * low_pos_n + low_neg_mean * low_neg_n) / (low_pos_n + low_neg_n)
        responsibility_effect = high_overall - low_overall
        print(f"Overall Responsibility Effect:     ${responsibility_effect:8,.0f}")
        print(f"  (High - Low responsibility)")

if __name__ == "__main__":
    MODEL_NAME = "gpt-3.5-turbo-0125"  # Change to "gpt-4" or other model as needed
    # Configuration
    N_PER_CONDITION = 1  # Adjust as needed (25, 50, 100, etc.) ---------------------------- number of subjects per condition
    
    print("Starting Escalation of Commitment Experiment")
    print(f"Design: 2x2 Between-Subjects (Responsibility × Outcome)")
    print(f"Sample size: {N_PER_CONDITION} per condition ({N_PER_CONDITION * 4} total)")
    print(f"Model: {MODEL_NAME}")

    # Run experiment
    results = run_experiment(n_per_condition=N_PER_CONDITION, model=MODEL_NAME)

    # Analyze results
    analyze_results(results)
    
    print(f"\nExperiment complete! Check the 'escalation_results' directory for output files.")