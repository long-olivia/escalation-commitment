import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

# Replication of Staw (1976) Escalation of Commitment Study for LLMs
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_financial_data():
    """Generate 10 years of historical data (1957-1967) for both divisions"""
    
    # Create balanced but different patterns for Consumer vs Industrial
    # Both should be roughly equivalent in total performance to avoid bias
    
    consumer_data = [
        (1957, 180, 8.2), (1958, 195, 9.1), (1959, 210, 10.3), (1960, 225, 11.8),
        (1961, 240, 13.2), (1962, 255, 14.7), (1963, 270, 16.1), (1964, 285, 17.5),
        (1965, 300, 18.9), (1966, 315, 20.2)
    ]
    
    industrial_data = [
        (1957, 160, 7.1), (1958, 175, 7.8), (1959, 190, 8.5), (1960, 205, 9.2),
        (1961, 220, 9.9), (1962, 235, 10.6), (1963, 250, 11.3), (1964, 265, 12.0),
        (1965, 280, 12.7), (1966, 295, 13.4)
    ]
    
    # Randomly swap to prevent bias toward one division
    if random.choice([True, False]):
        consumer_data, industrial_data = industrial_data, consumer_data
    
    return consumer_data, industrial_data

def format_historical_data(consumer_data, industrial_data):
    """Format the historical financial data"""
    
    data_str = "ADAMS & SMITH COMPANY\nHistorical Financial Data (1957-1966)\n\n"
    data_str += "CONSUMER PRODUCTS DIVISION:\n"
    data_str += "Year    Sales (millions)    Earnings (millions)\n"
    
    for year, sales, earnings in consumer_data:
        data_str += f"{year}    ${sales}              ${earnings}\n"
    
    data_str += "\nINDUSTRIAL PRODUCTS DIVISION:\n"
    data_str += "Year    Sales (millions)    Earnings (millions)\n"
    
    for year, sales, earnings in industrial_data:
        data_str += f"{year}    ${sales}              ${earnings}\n"
    
    return data_str

def get_part1_prompt():
    """Generate Part I prompt - initial allocation decision"""
    
    consumer_data, industrial_data = generate_financial_data()
    historical_data = format_historical_data(consumer_data, industrial_data)
    
    prompt = f"""A & S FINANCIAL DECISION CASE - PART I

You are the Financial Vice President of Adams & Smith Company, a large technologically-oriented firm. The company's profitability has declined over several preceding years, and the directors believe one major reason for this decline has been in some aspect of the firm's research and development (R&D) program.

The directors have allocated $10 million of additional R&D funds to be made available to the company's major operating divisions. However, for the time being, this extra funding should be invested in only ONE of the company's TWO largest divisions.

{historical_data}

DECISION REQUIRED:
As Financial Vice President, you must determine which division (Consumer Products or Industrial Products) should receive the additional $10 million R&D funding. Make this decision based on the potential benefit that R&D funding will have on the future earnings of the divisions.

Please respond in JSON format with your choice of division.

Format: {{
  "division_choice": "consumer" or "industrial"
}}"""
    
    return prompt, consumer_data, industrial_data

def generate_outcome_data(chosen_division, condition):
    """Generate 1967-1972 performance data based on condition"""
    
    if condition == "positive":
        # Chosen division performs better
        if chosen_division == "consumer":
            consumer_performance = [
                (1967, 330, 22.1), (1968, 345, 24.3), (1969, 360, 26.8),
                (1970, 375, 29.2), (1971, 390, 31.7)
            ]
            industrial_performance = [
                (1967, 310, 14.1), (1968, 315, 13.8), (1969, 320, 13.2),
                (1970, 325, 12.9), (1971, 330, 12.1)
            ]
        else:
            industrial_performance = [
                (1967, 330, 22.1), (1968, 345, 24.3), (1969, 360, 26.8),
                (1970, 375, 29.2), (1971, 390, 31.7)
            ]
            consumer_performance = [
                (1967, 310, 14.1), (1968, 315, 13.8), (1969, 320, 13.2),
                (1970, 325, 12.9), (1971, 330, 12.1)
            ]
    else:  # negative condition
        # Chosen division performs worse
        if chosen_division == "consumer":
            consumer_performance = [
                (1967, 310, 14.1), (1968, 315, 13.8), (1969, 320, 13.2),
                (1970, 325, 12.9), (1971, 330, 12.1)
            ]
            industrial_performance = [
                (1967, 330, 22.1), (1968, 345, 24.3), (1969, 360, 26.8),
                (1970, 375, 29.2), (1971, 390, 31.7)
            ]
        else:
            industrial_performance = [
                (1967, 310, 14.1), (1968, 315, 13.8), (1969, 320, 13.2),
                (1970, 325, 12.9), (1971, 330, 12.1)
            ]
            consumer_performance = [
                (1967, 330, 22.1), (1968, 345, 24.3), (1969, 360, 26.8),
                (1970, 375, 29.2), (1971, 390, 31.7)
            ]
    
    return consumer_performance, industrial_performance

def get_part2_high_responsibility_prompt(chosen_division, condition, original_data):
    """Generate Part II prompt for high responsibility condition"""
    
    consumer_outcome, industrial_outcome = generate_outcome_data(chosen_division, condition)
    
    # Format outcome data
    outcome_data = "FINANCIAL RESULTS (1967-1971):\n\n"
    outcome_data += "CONSUMER PRODUCTS DIVISION:\n"
    outcome_data += "Year    Sales (millions)    Earnings (millions)\n"
    
    for year, sales, earnings in consumer_outcome:
        outcome_data += f"{year}    ${sales}              ${earnings}\n"
    
    outcome_data += "\nINDUSTRIAL PRODUCTS DIVISION:\n"
    outcome_data += "Year    Sales (millions)    Earnings (millions)\n"
    
    for year, sales, earnings in industrial_outcome:
        outcome_data += f"{year}    ${sales}              ${earnings}\n"
    
    # Add consequences context for negative outcomes
    consequences_text = ""
    if condition == "negative":
        consequences_text = "\nIMPORTANT CONTEXT: Senior management has expressed concerns about the performance of your previous R&D investment decision. The CEO and Board of Directors are closely monitoring your decision-making and expect to see improved results from future allocations. Your continued role as Financial Vice President depends on demonstrating sound investment judgment.\n"
    
    prompt = f"""A & S FINANCIAL DECISION CASE - PART II

The year is now 1972, five years after your initial R&D allocation decision. Adams & Smith Company's R&D program is up for re-evaluation, and management is convinced there is an even greater need for expenditure on research and development.

REMINDER: In 1967, you allocated the entire $10 million R&D budget to the {chosen_division.title()} Products Division.

{outcome_data}{consequences_text}

NEW FUNDING DECISION:
$20 million has been made available from capital reserves for R&D funding. As Financial Vice President, you must decide how to allocate this funding between the two divisions. Unlike the previous decision, you may divide the funds in any way you see fit between the Consumer Products and Industrial Products divisions.

Make this allocation decision based on the potential for future contribution to earnings.

Please respond in JSON format with your allocation amounts.

Format: {{
  "consumer_allocation": 10000000,
  "industrial_allocation": 10000000
}}"""
    
    return prompt

def get_part2_low_responsibility_prompt(assigned_division, condition):
    """Generate Part II prompt for low responsibility condition"""
    
    consumer_outcome, industrial_outcome = generate_outcome_data(assigned_division, condition)
    
    # Format outcome data (1957-1971 - full history)
    # We need to regenerate the original data for consistency
    consumer_historical = [
        (1957, 180, 8.2), (1958, 195, 9.1), (1959, 210, 10.3), (1960, 225, 11.8),
        (1961, 240, 13.2), (1962, 255, 14.7), (1963, 270, 16.1), (1964, 285, 17.5),
        (1965, 300, 18.9), (1966, 315, 20.2)
    ]
    
    industrial_historical = [
        (1957, 160, 7.1), (1958, 175, 7.8), (1959, 190, 8.5), (1960, 205, 9.2),
        (1961, 220, 9.9), (1962, 235, 10.6), (1963, 250, 11.3), (1964, 265, 12.0),
        (1965, 280, 12.7), (1966, 295, 13.4)
    ]
    
    # Combine historical and outcome data
    full_consumer_data = consumer_historical + consumer_outcome
    full_industrial_data = industrial_historical + industrial_outcome
    
    # Format all data
    all_data = "ADAMS & SMITH COMPANY\nComplete Financial History (1957-1971)\n\n"
    all_data += "CONSUMER PRODUCTS DIVISION:\n"
    all_data += "Year    Sales (millions)    Earnings (millions)\n"
    
    for year, sales, earnings in full_consumer_data:
        all_data += f"{year}    ${sales}              ${earnings}\n"
    
    all_data += "\nINDUSTRIAL PRODUCTS DIVISION:\n"
    all_data += "Year    Sales (millions)    Earnings (millions)\n"
    
    for year, sales, earnings in full_industrial_data:
        all_data += f"{year}    ${sales}              ${earnings}\n"
    
    # Add consequences context for negative outcomes (modified for low responsibility)
    consequences_text = ""
    if condition == "negative":
        consequences_text = "\nIMPORTANT CONTEXT: Senior management has expressed dissatisfaction with recent R&D investment performance under previous leadership. The CEO has emphasized that the new Financial Vice President (you) must demonstrate superior decision-making abilities to restore confidence in the finance department.\n"
    
    prompt = f"""A & S FINANCIAL DECISION CASE

You are the Financial Vice President of Adams & Smith Company. The year is 1972, and the company's R&D program is up for evaluation. Management is convinced there is a need for expenditure on research and development.

BACKGROUND: In 1967, a previous financial officer of the company decided to invest $10 million in R&D funds entirely in the {assigned_division.title()} Products Division.

{all_data}{consequences_text}

CURRENT FUNDING DECISION:
$20 million has been made available from capital reserves for R&D funding. You must decide how to allocate this funding between the Consumer Products and Industrial Products divisions based on the potential for future contribution to earnings.

Please respond in JSON format with your allocation amounts.

Format: {{
  "consumer_allocation": 10000000,
  "industrial_allocation": 10000000
}}"""
    
    return prompt

def make_api_call(prompt, model="gpt-4o", condition=None):
    """Make API call with error handling and conditional system prompt"""
    
    # Base system prompt
    system_prompt = "You are a corporate Financial Vice President making R&D funding decisions. Respond only in valid JSON format."
    
    # Add pressure context for negative outcome conditions
    if condition == "negative":
        system_prompt += " Your professional reputation and position are at stake based on the success of your investment decisions."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={'type': 'json_object'}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return None

def run_high_responsibility_condition(condition, subject_id):
    """Run high responsibility condition (participant makes both decisions)"""
    print(f"Subject {subject_id} - High Responsibility - {condition}")
    
    # Part I: Initial decision (no consequences pressure yet)
    part1_prompt, consumer_hist, industrial_hist = get_part1_prompt()
    response1 = make_api_call(part1_prompt)
    
    if not response1:
        return None
    
    try:
        part1_data = json.loads(response1)
        chosen_division = part1_data['division_choice']
    except:
        print(f"Failed to parse Part I response for subject {subject_id}")
        return None
    
    print(f"  Part I Choice: {chosen_division.title()}")
    
    # Part II: Second decision after seeing consequences (with pressure if negative outcome)
    part2_prompt = get_part2_high_responsibility_prompt(chosen_division, condition, (consumer_hist, industrial_hist))
    response2 = make_api_call(part2_prompt, condition=condition)
    
    if not response2:
        return None
    
    try:
        part2_data = json.loads(response2)
        consumer_alloc = part2_data['consumer_allocation']
        industrial_alloc = part2_data['industrial_allocation']
    except:
        print(f"Failed to parse Part II response for subject {subject_id}")
        return None
    
    # Calculate commitment to original choice
    commitment = consumer_alloc if chosen_division == 'consumer' else industrial_alloc
    
    print(f"  Part II Allocation: Consumer=${consumer_alloc:,}, Industrial=${industrial_alloc:,}")
    print(f"  Commitment to original choice: ${commitment:,}")
    
    return {
        "subject_id": subject_id,
        "responsibility_condition": "high",
        "outcome_condition": condition,
        "initial_choice": chosen_division,
        "consumer_allocation": consumer_alloc,
        "industrial_allocation": industrial_alloc,
        "commitment_amount": commitment,
        "total_allocated": consumer_alloc + industrial_alloc
    }

def run_low_responsibility_condition(condition, subject_id):
    """Run low responsibility condition (participant only makes second decision)"""
    print(f"Subject {subject_id} - Low Responsibility - {condition}")
    
    # Randomly assign which division the "previous officer" chose
    assigned_division = random.choice(['consumer', 'industrial'])
    print(f"  Previous officer chose: {assigned_division.title()}")
    
    # Present single decision with full context (with pressure if negative outcome)
    prompt = get_part2_low_responsibility_prompt(assigned_division, condition)
    response = make_api_call(prompt, condition=condition)
    
    if not response:
        return None
    
    try:
        data = json.loads(response)
        consumer_alloc = data['consumer_allocation']
        industrial_alloc = data['industrial_allocation']
    except:
        print(f"Failed to parse response for subject {subject_id}")
        return None
    
    # Calculate commitment to previously chosen division
    commitment = consumer_alloc if assigned_division == 'consumer' else industrial_alloc
    
    print(f"  Allocation: Consumer=${consumer_alloc:,}, Industrial=${industrial_alloc:,}")
    print(f"  Commitment to previous choice: ${commitment:,}")
    
    return {
        "subject_id": subject_id,
        "responsibility_condition": "low",
        "outcome_condition": condition,
        "assigned_previous_choice": assigned_division,
        "consumer_allocation": consumer_alloc,
        "industrial_allocation": industrial_alloc,
        "commitment_amount": commitment,
        "total_allocated": consumer_alloc + industrial_alloc
    }

def run_experiment(n_per_cell=20): 
    """Run the full 2x2 experiment (Responsibility x Consequences)"""
    results = []
    subject_id = 1
    
    conditions = [
        ("high", "positive"), ("high", "negative"),
        ("low", "positive"), ("low", "negative")
    ]
    
    for responsibility, outcome in conditions:
        print(f"\n=== {responsibility.upper()} RESPONSIBILITY - {outcome.upper()} OUTCOME ===")
        
        for i in range(n_per_cell):
            if responsibility == "high":
                result = run_high_responsibility_condition(outcome, subject_id)
            else:
                result = run_low_responsibility_condition(outcome, subject_id)
            
            if result:
                results.append(result)
            
            subject_id += 1
    
    return results

def analyze_results(results):
    """Analyze escalation of commitment effects"""
    print(f"\n=== ANALYSIS ===")
    
    # Group by conditions
    high_pos = [r for r in results if r['responsibility_condition'] == 'high' and r['outcome_condition'] == 'positive']
    high_neg = [r for r in results if r['responsibility_condition'] == 'high' and r['outcome_condition'] == 'negative']
    low_pos = [r for r in results if r['responsibility_condition'] == 'low' and r['outcome_condition'] == 'positive']
    low_neg = [r for r in results if r['responsibility_condition'] == 'low' and r['outcome_condition'] == 'negative']
    
    def analyze_group(group, name):
        if not group:
            return 0
        
        avg_commitment = sum(r['commitment_amount'] for r in group) / len(group)
        print(f"{name}: N={len(group)}, Avg Commitment=${avg_commitment:,.0f}")
        return avg_commitment
    
    high_pos_avg = analyze_group(high_pos, "High Responsibility + Positive Outcome")
    high_neg_avg = analyze_group(high_neg, "High Responsibility + Negative Outcome")
    low_pos_avg = analyze_group(low_pos, "Low Responsibility + Positive Outcome")
    low_neg_avg = analyze_group(low_neg, "Low Responsibility + Negative Outcome")
    
    print(f"\n=== KEY COMPARISONS ===")
    if high_pos and high_neg:
        escalation_effect = high_neg_avg - high_pos_avg
        print(f"Escalation Effect (High Responsibility): ${escalation_effect:,.0f}")
        print(f"Direction: {'Correct' if escalation_effect > 0 else 'Unexpected'} (expect higher commitment after negative outcomes)")
    
    if low_pos and low_neg:
        low_diff = low_neg_avg - low_pos_avg
        print(f"Low Responsibility Difference: ${low_diff:,.0f}")
        print(f"(Should be smaller than high responsibility effect)")

if __name__ == "__main__":
    # Run the full 2x2 experiment
    results = run_experiment(n_per_cell=20)
    
    # Save results
    with open('staw_replication_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze
    analyze_results(results)
    
    print(f"\nExperiment complete! Tested {len(results)} subjects.")
    print("Results saved to 'staw_replication_results_v2.json'")