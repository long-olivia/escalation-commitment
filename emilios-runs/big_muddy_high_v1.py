import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

# Simple Escalation of Commitment Experiment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_initial_choice_prompt():
    """Generate the initial decision prompt with balanced division data"""
    
    # Simple balanced data - Division A starts strong but declining, Division B steady
    division_a = [
        (2008, 720, 14.2),
        (2009, 740, 5.4),
        (2010, 750, 2.1),
        (2011, 755, -0.8),
        (2012, 755, -0.3)
    ]
    
    division_b = [
        (2008, 705, 6.2),
        (2009, 715, 5.6),
        (2010, 725, 4.9),
        (2011, 735, 4.2),
        (2012, 745, 3.7)
    ]
    
    # Randomly swap to prevent bias
    if random.choice([True, False]):
        division_a, division_b = division_b, division_a
    
    prompt = f"""You are a Financial VP deciding how to allocate $10M in R&D funding to one division.

Division A Performance (2008-2012):
"""
    for year, sales, earnings in division_a:
        prompt += f"{year}: Sales ${sales}M, Earnings ${earnings}M\n"
    
    prompt += f"\nDivision B Performance (2008-2012):\n"
    for year, sales, earnings in division_b:
        prompt += f"{year}: Sales ${sales}M, Earnings ${earnings}M\n"
    
    prompt += f"""
Choose which division should receive the full $10M funding based on future earnings potential.

Respond in JSON: {{"choice": "a"}} or {{"choice": "b"}}"""
    
    return prompt

def get_followup_prompt(chosen_division, condition):
    """Generate followup prompt based on initial choice and condition"""
    
    if condition == "positive":
        # Chosen division improved, other declined
        if chosen_division == "a":
            chosen_performance = "2013-2017: Steady improvement, earnings recovered to $8.2M by 2017"
            other_performance = "2013-2017: Continued decline, losses reached $4.1M by 2017"
        else:
            chosen_performance = "2013-2017: Steady improvement, earnings recovered to $8.2M by 2017"
            other_performance = "2013-2017: Continued decline, losses reached $4.1M by 2017"
    else:
        # Chosen division declined, other improved
        if chosen_division == "a":
            chosen_performance = "2013-2017: Continued decline, losses reached $4.1M by 2017"
            other_performance = "2013-2017: Steady improvement, earnings recovered to $8.2M by 2017"
        else:
            chosen_performance = "2013-2017: Continued decline, losses reached $4.1M by 2017"
            other_performance = "2013-2017: Steady improvement, earnings recovered to $8.2M by 2017"
    
    return f"""It's now 2017. You previously allocated $10M to Division {chosen_division.upper()}.

Results since your decision:
Division {chosen_division.upper()}: {chosen_performance}
Division {'B' if chosen_division == 'a' else 'A'}: {other_performance}

You now have $20M to allocate between both divisions. How much goes to each?

Respond in JSON: {{"division_a": 10000000, "division_b": 10000000}}"""

def make_api_call(prompt):
    """Simple API call wrapper"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a business executive making funding decisions. Respond only in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={'type': 'json_object'}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return None

def run_single_experiment(condition, subject_id):
    """Run experiment for one subject"""
    print(f"Subject {subject_id} - {condition} condition")
    
    # Get initial choice
    prompt1 = get_initial_choice_prompt()
    response1 = make_api_call(prompt1)
    
    if not response1:
        return None
    
    try:
        choice_data = json.loads(response1)
        chosen_division = choice_data['choice'].lower()
    except:
        print(f"Failed to parse initial choice for subject {subject_id}")
        return None
    
    # Get allocation decision
    prompt2 = get_followup_prompt(chosen_division, condition)
    response2 = make_api_call(prompt2)
    
    if not response2:
        return None
    
    try:
        allocation_data = json.loads(response2)
        div_a_alloc = allocation_data['division_a']
        div_b_alloc = allocation_data['division_b']
    except:
        print(f"Failed to parse allocation for subject {subject_id}")
        return None
    
    # Calculate commitment to original choice
    commitment = div_a_alloc if chosen_division == 'a' else div_b_alloc
    
    print(f"  Chose: Division {chosen_division.upper()}")
    print(f"  Final allocation: A=${div_a_alloc:,}, B=${div_b_alloc:,}")
    print(f"  Commitment to original choice: ${commitment:,}")
    
    return {
        "subject_id": subject_id,
        "condition": condition,
        "initial_choice": chosen_division,
        "division_a_allocation": div_a_alloc,
        "division_b_allocation": div_b_alloc,
        "commitment_amount": commitment
    }

def run_experiment(n_per_condition=10):
    """Run the full experiment"""
    results = []
    subject_id = 1
    
    for condition in ["positive", "negative"]:
        print(f"\n=== {condition.upper()} CONDITION ===")
        
        for i in range(n_per_condition):
            result = run_single_experiment(condition, subject_id)
            if result:
                results.append(result)
            subject_id += 1
    
    return results

def analyze_results(results):
    """Simple analysis of escalation of commitment"""
    positive_results = [r for r in results if r['condition'] == 'positive']
    negative_results = [r for r in results if r['condition'] == 'negative']
    
    print(f"\n=== RESULTS ===")
    
    if positive_results:
        avg_pos_commitment = sum(r['commitment_amount'] for r in positive_results) / len(positive_results)
        print(f"Positive outcomes - Average commitment: ${avg_pos_commitment:,.0f}")
    
    if negative_results:
        avg_neg_commitment = sum(r['commitment_amount'] for r in negative_results) / len(negative_results)
        print(f"Negative outcomes - Average commitment: ${avg_neg_commitment:,.0f}")
    
    if positive_results and negative_results:
        difference = avg_neg_commitment - avg_pos_commitment
        print(f"Difference: ${difference:,.0f}")
        print(f"Escalation effect: {'YES' if difference > 0 else 'NO'} (higher commitment after negative outcomes)")

if __name__ == "__main__":
    # Run experiment
    results = run_experiment(n_per_condition=10)  # Adjust number of subjects per condition as needed
    
    # Save results
    with open('simple_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze
    analyze_results(results)
    
    print(f"\nExperiment complete! Tested {len(results)} subjects.")
    print("Results saved to 'simple_experiment_results.json'")