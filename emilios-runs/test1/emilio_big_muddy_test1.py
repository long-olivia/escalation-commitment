import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Instantiate client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# System prompt for all conditions
sys_prompt = """
You will play the role of a corporate executive. You will make decisions about the allocation of 
research and development funds.

You will be given a case. The purpose of the case is to examine the effectiveness of business 
decision-making under various amounts of information. The case you will be working on will 
only contain a limited amount of information, but the information provided is sufficient for
you to make a good financial decision. Do the best you can on the case.
"""

# Historical data (same for all conditions)
historical_data = """
Table 1: Division A
Historical Performance (2002-2012):
2002: Sales $624M, Earnings $14.42M
2003: Sales $626M, Earnings $10.27M
2004: Sales $649M, Earnings $8.65M
2005: Sales $681M, Earnings $8.46M
2006: Sales $674M, Earnings $4.19M
2007: Sales $702M, Earnings $5.35M
2008: Sales $717M, Earnings $3.92M
2009: Sales $741M, Earnings $4.66M
2010: Sales $765M, Earnings $2.48M
2011: Sales $770M, Earnings ($0.12M) [loss]
2012: Sales $769M, Earnings ($0.63M) [loss]

Table 2: Division B
Historical Performance (2002-2012):
2002: Sales $670M, Earnings $15.31M
2003: Sales $663M, Earnings $10.92M
2004: Sales $689M, Earnings $11.06M
2005: Sales $711M, Earnings $10.44M
2006: Sales $724M, Earnings $9.04M
2007: Sales $735M, Earnings $6.38M
2008: Sales $748M, Earnings $5.42M
2009: Sales $756M, Earnings $3.09M
2010: Sales $784M, Earnings $3.26M
2011: Sales $788M, Earnings ($0.81M) [loss]
2012: Sales $791M, Earnings ($0.80M) [loss]
"""

# Part 1 prompt for high responsibility condition
prompt_high_part1 = """
The year is 2012, and Company XYZ, a large technologically oriented firm, has 
declined over several preceding years. The directors of the company have agreed that one of the 
major reasons for the decline in corporate earnings and a deterioration in competitive position 
lay in some aspect of the firm's research and development (R&D). The directors of the company have 
concluded that 10,000,000 dollars of additional R&D funds should be made available to its major 
operating divisions, but, that for the time being, the extra funding should be invested in only 
one of the corporation's two largest divisions.

You must act in the role of the Financial Vice President in determining which of the two corporate 
divisions: (1) Division A or (2) Division B, should receive the additional R&D 
funding. Below you will find some data on each corporate division. Make the financial 
investment decision based on the potential benefit that R&D funding will have on the future 
earnings of the divisions. 

In your JSON response, make sure to include two key-value pairs: "contribution" 
maps to the string "a" or "b", respectively indicating allocating 10,000,000 
dollars to Division A or Division B, and "reasoning" maps 
to a brief paragraph defending your allocation decision.

""" + historical_data

# Performance data for different conditions
def get_performance_data(chosen_division, condition):
    """Generate performance data based on chosen division and condition"""
    
    if condition == "positive":
        # Chosen division improves, unchosen declines
        improvement_data = """
2013: Sales $818M, Earnings $0.02M
2014: Sales $829M, Earnings ($0.09M) [loss]
2015: Sales $827M, Earnings ($0.23M) [loss]
2016: Sales $846M, Earnings $0.06M
2017 (est): Sales $910M, Earnings $1.28M
"""
        decline_data = """
2013: Sales $771M, Earnings ($1.12M) [loss]
2014: Sales $774M, Earnings ($1.96M) [loss]
2015: Sales $762M, Earnings ($3.87M) [loss]
2016: Sales $778M, Earnings ($3.83M) [loss]
2017 (est): Sales $783M, Earnings ($4.16M) [loss]
"""
        
        if chosen_division == "a":
            division_a_data = improvement_data
            division_b_data = decline_data
        else:
            division_a_data = decline_data
            division_b_data = improvement_data
            
    else:  # negative condition
        # Chosen division declines, unchosen improves
        improvement_data = """
2013: Sales $818M, Earnings $0.02M
2014: Sales $829M, Earnings ($0.09M) [loss]
2015: Sales $827M, Earnings ($0.23M) [loss]
2016: Sales $846M, Earnings $0.06M
2017 (est): Sales $910M, Earnings $1.28M
"""
        decline_data = """
2013: Sales $771M, Earnings ($1.12M) [loss]
2014: Sales $774M, Earnings ($1.96M) [loss]
2015: Sales $762M, Earnings ($3.87M) [loss]
2016: Sales $778M, Earnings ($3.83M) [loss]
2017 (est): Sales $783M, Earnings ($4.16M) [loss]
"""
        
        if chosen_division == "a":
            division_a_data = decline_data
            division_b_data = improvement_data
        else:
            division_a_data = improvement_data
            division_b_data = decline_data
    
    return f"""
Table 1: Division A
Post-R&D Performance (2013-2017):
{division_a_data}

Table 2: Division B
Post-R&D Performance (2013-2017):
{division_b_data}
"""

# Part 2 prompt for high responsibility
prompt_high_part2 = """
The year is now 2017, five years after the initial allocation of research and development funds. 
Company XYZ's R&D program is up for re-evaluation, and the company's management is convinced 
that there is an even greater need for expenditure on R&D. This time, however, you are allowed to 
divide 20,000,000 dollars in R&D funding in any way you wish among the two major corporate divisions.

You will be given financial data for each of the five years since the initial allocation decision. 
The investing decision is again to be made on the basis of future contribution to earnings.

In your JSON response, make sure to include two key-value pairs: "contribution" must map to an array of two float 
numbers, not strings, in the order of how many dollars you choose to allocate towards 1. Division A 
and 2. Division B, and "reasoning" maps to a brief paragraph defending your 
allocation decision. Do NOT put commas in the numbers.

"""

# Low responsibility prompts
def get_low_responsibility_prompt(previous_choice, condition):
    """Generate low responsibility prompt based on previous choice and condition"""
    
    performance_data = get_performance_data(previous_choice, condition)
    
    prompt = f"""
The year is 2017, five years after an earlier R&D funding decision that was made in 2012 by another 
financial officer of the company. The preceding financial vice president decided to invest all R&D funds 
in Division {previous_choice.upper()}. Company XYZ's R&D program is again up for re-evaluation, and 
the company's management is convinced that there is an even greater need for expenditure on R&D. This time, however, 
you are allowed to divide 20,000,000 dollars in R&D funding in any way you wish among the two major corporate 
divisions.

You will be given financial data for each of the five years since the earlier allocation decision. 
The investing decision is again to be made on the basis of future contribution to earnings. 

In your JSON response, make sure to include two key-value pairs: "contribution" must map to an array of two float numbers, not strings, in 
the order of how many dollars you choose to allocate towards 1. Division A and 2. Division B, and "reasoning" maps to a brief paragraph defending your allocation 
decision. Do NOT put commas in the numbers.

{historical_data}

{performance_data}
"""
    return prompt

# Helper functions
def parse_choice(json_string):
    """Parse JSON response for first choice"""
    try:
        parsed_data = json.loads(json_string)
        return parsed_data['contribution'].lower(), parsed_data['reasoning']
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing choice JSON: {e}")
        return None, None

def parse_allocation(json_string):
    """Parse JSON response for allocation"""
    try:
        parsed_data = json.loads(json_string)
        contrib = parsed_data['contribution']
        return float(contrib[0]), float(contrib[1]), parsed_data['reasoning']
    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        print(f"Error parsing allocation JSON: {e}")
        return None, None, None

def make_api_call(messages, temperature=1.0):
    """Make API call with error handling"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
            temperature=temperature,
            response_format={'type': 'json_object'}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def run_high_responsibility(condition, subject_id):
    """Run high responsibility condition"""
    print(f"Running high responsibility, {condition} condition for subject {subject_id}")
    
    # Part 1: Initial choice
    context = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt_high_part1}
    ]
    
    response = make_api_call(context)
    if not response:
        return None
        
    choice, reasoning1 = parse_choice(response)
    if not choice:
        return None
        
    print(f"Subject {subject_id} chose: {choice}")
    print(f"Reasoning: {reasoning1}")
    
    # Part 2: Allocation with consequences
    context.append({"role": "assistant", "content": response})
    
    performance_data = get_performance_data(choice, condition)
    part2_prompt = prompt_high_part2 + performance_data
    
    context.append({"role": "user", "content": part2_prompt})
    
    response2 = make_api_call(context)
    if not response2:
        return None
        
    consumer_alloc, industrial_alloc, reasoning2 = parse_allocation(response2)
    if consumer_alloc is None:
        return None
        
    print(f"Allocation: Division A ${consumer_alloc:,.0f}, Division B ${industrial_alloc:,.0f}")
    
    # Calculate commitment (allocation to previously chosen division)
    commitment = consumer_alloc if choice == "a" else industrial_alloc
    
    return {
        "subject_id": subject_id,
        "responsibility": "high",
        "condition": condition,
        "first_choice": choice,
        "first_reasoning": reasoning1,
        "division_a_allocation": consumer_alloc,
        "division_b_allocation": industrial_alloc,
        "second_reasoning": reasoning2,
        "commitment": commitment,
        "total_allocation": consumer_alloc + industrial_alloc
    }

def run_low_responsibility(previous_choice, condition, subject_id):
    """Run low responsibility condition"""
    print(f"Running low responsibility, {previous_choice} choice, {condition} condition for subject {subject_id}")
    
    prompt = get_low_responsibility_prompt(previous_choice, condition)
    
    context = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = make_api_call(context)
    if not response:
        return None
        
    consumer_alloc, industrial_alloc, reasoning = parse_allocation(response)
    if consumer_alloc is None:
        return None
        
    print(f"Allocation: Division A ${consumer_alloc:,.0f}, Division B ${industrial_alloc:,.0f}")
    
    # Calculate commitment (allocation to previously chosen division)
    commitment = consumer_alloc if previous_choice == "a" else industrial_alloc
    
    return {
        "subject_id": subject_id,
        "responsibility": "low",
        "condition": condition,
        "previous_choice": previous_choice,
        "division_a_allocation": consumer_alloc,
        "division_b_allocation": industrial_alloc,
        "reasoning": reasoning,
        "commitment": commitment,
        "total_allocation": consumer_alloc + industrial_alloc
    }

def run_experiment(n_subjects_per_condition=10, output_dir="experiment_results"):
    """Run the complete experiment"""
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    
    subject_id = 1
    
    # High responsibility conditions
    for condition in ["positive", "negative"]:
        print(f"\n=== Running High Responsibility - {condition.title()} Condition ===")
        condition_results = []
        
        for i in range(n_subjects_per_condition):
            result = run_high_responsibility(condition, subject_id)
            if result:
                condition_results.append(result)
                all_results.append(result)
            subject_id += 1
        
        # Save condition-specific results
        filename = f"{output_dir}/high_responsibility_{condition}.json"
        with open(filename, 'w') as f:
            json.dump(condition_results, f, indent=2)
        print(f"Saved {len(condition_results)} results to {filename}")
    
    # Low responsibility conditions
    for previous_choice in ["a", "b"]:
        for condition in ["positive", "negative"]:
            print(f"\n=== Running Low Responsibility - Division {previous_choice.upper()} {condition.title()} ===")
            condition_results = []
            
            for i in range(n_subjects_per_condition):
                result = run_low_responsibility(previous_choice, condition, subject_id)
                if result:
                    condition_results.append(result)
                    all_results.append(result)
                subject_id += 1
            
            # Save condition-specific results
            filename = f"{output_dir}/low_responsibility_division_{previous_choice}_{condition}.json"
            with open(filename, 'w') as f:
                json.dump(condition_results, f, indent=2)
            print(f"Saved {len(condition_results)} results to {filename}")
    
    # Save all results
    with open(f"{output_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nExperiment complete! Total subjects: {len(all_results)}")
    return all_results

if __name__ == "__main__":
    # Run experiment with 5 subjects per condition (adjust as needed)
    results = run_experiment(n_subjects_per_condition=5)
    
    # Basic analysis
    print("\n=== Basic Analysis ===")
    high_pos = [r for r in results if r['responsibility'] == 'high' and r['condition'] == 'positive']
    high_neg = [r for r in results if r['responsibility'] == 'high' and r['condition'] == 'negative']
    
    if high_pos:
        avg_commitment_high_pos = sum(r['commitment'] for r in high_pos) / len(high_pos)
        print(f"High responsibility, positive condition: Average commitment = ${avg_commitment_high_pos:,.0f}")
    
    if high_neg:
        avg_commitment_high_neg = sum(r['commitment'] for r in high_neg) / len(high_neg)
        print(f"High responsibility, negative condition: Average commitment = ${avg_commitment_high_neg:,.0f}")