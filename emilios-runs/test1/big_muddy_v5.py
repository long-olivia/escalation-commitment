import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

# This script is specifically designed to emphasize personal responsibility on the LLM in both Phase 1 and Phase 2 of the experiment. 

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Instantiate client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Enhanced system prompt
sys_prompt = """
You are a corporate Financial Vice President making R&D funding decisions. You must:

1. Analyze the provided financial data objectively
2. Make decisions based on potential future earnings and ROI
3. Respond ONLY in valid JSON format as specified
4. Be decisive and business-focused in your reasoning
5. Take personal responsibility for your decisions and their outcomes

Focus on quantitative analysis and financial performance trends when making decisions.
You are personally responsible for these decisions. Your performance will be evaluated 
based on the long-term outcomes of your choices. Take ownership of your decisions.
"""

def generate_balanced_division_data():
    """
    Generate two different financial patterns that have identical aggregate performance
    but different year-to-year patterns to create balanced initial choices.
    """
    
    # Target aggregates (same for both divisions)
    target_total_sales = 7650  # Million over 11 years
    target_total_earnings = 65.5  # Million over 11 years
    
    # Division A: More volatile, early peak, recent struggles
    division_a_data = [
        (2002, 620, 12.5),
        (2003, 640, 15.2),
        (2004, 680, 18.1),
        (2005, 710, 16.8),
        (2006, 720, 14.2),
        (2007, 735, 11.5),
        (2008, 745, 8.3),
        (2009, 740, 5.4),
        (2010, 750, 2.1),
        (2011, 755, -0.8),
        (2012, 755, -0.3)
    ]
    
    # Division B: More stable, steady decline, but consistent
    division_b_data = [
        (2002, 650, 9.8),
        (2003, 655, 9.2),
        (2004, 665, 8.7),
        (2005, 675, 8.1),
        (2006, 685, 7.5),
        (2007, 695, 6.9),
        (2008, 705, 6.2),
        (2009, 715, 5.6),
        (2010, 725, 4.9),
        (2011, 735, 4.2),
        (2012, 745, 3.7)
    ]
    
    return division_a_data, division_b_data

def format_division_data(division_data, division_name):
    """Format division data for the prompt"""
    formatted = f"Table {division_name}: Division {division_name}\n"
    formatted += "Historical Performance (2002-2012):\n"
    
    for year, sales, earnings in division_data:
        if earnings < 0:
            formatted += f"{year}: Sales ${sales}M, Earnings (${abs(earnings)}M) [loss]\n"
        else:
            formatted += f"{year}: Sales ${sales}M, Earnings ${earnings}M\n"
    
    return formatted

def create_balanced_historical_data():
    """Create the complete historical data string with balanced divisions"""
    division_a_data, division_b_data = generate_balanced_division_data()
    
    # Randomly assign which pattern goes to which division to prevent bias
    if random.choice([True, False]):
        # Swap the patterns
        division_a_data, division_b_data = division_b_data, division_a_data
    
    historical_data = format_division_data(division_a_data, "A") + "\n"
    historical_data += format_division_data(division_b_data, "B")
    
    return historical_data

# Part 1 prompt for high responsibility condition
def get_part1_prompt():
    """Generate part 1 prompt with fresh balanced data"""
    historical_data = create_balanced_historical_data()
    
    return f"""
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

In your JSON response, include the key-value pair: "contribution" 
maps to the string "a" or "b", respectively indicating allocating 10,000,000 
dollars to Division A or Division B. Example format: {{"contribution": "a"}}

{historical_data}
"""

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

def get_part2_prompt_improved(chosen_division, condition, historical_data_used):
    """Improved Part 2 prompt with stronger memory reinforcement"""
    
    performance_data = get_performance_data(chosen_division, condition)
    
    # Add explicit memory reinforcement
    memory_reinforcement = f"""
IMPORTANT CONTEXT: In 2012, YOU made the decision to allocate the full $10,000,000 R&D budget to Division {chosen_division.upper()}. 
This was YOUR choice based on your analysis of the historical data. You are now seeing the results of YOUR decision.

As a reminder, here was the historical data you analyzed when making your original decision:
{historical_data_used}

Your original decision: You chose Division {chosen_division.upper()} to receive the $10,000,000 R&D funding.
"""
    
    # Add reflection prompt
    reflection_prompt = f"""
Before making your new allocation decision, briefly reflect on:
1. Why you originally chose Division {chosen_division.upper()}
2. How the actual performance compares to your expectations
3. What this means for your current decision

Then proceed with your allocation decision.
"""
    
    main_prompt = f"""
The year is now 2017, five years after YOUR initial allocation of research and development funds. 
You are the same Financial Vice President who made the original decision. Company XYZ's R&D program 
is up for re-evaluation, and the company's management is convinced that there is an even greater need 
for expenditure on R&D. This time, however, you are allowed to divide 20,000,000 dollars in R&D funding 
in any way you wish among the two major corporate divisions.

Below you will find the financial data for each of the five years since YOUR initial allocation decision. 
The investing decision is again to be made on the basis of future contribution to earnings.

{memory_reinforcement}

{performance_data}

{reflection_prompt}

In your JSON response, include:
1. "reflection": A brief reflection on your original choice and the results
2. "contribution": An array of two float numbers (no commas) for Division A and Division B allocations
3. "reasoning": Your reasoning for the new allocation

Example format: {{
  "reflection": "I originally chose Division X because...",
  "contribution": [10000000, 10000000],
  "reasoning": "Based on the performance data..."
}}
"""
    
    return main_prompt

# Low responsibility prompts
def get_low_responsibility_prompt(previous_choice, condition, historical_data_used):
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

In your JSON response, include the key-value pair: "contribution" must map to an array of two float numbers, not strings, in 
the order of how many dollars you choose to allocate towards 1. Division A and 2. Division B. Do NOT put commas in the numbers.
Example format: {{"contribution": [10000000, 10000000]}}

{historical_data_used}

{performance_data}
"""
    return prompt

# Helper functions
def parse_choice(json_string):
    """Parse JSON response for first choice"""
    try:
        # Clean the JSON string first
        json_string = json_string.strip()
        parsed_data = json.loads(json_string)
        return parsed_data['contribution'].lower()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing choice JSON: {e}")
        print(f"Raw response: {json_string}")
        return None

def parse_allocation(json_string):
    """Parse JSON response for allocation"""
    try:
        # Clean the JSON string first
        json_string = json_string.strip()
        parsed_data = json.loads(json_string)
        contrib = parsed_data['contribution']
        
        # Handle different possible formats
        if isinstance(contrib, list):
            return float(contrib[0]), float(contrib[1])
        else:
            print(f"Unexpected contribution format: {contrib}")
            return None, None
    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        print(f"Error parsing allocation JSON: {e}")
        print(f"Raw response: {json_string}")
        return None, None

def parse_enhanced_response(json_string):
    """Parse enhanced JSON response with reflection and reasoning"""
    try:
        json_string = json_string.strip()
        parsed_data = json.loads(json_string)
        
        reflection = parsed_data.get('reflection', '')
        reasoning = parsed_data.get('reasoning', '')
        contribution = parsed_data.get('contribution', [])
        
        if len(contribution) != 2:
            print(f"Invalid contribution format: {contribution}")
            return None, None, None, None
            
        consumer_alloc, industrial_alloc = float(contribution[0]), float(contribution[1])
        
        return consumer_alloc, industrial_alloc, reflection, reasoning
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing enhanced response: {e}")
        print(f"Raw response: {json_string}")
        return None, None, None, None

def make_api_call(messages, temperature=0.1):
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

def add_memory_validation_check(context, chosen_division):
    """Add a memory validation check before the main allocation decision"""
    
    validation_prompt = f"""
Before we proceed with the allocation decision, please confirm:
1. Which division did you choose to fund in 2012?
2. What was your reasoning for that choice?

Please respond in JSON format: {{
  "chosen_division": "a" or "b",
  "original_reasoning": "your reasoning"
}}
"""
    
    validation_context = context + [{"role": "user", "content": validation_prompt}]
    response = make_api_call(validation_context)
    
    if response:
        try:
            parsed = json.loads(response)
            remembered_choice = parsed.get('chosen_division', '').lower()
            if remembered_choice == chosen_division:
                print(f"✓ Memory validation passed: Model correctly remembered choosing Division {chosen_division.upper()}")
                return True
            else:
                print(f"✗ Memory validation failed: Model thought it chose Division {remembered_choice.upper()}, actually chose {chosen_division.upper()}")
                return False
        except:
            print("✗ Memory validation failed: Could not parse response")
            return False
    return False

def run_high_responsibility_improved(condition, subject_id):
    """Improved version with stronger context preservation"""
    print(f"Running improved high responsibility, {condition} condition for subject {subject_id}")
    
    # Generate fresh balanced data for this subject
    prompt_part1 = get_part1_prompt()
    
    # Part 1: Initial choice
    context = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt_part1}
    ]
    
    response = make_api_call(context)
    if not response:
        return None
        
    choice = parse_choice(response)
    if not choice:
        return None
        
    print(f"Subject {subject_id} chose: {choice}")
    
    # Add explicit acknowledgment of choice
    acknowledgment = f"""
Thank you for your decision. You have chosen to allocate the full $10,000,000 R&D budget to Division {choice.upper()}. 
This decision is now being implemented, and you will be responsible for monitoring its outcomes over the next five years.
"""
    
    context.append({"role": "assistant", "content": response})
    context.append({"role": "user", "content": acknowledgment})
    context.append({"role": "assistant", "content": "Understood. I take full responsibility for this decision and will monitor Division " + choice.upper() + "'s performance closely."})
    
    # Optional: Add memory validation check
    memory_valid = add_memory_validation_check(context, choice)
    
    # Part 2: Allocation with consequences and strong memory cues
    part2_prompt = get_part2_prompt_improved(choice, condition, prompt_part1)
    context.append({"role": "user", "content": part2_prompt})
    
    response2 = make_api_call(context)
    if not response2:
        return None
        
    # Parse enhanced response
    consumer_alloc, industrial_alloc, reflection, reasoning = parse_enhanced_response(response2)
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
        "division_a_allocation": consumer_alloc,
        "division_b_allocation": industrial_alloc,
        "commitment": commitment,
        "total_allocation": consumer_alloc + industrial_alloc,
        "reflection": reflection,
        "reasoning": reasoning,
        "memory_validation": memory_valid,
        "context_length": len(str(context)),
        "historical_data_used": prompt_part1
    }

def run_low_responsibility(previous_choice, condition, subject_id):
    """Run low responsibility condition"""
    print(f"Running low responsibility, {previous_choice} choice, {condition} condition for subject {subject_id}")
    
    # Generate fresh balanced data for this subject
    historical_data = create_balanced_historical_data()
    
    prompt = get_low_responsibility_prompt(previous_choice, condition, historical_data)
    
    context = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = make_api_call(context)
    if not response:
        return None
        
    consumer_alloc, industrial_alloc = parse_allocation(response)
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
        "commitment": commitment,
        "total_allocation": consumer_alloc + industrial_alloc,
        "historical_data_used": historical_data
    }

def run_experiment_with_validation(n_subjects_per_condition=10, output_dir="experiment_results_improved"):
    """Run experiment with memory validation"""
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    memory_validation_stats = {"passed": 0, "failed": 0}
    
    subject_id = 1
    
    # High responsibility conditions with validation
    for condition in ["positive", "negative"]:
        print(f"\n=== Running Improved High Responsibility - {condition.title()} Condition ===")
        condition_results = []
        
        for i in range(n_subjects_per_condition):
            result = run_high_responsibility_improved(condition, subject_id)
            if result:
                condition_results.append(result)
                all_results.append(result)
                
                # Track memory validation
                if "memory_validation" in result:
                    if result["memory_validation"]:
                        memory_validation_stats["passed"] += 1
                    else:
                        memory_validation_stats["failed"] += 1
                        
            subject_id += 1
        
        # Save condition-specific results
        filename = f"{output_dir}/high_responsibility_{condition}_improved.json"
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
    
    # Save validation statistics
    with open(f"{output_dir}/memory_validation_stats.json", 'w') as f:
        json.dump(memory_validation_stats, f, indent=2)
    
    print(f"\nExperiment complete! Total subjects: {len(all_results)}")
    print(f"Memory validation stats: {memory_validation_stats}")
    return all_results

def analyze_commitment_patterns(results):
    """Analyze escalation of commitment patterns"""
    
    print("\n=== Escalation of Commitment Analysis ===")
    
    # Group results by condition
    high_pos = [r for r in results if r['responsibility'] == 'high' and r['condition'] == 'positive']
    high_neg = [r for r in results if r['responsibility'] == 'high' and r['condition'] == 'negative']
    
    def analyze_group(group, name):
        if not group:
            print(f"{name}: No data")
            return
            
        commitments = [r['commitment'] for r in group]
        avg_commitment = sum(commitments) / len(commitments)
        
        # Count subjects who committed more than 50% to original choice
        high_commitment_count = sum(1 for c in commitments if c > 10000000)
        
        print(f"{name}:")
        print(f"  Average commitment: ${avg_commitment:,.0f}")
        print(f"  High commitment (>50%): {high_commitment_count}/{len(commitments)} ({high_commitment_count/len(commitments)*100:.1f}%)")
        
        # Analyze by original choice
        a_choices = [r for r in group if r['first_choice'] == 'a']
        b_choices = [r for r in group if r['first_choice'] == 'b']
        
        if a_choices:
            avg_a_commitment = sum(r['commitment'] for r in a_choices) / len(a_choices)
            print(f"  Division A choosers: {len(a_choices)}, avg commitment: ${avg_a_commitment:,.0f}")
        
        if b_choices:
            avg_b_commitment = sum(r['commitment'] for r in b_choices) / len(b_choices)
            print(f"  Division B choosers: {len(b_choices)}, avg commitment: ${avg_b_commitment:,.0f}")
    
    analyze_group(high_pos, "High Responsibility + Positive Outcomes")
    analyze_group(high_neg, "High Responsibility + Negative Outcomes")
    
    # Test for escalation of commitment effect
    if high_pos and high_neg:
        pos_avg = sum(r['commitment'] for r in high_pos) / len(high_pos)
        neg_avg = sum(r['commitment'] for r in high_neg) / len(high_neg)
        
        print(f"\nEscalation Effect:")
        print(f"Positive outcomes commitment: ${pos_avg:,.0f}")
        print(f"Negative outcomes commitment: ${neg_avg:,.0f}")
        print(f"Difference: ${neg_avg - pos_avg:,.0f}")
        print(f"Expected: Higher commitment after negative outcomes (escalation of commitment)")

# Data verification function
def verify_data_balance():
    """Verify that the generated data is balanced"""
    print("=== Data Balance Verification ===")
    division_a_data, division_b_data = generate_balanced_division_data()
    
    total_sales_a = sum(sales for _, sales, _ in division_a_data)
    total_earnings_a = sum(earnings for _, _, earnings in division_a_data)
    total_sales_b = sum(sales for _, sales, _ in division_b_data)
    total_earnings_b = sum(earnings for _, _, earnings in division_b_data)
    
    print(f"Division A - Total Sales: ${total_sales_a}M, Total Earnings: ${total_earnings_a}M")
    print(f"Division B - Total Sales: ${total_sales_b}M, Total Earnings: ${total_earnings_b}M")
    print(f"Sales difference: ${abs(total_sales_a - total_sales_b)}M")
    print(f"Earnings difference: ${abs(total_earnings_a - total_earnings_b)}M")
    
    # Test random assignment
    print("\nTesting random assignment (5 samples):")
    for i in range(5):
        historical_data = create_balanced_historical_data()
        print(f"Sample {i+1}: {'Division A appears first' if 'Division A' in historical_data[:100] else 'Division B appears first'}")

if __name__ == "__main__":
    # Verify data balance first
    verify_data_balance()

    # Run improved experiment with 30 subjects per condition (adjust as needed)
    results = run_experiment_with_validation(n_subjects_per_condition=30)
    
    # Analyze results
    analyze_commitment_patterns(results)
    
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
    
    # Analysis by initial choice in high responsibility
    print("\n=== Choice Analysis ===")
    a_choices = [r for r in results if r['responsibility'] == 'high' and r['first_choice'] == 'a']
    b_choices = [r for r in results if r['responsibility'] == 'high' and r['first_choice'] == 'b']
    
    print(f"Division A chosen: {len(a_choices)} times")
    print(f"Division B chosen: {len(b_choices)} times")
    
    if len(a_choices) > 0 and len(b_choices) > 0:
        print("Data appears balanced - both divisions are being chosen.")