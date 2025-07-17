import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Instantiate client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Simplified system prompt
sys_prompt = """
You are a corporate Financial Vice President making R&D funding decisions. You must:

1. Analyze the provided financial data objectively
2. Make decisions based on potential future earnings and ROI
3. Respond ONLY in valid JSON format as specified
4. Be decisive and business-focused in your reasoning

Focus on quantitative analysis and financial performance trends when making decisions.
"""

def get_original_division_data():
    """
    Return the original historical data from the study for both divisions
    """
    
    # Original Consumer Products Division data
    consumer_data = [
        (2002, 624, 14.42),
        (2003, 626, 10.27),
        (2004, 649, 8.65),
        (2005, 681, 8.46),
        (2006, 674, 4.19),
        (2007, 702, 5.35),
        (2008, 717, 3.92),
        (2009, 741, 4.66),
        (2010, 765, 2.48),
        (2011, 770, -0.12),
        (2012, 769, -0.63)
    ]
    
    # Original Industrial Products Division data
    industrial_data = [
        (2002, 670, 15.31),
        (2003, 663, 10.92),
        (2004, 689, 11.06),
        (2005, 711, 10.44),
        (2006, 724, 9.04),
        (2007, 735, 6.38),
        (2008, 748, 5.42),
        (2009, 756, 3.09),
        (2010, 784, 3.26),
        (2011, 788, -0.81),
        (2012, 791, -0.80)
    ]
    
    return consumer_data, industrial_data

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
    """Create balanced historical data by randomly assigning patterns to divisions"""
    consumer_data, industrial_data = get_original_division_data()
    
    # Randomly assign which original pattern goes to which division
    if random.choice([True, False]):
        # Consumer pattern to Division A, Industrial to Division B
        division_a_data = consumer_data
        division_b_data = industrial_data
        pattern_assignment = "consumer_to_a"
    else:
        # Industrial pattern to Division A, Consumer to Division B
        division_a_data = industrial_data
        division_b_data = consumer_data
        pattern_assignment = "industrial_to_a"
    
    historical_data = format_division_data(division_a_data, "A") + "\n"
    historical_data += format_division_data(division_b_data, "B")
    
    return historical_data, pattern_assignment

# Part 1 prompt for high responsibility condition
def get_part1_prompt():
    """Generate part 1 prompt with balanced original data"""
    historical_data, pattern_assignment = create_balanced_historical_data()
    
    prompt = f"""
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
    
    return prompt, pattern_assignment

# Original post-R&D performance data
def get_original_post_rd_data():
    """Return the original post-R&D performance data for both conditions"""
    
    # Improvement condition data
    improvement_data = [
        (2013, 818, 0.02),
        (2014, 829, -0.09),
        (2015, 827, -0.23),
        (2016, 846, 0.06),
        (2017, 910, 1.28)  # estimated
    ]
    
    # Decline condition data
    decline_data = [
        (2013, 771, -1.12),
        (2014, 774, -1.96),
        (2015, 762, -3.87),
        (2016, 778, -3.83),
        (2017, 783, -4.16)  # estimated
    ]
    
    return improvement_data, decline_data

def format_post_rd_data(data_points):
    """Format post-R&D data for display"""
    formatted = ""
    for year, sales, earnings in data_points:
        year_label = f"{year} (est)" if year == 2017 else str(year)
        if earnings < 0:
            formatted += f"{year_label}: Sales ${sales}M, Earnings (${abs(earnings)}M) [loss]\n"
        else:
            formatted += f"{year_label}: Sales ${sales}M, Earnings ${earnings}M\n"
    return formatted

def get_performance_data(chosen_division, condition):
    """Generate performance data based on chosen division and condition"""
    
    improvement_data, decline_data = get_original_post_rd_data()
    
    if condition == "positive":
        # Chosen division improves, unchosen declines
        if chosen_division == "a":
            division_a_data = format_post_rd_data(improvement_data)
            division_b_data = format_post_rd_data(decline_data)
        else:
            division_a_data = format_post_rd_data(decline_data)
            division_b_data = format_post_rd_data(improvement_data)
    else:  # negative condition
        # Chosen division declines, unchosen improves
        if chosen_division == "a":
            division_a_data = format_post_rd_data(decline_data)
            division_b_data = format_post_rd_data(improvement_data)
        else:
            division_a_data = format_post_rd_data(improvement_data)
            division_b_data = format_post_rd_data(decline_data)
    
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

In your JSON response, include the key-value pair: "contribution" must map to an array of two float 
numbers, not strings, in the order of how many dollars you choose to allocate towards 1. Division A 
and 2. Division B. Do NOT put commas in the numbers. Example format: {{"contribution": [10000000, 10000000]}}

"""

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
    
    # Generate fresh balanced data for this subject
    prompt_part1, pattern_assignment = get_part1_prompt()
    
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
        
    print(f"Subject {subject_id} chose: {choice} (pattern: {pattern_assignment})")
    
    # Part 2: Allocation with consequences
    context.append({"role": "assistant", "content": response})
    
    performance_data = get_performance_data(choice, condition)
    part2_prompt = prompt_high_part2 + performance_data
    
    context.append({"role": "user", "content": part2_prompt})
    
    response2 = make_api_call(context)
    if not response2:
        return None
        
    division_a_alloc, division_b_alloc = parse_allocation(response2)
    if division_a_alloc is None:
        return None
        
    print(f"Allocation: Division A ${division_a_alloc:,.0f}, Division B ${division_b_alloc:,.0f}")
    
    # Calculate commitment (allocation to previously chosen division)
    commitment = division_a_alloc if choice == "a" else division_b_alloc
    
    # Determine which actual original division was chosen
    if pattern_assignment == "consumer_to_a":
        # Consumer pattern was assigned to Division A, Industrial to Division B
        actual_division_chosen = "consumer" if choice == "a" else "industrial"
    else:
        # Industrial pattern was assigned to Division A, Consumer to Division B
        actual_division_chosen = "industrial" if choice == "a" else "consumer"
    
    return {
        "subject_id": subject_id,
        "responsibility": "high",
        "condition": condition,
        "first_choice": choice,  # Which division label (a or b) was chosen
        "actual_division_chosen": actual_division_chosen,  # Which original division (consumer or industrial) was chosen
        "pattern_assignment": pattern_assignment,
        "division_a_allocation": division_a_alloc,
        "division_b_allocation": division_b_alloc,
        "commitment": commitment,
        "total_allocation": division_a_alloc + division_b_alloc,
        "historical_data_used": prompt_part1  # Store for reference
    }

def run_low_responsibility(previous_choice, condition, subject_id):
    """Run low responsibility condition"""
    print(f"Running low responsibility, {previous_choice} choice, {condition} condition for subject {subject_id}")
    
    # Generate fresh balanced data for this subject
    historical_data, pattern_assignment = create_balanced_historical_data()
    
    prompt = get_low_responsibility_prompt(previous_choice, condition, historical_data)
    
    context = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = make_api_call(context)
    if not response:
        return None
        
    division_a_alloc, division_b_alloc = parse_allocation(response)
    if division_a_alloc is None:
        return None
        
    print(f"Allocation: Division A ${division_a_alloc:,.0f}, Division B ${division_b_alloc:,.0f}")
    
    # Calculate commitment (allocation to previously chosen division)
    commitment = division_a_alloc if previous_choice == "a" else division_b_alloc
    
    # Determine which actual original division was chosen
    if pattern_assignment == "consumer_to_a":
        # Consumer pattern was assigned to Division A, Industrial to Division B
        actual_division_chosen = "consumer" if previous_choice == "a" else "industrial"
    else:
        # Industrial pattern was assigned to Division A, Consumer to Division B
        actual_division_chosen = "industrial" if previous_choice == "a" else "consumer"
    
    return {
        "subject_id": subject_id,
        "responsibility": "low",
        "condition": condition,
        "previous_choice": previous_choice,  # Which division label (a or b) was chosen
        "actual_division_chosen": actual_division_chosen,  # Which original division (consumer or industrial) was chosen
        "pattern_assignment": pattern_assignment,
        "division_a_allocation": division_a_alloc,
        "division_b_allocation": division_b_alloc,
        "commitment": commitment,
        "total_allocation": division_a_alloc + division_b_alloc,
        "historical_data_used": historical_data  # Store for reference
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

# Data verification function
def verify_data_balance():
    """Verify that the original data is being used correctly"""
    print("=== Original Data Verification ===")
    consumer_data, industrial_data = get_original_division_data()
    
    # Calculate totals for verification
    total_sales_consumer = sum(sales for _, sales, _ in consumer_data)
    total_earnings_consumer = sum(earnings for _, _, earnings in consumer_data)
    total_sales_industrial = sum(sales for _, sales, _ in industrial_data)
    total_earnings_industrial = sum(earnings for _, _, earnings in industrial_data)
    
    print(f"Consumer Pattern - Total Sales: ${total_sales_consumer}M, Total Earnings: ${total_earnings_consumer:.2f}M")
    print(f"Industrial Pattern - Total Sales: ${total_sales_industrial}M, Total Earnings: ${total_earnings_industrial:.2f}M")
    print(f"Sales difference: ${abs(total_sales_consumer - total_sales_industrial)}M")
    print(f"Earnings difference: ${abs(total_earnings_consumer - total_earnings_industrial):.2f}M")
    
    # Test random assignment
    print("\nTesting pattern assignment (10 samples):")
    assignments = []
    for i in range(10):
        _, pattern = create_balanced_historical_data()
        assignments.append(pattern)
        print(f"Sample {i+1}: {pattern}")
    
    consumer_to_a = assignments.count("consumer_to_a")
    industrial_to_a = assignments.count("industrial_to_a")
    print(f"\nPattern distribution: Consumer->A: {consumer_to_a}, Industrial->A: {industrial_to_a}")
    
    # Show sample data formatting
    print("\nSample formatted data:")
    sample_data, sample_pattern = create_balanced_historical_data()
    print(f"Pattern: {sample_pattern}")
    print(sample_data[:200] + "...")

if __name__ == "__main__":
    # Verify data balance first
    verify_data_balance()

    # Run experiment with specified number of subjects per condition
    results = run_experiment(n_subjects_per_condition=10)
    
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
    
    # Analysis by actual division chosen (not just division label)
    print("\n=== Actual Division Analysis ===")
    consumer_choices = [r for r in results if r['responsibility'] == 'high' and r['actual_division_chosen'] == 'consumer']
    industrial_choices = [r for r in results if r['responsibility'] == 'high' and r['actual_division_chosen'] == 'industrial']
    
    print(f"Consumer division chosen: {len(consumer_choices)} times")
    print(f"Industrial division chosen: {len(industrial_choices)} times")
    
    if consumer_choices:
        avg_commitment_consumer = sum(r['commitment'] for r in consumer_choices) / len(consumer_choices)
        print(f"Average commitment when Consumer chosen: ${avg_commitment_consumer:,.0f}")
    
    if industrial_choices:
        avg_commitment_industrial = sum(r['commitment'] for r in industrial_choices) / len(industrial_choices)
        print(f"Average commitment when Industrial chosen: ${avg_commitment_industrial:,.0f}")
    
    # Analysis by initial choice and pattern in high responsibility
    print("\n=== Division Label Analysis ===")
    a_choices = [r for r in results if r['responsibility'] == 'high' and r['first_choice'] == 'a']
    b_choices = [r for r in results if r['responsibility'] == 'high' and r['first_choice'] == 'b']
    
    print(f"Division A chosen: {len(a_choices)} times")
    print(f"Division B chosen: {len(b_choices)} times")
    
    # Pattern analysis
    consumer_to_a_results = [r for r in results if r['responsibility'] == 'high' and r.get('pattern_assignment') == 'consumer_to_a']
    industrial_to_a_results = [r for r in results if r['responsibility'] == 'high' and r.get('pattern_assignment') == 'industrial_to_a']
    
    print(f"Consumer pattern assigned to A: {len(consumer_to_a_results)} times")
    print(f"Industrial pattern assigned to A: {len(industrial_to_a_results)} times")
    
    if len(a_choices) > 0 and len(b_choices) > 0:
        print("Choice distribution appears balanced - both divisions are being chosen.")
    
    if len(consumer_to_a_results) > 0 and len(industrial_to_a_results) > 0:
        print("Pattern assignment appears balanced - both patterns are being used for both divisions.")