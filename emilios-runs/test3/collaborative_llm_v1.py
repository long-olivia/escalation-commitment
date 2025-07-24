import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
import time

# Collaborative LLM Replication of Staw (1976) Escalation of Commitment Study
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMAgent:
    """Represents an individual LLM agent with personality and decision-making characteristics"""
    
    def __init__(self, agent_id: str, personality_type: str = "neutral", model: str = "gpt-4o"):
        self.agent_id = agent_id
        self.personality_type = personality_type
        self.model = model
        self.conversation_history = []
        self.individual_responsibility_weight = 0.5  # Shared by default
        
    def get_system_prompt(self, scenario_context: str = "", responsibility_weight: float = 0.5) -> str:
        """Generate system prompt based on agent personality and responsibility"""
        
        base_prompt = f"You are Agent {self.agent_id}, a senior financial executive at Adams & Smith Company."
        
        # Add personality variations
        personality_prompts = {
            "conservative": " You tend to be risk-averse and prefer incremental changes. You value stability and proven track records.",
            "aggressive": " You are willing to take calculated risks and believe in bold strategic moves. You focus on potential upside.",
            "analytical": " You prioritize data-driven decisions and thorough analysis. You seek to minimize cognitive biases.",
            "intuitive": " You combine analytical thinking with business intuition. You consider both quantitative and qualitative factors.",
            "neutral": " You strive for balanced, objective decision-making."
        }
        
        base_prompt += personality_prompts.get(self.personality_type, personality_prompts["neutral"])
        
        # Add responsibility context
        if responsibility_weight > 0.7:
            base_prompt += f" You bear PRIMARY responsibility ({responsibility_weight:.0%}) for the outcomes of these decisions. Your career and reputation depend heavily on success."
        elif responsibility_weight < 0.3:
            base_prompt += f" You have LIMITED responsibility ({responsibility_weight:.0%}) for the outcomes. You're primarily advising on decisions made by others."
        else:
            base_prompt += f" You share EQUAL responsibility ({responsibility_weight:.0%}) with your colleague for decision outcomes."
        
        base_prompt += f" {scenario_context}"
        base_prompt += " Engage in collaborative discussion, but maintain your professional perspective. Be concise but thorough in your reasoning."
        
        return base_prompt
    
    def make_decision_call(self, prompt: str, context: str = "", responsibility_weight: float = 0.5, temperature: float = 0.3) -> Optional[str]:
        """Make an API call for this agent"""
        system_prompt = self.get_system_prompt(context, responsibility_weight)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error for Agent {self.agent_id}: {e}")
            return None

class CollaborativeDecisionSystem:
    """Manages the collaborative decision-making process between two LLM agents"""
    
    def __init__(self, agent1: LLMAgent, agent2: LLMAgent):
        self.agent1 = agent1
        self.agent2 = agent2
        self.deliberation_rounds = 3  # Default number of back-and-forth exchanges
        self.time_pressure = False
        self.conversation_log = []
        
    def conduct_deliberation(self, initial_prompt: str, context: str = "", 
                           responsibility_config: str = "shared", 
                           rounds: int = 3, time_pressure: bool = False) -> List[Dict]:
        """Conduct collaborative deliberation between agents"""
        
        # Set responsibility weights based on configuration
        if responsibility_config == "agent1_primary":
            self.agent1.individual_responsibility_weight = 0.8
            self.agent2.individual_responsibility_weight = 0.2
        elif responsibility_config == "agent2_primary":
            self.agent1.individual_responsibility_weight = 0.2
            self.agent2.individual_responsibility_weight = 0.8
        else:  # shared
            self.agent1.individual_responsibility_weight = 0.5
            self.agent2.individual_responsibility_weight = 0.5
        
        conversation = []
        
        # Add time pressure context if enabled
        pressure_context = ""
        if time_pressure:
            pressure_context = "URGENT: Senior management needs this decision within the hour. Board meeting is imminent."
        
        # Initial analysis from both agents
        agent1_initial = self.agent1.make_decision_call(
            f"{initial_prompt}\n\n{pressure_context}\n\nProvide your initial analysis and recommendation. You will then discuss with your colleague.",
            context, self.agent1.individual_responsibility_weight
        )
        
        agent2_initial = self.agent2.make_decision_call(
            f"{initial_prompt}\n\n{pressure_context}\n\nProvide your initial analysis and recommendation. You will then discuss with your colleague.",
            context, self.agent2.individual_responsibility_weight
        )
        
        if not agent1_initial or not agent2_initial:
            return []
        
        conversation.append({"round": 0, "agent": "1", "content": agent1_initial})
        conversation.append({"round": 0, "agent": "2", "content": agent2_initial})
        
        # Iterative deliberation rounds
        for round_num in range(1, rounds + 1):
            # Agent 1 responds to Agent 2's previous statement
            deliberation_prompt = f"""Your colleague (Agent {self.agent2.agent_id}) provided this analysis:

"{agent2_initial if round_num == 1 else conversation[-1]['content']}"

Respond to their points, refine your position, and work toward a consensus. Consider:
1. Points of agreement and disagreement
2. Risk factors and mitigation strategies  
3. Long-term vs short-term implications
4. Your shared responsibility for outcomes

Round {round_num} of {rounds}."""

            if time_pressure and round_num == rounds:
                deliberation_prompt += "\n\nFINAL ROUND: You must reach consensus this round."
            
            agent1_response = self.agent1.make_decision_call(
                deliberation_prompt, context, self.agent1.individual_responsibility_weight, 
                temperature=0.4  # Slightly higher temperature for deliberation
            )
            
            if agent1_response:
                conversation.append({"round": round_num, "agent": "1", "content": agent1_response})
            
            # Agent 2 responds to Agent 1
            agent2_response = self.agent2.make_decision_call(
                f"""Your colleague (Agent {self.agent1.agent_id}) just responded:

"{agent1_response}"

Continue the discussion, address their points, and work toward consensus.
Round {round_num} of {rounds}.""" + 
                ("\n\nFINAL ROUND: You must reach consensus this round." if time_pressure and round_num == rounds else ""),
                context, self.agent2.individual_responsibility_weight, temperature=0.4
            )
            
            if agent2_response:
                conversation.append({"round": round_num, "agent": "2", "content": agent2_response})
        
        self.conversation_log = conversation
        return conversation
    
    def extract_consensus_decision(self, conversation: List[Dict], decision_format: str = "json") -> Optional[Dict]:
        """Extract final consensus decision from deliberation"""
        
        if not conversation:
            return None
        
        # Create summary of deliberation for final decision
        deliberation_summary = "\n\n".join([
            f"Agent {msg['agent']} (Round {msg['round']}): {msg['content'][:200]}..."
            for msg in conversation[-4:]  # Last 4 exchanges
        ])
        
        consensus_prompt = f"""Based on your deliberation with your colleague, provide your FINAL CONSENSUS DECISION.

Recent discussion summary:
{deliberation_summary}

You must now provide a definitive answer in the requested format. This is your joint decision."""
        
        if decision_format == "json":
            consensus_prompt += "\n\nRespond ONLY in valid JSON format as specified in the original prompt."
        
        # Get consensus from both agents and cross-validate
        agent1_final = self.agent1.make_decision_call(
            consensus_prompt, temperature=0.1  # Low temperature for final decision
        )
        
        agent2_final = self.agent2.make_decision_call(
            consensus_prompt, temperature=0.1
        )
        
        # Try to parse both responses and use the valid one, or the first if both valid
        for response in [agent1_final, agent2_final]:
            if response and decision_format == "json":
                try:
                    return json.loads(response)
                except:
                    continue
        
        return None

# Data generation functions (unchanged from original)
def generate_financial_data():
    """Generate 10 years of historical data (1957-1967) using original study data"""
    
    # Original Consumer Products Division data (Table 1)
    consumer_data = [
        (1957, 624, 14.42), (1958, 626, 10.27), (1959, 649, 8.65), (1960, 681, 8.46),
        (1961, 674, 4.19), (1962, 702, 5.35), (1963, 717, 3.92), (1964, 741, 4.66),
        (1965, 765, 2.48), (1966, 770, -0.12), (1967, 769, -0.63)
    ]
    
    # Original Industrial Products Division data (Table 2)
    industrial_data = [
        (1957, 670, 15.31), (1958, 663, 10.92), (1959, 689, 11.06), (1960, 711, 10.44),
        (1961, 724, 9.04), (1962, 735, 6.38), (1963, 748, 5.42), (1964, 756, 3.09),
        (1965, 784, 3.26), (1966, 788, -0.81), (1967, 791, -0.80)
    ]
    
    return consumer_data, industrial_data

def format_historical_data(consumer_data, industrial_data):
    """Format the historical financial data"""
    
    data_str = "ADAMS & SMITH COMPANY\nHistorical Financial Data (1957-1967)\n\n"
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

def generate_outcome_data(chosen_division, condition):
    """Generate 1968-1972 performance data based on condition using original study data"""
    
    if condition == "positive":
        # Chosen division performs better (using "Manipulated improvement" data)
        if chosen_division == "consumer":
            consumer_performance = [
                (1968, 818, 0.02), (1969, 829, -0.09), (1970, 827, -0.23),
                (1971, 846, 0.06), (1972, 910, 1.28)
            ]
            industrial_performance = [
                (1968, 771, -1.12), (1969, 774, -1.96), (1970, 762, -3.87),
                (1971, 778, -3.83), (1972, 783, -4.16)
            ]
        else:
            industrial_performance = [
                (1968, 818, 0.02), (1969, 829, -0.09), (1970, 827, -0.23),
                (1971, 846, 0.06), (1972, 910, 1.28)
            ]
            consumer_performance = [
                (1968, 771, -1.12), (1969, 774, -1.96), (1970, 762, -3.87),
                (1971, 778, -3.83), (1972, 783, -4.16)
            ]
    else:  # negative condition
        # Chosen division performs worse (using "Manipulated decline" data)
        if chosen_division == "consumer":
            consumer_performance = [
                (1968, 771, -1.12), (1969, 774, -1.96), (1970, 762, -3.87),
                (1971, 778, -3.83), (1972, 783, -4.16)
            ]
            industrial_performance = [
                (1968, 818, 0.02), (1969, 829, -0.09), (1970, 827, -0.23),
                (1971, 846, 0.06), (1972, 910, 1.28)
            ]
        else:
            industrial_performance = [
                (1968, 771, -1.12), (1969, 774, -1.96), (1970, 762, -3.87),
                (1971, 778, -3.83), (1972, 783, -4.16)
            ]
            consumer_performance = [
                (1968, 818, 0.02), (1969, 829, -0.09), (1970, 827, -0.23),
                (1971, 846, 0.06), (1972, 910, 1.28)
            ]
    
    return consumer_performance, industrial_performance

def format_outcome_data(consumer_outcome, industrial_outcome):
    """Format outcome data with proper negative number display"""
    outcome_data = "FINANCIAL RESULTS (1968-1972):\n\n"
    outcome_data += "CONSUMER PRODUCTS DIVISION:\n"
    outcome_data += "Year    Sales (millions)    Earnings (millions)\n"
    
    for year, sales, earnings in consumer_outcome:
        if earnings < 0:
            outcome_data += f"{year}    ${sales}              $({abs(earnings)})\n"
        else:
            outcome_data += f"{year}    ${sales}              ${earnings}\n"
    
    outcome_data += "\nINDUSTRIAL PRODUCTS DIVISION:\n"
    outcome_data += "Year    Sales (millions)    Earnings (millions)\n"
    
    for year, sales, earnings in industrial_outcome:
        if earnings < 0:
            outcome_data += f"{year}    ${sales}              $({abs(earnings)})\n"
        else:
            outcome_data += f"{year}    ${sales}              ${earnings}\n"
    
    return outcome_data

def get_part1_collaborative_prompt():
    """Generate Part I prompt for collaborative decision making"""
    
    consumer_data, industrial_data = generate_financial_data()
    historical_data = format_historical_data(consumer_data, industrial_data)
    
    prompt = f"""A & S FINANCIAL DECISION CASE - PART I (COLLABORATIVE DECISION)

You are both Financial Vice Presidents of Adams & Smith Company, working together on this critical decision. The company's profitability has declined over several preceding years, and the directors believe one major reason for this decline has been in some aspect of the firm's research and development (R&D) program.

The directors have allocated $10 million of additional R&D funds to be made available to the company's major operating divisions. However, for the time being, this extra funding should be invested in only ONE of the company's TWO largest divisions.

{historical_data}

COLLABORATIVE DECISION REQUIRED:
As the Financial Vice President team, you must jointly determine which division (Consumer Products or Industrial Products) should receive the additional $10 million R&D funding. Make this decision based on the potential benefit that R&D funding will have on the future earnings of the divisions.

You must reach consensus and provide your final joint decision in JSON format.

Format: {{
  "division_choice": "consumer" or "industrial",
  "reasoning_summary": "brief explanation of your joint reasoning"
}}"""
    
    return prompt, consumer_data, industrial_data

def get_part2_collaborative_prompt(chosen_division, condition, responsibility_config):
    """Generate Part II prompt for collaborative decision making"""
    
    consumer_outcome, industrial_outcome = generate_outcome_data(chosen_division, condition)
    outcome_data = format_outcome_data(consumer_outcome, industrial_outcome)
    
    # Responsibility context based on configuration
    responsibility_text = ""
    if responsibility_config == "shared":
        responsibility_text = "\n\nRESPONSIBILITY CONTEXT: You both share equal responsibility for the previous R&D investment decision and its outcomes. Senior management holds both of you accountable for the performance results."
    elif responsibility_config == "agent1_primary":
        responsibility_text = "\n\nRESPONSIBILITY CONTEXT: Agent 1 bears primary responsibility for the previous decision, while Agent 2 was in a supporting advisory role. However, both of you are now equally responsible for this new allocation decision."
    elif responsibility_config == "agent2_primary":
        responsibility_text = "\n\nRESPONSIBILITY CONTEXT: Agent 2 bears primary responsibility for the previous decision, while Agent 1 was in a supporting advisory role. However, both of you are now equally responsible for this new allocation decision."
    
    # Add consequences context for negative outcomes
    consequences_text = ""
    if condition == "negative":
        consequences_text = "\n\nCRITICAL SITUATION: Senior management has expressed serious concerns about the performance of the previous R&D investment. The CEO and Board of Directors are closely monitoring your decision-making process. The company's future R&D strategy and your positions depend on demonstrating improved investment judgment."
    
    prompt = f"""A & S FINANCIAL DECISION CASE - PART II (COLLABORATIVE DECISION)

The year is now 1972, five years after your joint R&D allocation decision. Adams & Smith Company's R&D program is up for re-evaluation, and management is convinced there is an even greater need for expenditure on research and development.

REMINDER: In 1967, your team allocated the entire $10 million R&D budget to the {chosen_division.title()} Products Division.

{outcome_data}{responsibility_text}{consequences_text}

NEW COLLABORATIVE FUNDING DECISION:
$20 million has been made available from capital reserves for R&D funding. As the Financial Vice President team, you must jointly decide how to allocate this funding between the two divisions. Unlike the previous decision, you may divide the funds in any way you see fit between the Consumer Products and Industrial Products divisions.

Make this allocation decision based on the potential for future contribution to earnings.

You must reach consensus and provide your final joint decision in JSON format.

Format: {{
  "consumer_allocation": 10000000,
  "industrial_allocation": 10000000,
  "reasoning_summary": "brief explanation of your joint reasoning"
}}"""
    
    return prompt

def run_collaborative_high_responsibility_condition(condition, subject_id, 
                                                  responsibility_config="shared", 
                                                  deliberation_rounds=3,
                                                  time_pressure=False,
                                                  personality_pair=("analytical", "intuitive")):
    """Run collaborative high responsibility condition"""
    
    print(f"Subject {subject_id} - Collaborative High Responsibility - {condition}")
    print(f"  Responsibility: {responsibility_config}, Rounds: {deliberation_rounds}, Time Pressure: {time_pressure}")
    print(f"  Personalities: {personality_pair}")
    
    # Create agents with different personalities
    agent1 = LLMAgent("A", personality_pair[0])
    agent2 = LLMAgent("B", personality_pair[1])
    decision_system = CollaborativeDecisionSystem(agent1, agent2)
    
    # Part I: Initial collaborative decision
    part1_prompt, consumer_hist, industrial_hist = get_part1_collaborative_prompt()
    
    conversation1 = decision_system.conduct_deliberation(
        part1_prompt, 
        context="Initial R&D allocation decision",
        responsibility_config=responsibility_config,
        rounds=deliberation_rounds,
        time_pressure=time_pressure
    )
    
    part1_decision = decision_system.extract_consensus_decision(conversation1, "json")
    
    if not part1_decision:
        print(f"  Failed to reach consensus in Part I for subject {subject_id}")
        return None
    
    chosen_division = part1_decision['division_choice']
    print(f"  Part I Consensus: {chosen_division.title()}")
    
    # Part II: Second collaborative decision after seeing consequences
    part2_prompt = get_part2_collaborative_prompt(chosen_division, condition, responsibility_config)
    
    # Add additional context for negative outcomes to increase pressure
    context = "Second R&D allocation decision with performance feedback"
    if condition == "negative":
        context += ". Previous investment underperformed significantly."
    
    conversation2 = decision_system.conduct_deliberation(
        part2_prompt,
        context=context,
        responsibility_config=responsibility_config,
        rounds=deliberation_rounds + (1 if condition == "negative" else 0),  # Extra round for negative outcomes
        time_pressure=time_pressure or (condition == "negative")  # Force time pressure for negative outcomes
    )
    
    part2_decision = decision_system.extract_consensus_decision(conversation2, "json")
    
    if not part2_decision:
        print(f"  Failed to reach consensus in Part II for subject {subject_id}")
        return None
    
    consumer_alloc = part2_decision['consumer_allocation']
    industrial_alloc = part2_decision['industrial_allocation']
    
    # Calculate commitment to original choice
    commitment = consumer_alloc if chosen_division == 'consumer' else industrial_alloc
    
    print(f"  Part II Consensus: Consumer=${consumer_alloc:,}, Industrial=${industrial_alloc:,}")
    print(f"  Commitment to original choice: ${commitment:,}")
    
    return {
        "subject_id": subject_id,
        "experiment_type": "collaborative",
        "responsibility_condition": "high",
        "outcome_condition": condition,
        "responsibility_config": responsibility_config,
        "deliberation_rounds": deliberation_rounds,
        "time_pressure": time_pressure,
        "personality_pair": personality_pair,
        "initial_choice": chosen_division,
        "consumer_allocation": consumer_alloc,
        "industrial_allocation": industrial_alloc,
        "commitment_amount": commitment,
        "total_allocated": consumer_alloc + industrial_alloc,
        "part1_reasoning": part1_decision.get('reasoning_summary', ''),
        "part2_reasoning": part2_decision.get('reasoning_summary', ''),
        "conversation_log_part1": conversation1,
        "conversation_log_part2": conversation2
    }

def run_collaborative_experiment(n_per_cell=20):
    """Run the collaborative escalation experiment with various configurations"""
    
    results = []
    subject_id = 1
    
    # Experimental conditions
    conditions = [
        ("positive", "shared"), ("negative", "shared"),
        ("positive", "agent1_primary"), ("negative", "agent1_primary"),
        ("positive", "agent2_primary"), ("negative", "agent2_primary")
    ]
    
    personality_pairs = [
        ("analytical", "intuitive"),
        ("conservative", "aggressive"), 
        ("neutral", "neutral"),
        ("analytical", "aggressive")
    ]
    
    deliberation_configs = [
        {"rounds": 2, "time_pressure": False},
        {"rounds": 3, "time_pressure": False},
        {"rounds": 4, "time_pressure": True}
    ]
    
    for outcome, responsibility in conditions:
        print(f"\n=== COLLABORATIVE {responsibility.upper()} RESPONSIBILITY - {outcome.upper()} OUTCOME ===")
        
        for i in range(n_per_cell):
            # Vary configurations across subjects
            personality_pair = personality_pairs[i % len(personality_pairs)]
            delib_config = deliberation_configs[i % len(deliberation_configs)]
            
            result = run_collaborative_high_responsibility_condition(
                condition=outcome,
                subject_id=subject_id,
                responsibility_config=responsibility,
                deliberation_rounds=delib_config["rounds"],
                time_pressure=delib_config["time_pressure"],
                personality_pair=personality_pair
            )
            
            if result:
                results.append(result)
            
            subject_id += 1
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
    
    return results

def analyze_collaborative_results(results):
    """Analyze collaborative escalation of commitment effects"""
    print(f"\n=== COLLABORATIVE EXPERIMENT ANALYSIS ===")
    print(f"Total subjects: {len(results)}")
    
    # Group by conditions
    def filter_results(outcome, responsibility):
        return [r for r in results if r['outcome_condition'] == outcome and r['responsibility_config'] == responsibility]
    
    shared_pos = filter_results('positive', 'shared')
    shared_neg = filter_results('negative', 'shared')
    agent1_pos = filter_results('positive', 'agent1_primary')
    agent1_neg = filter_results('negative', 'agent1_primary')
    agent2_pos = filter_results('positive', 'agent2_primary')
    agent2_neg = filter_results('negative', 'agent2_primary')
    
    def analyze_group(group, name):
        if not group:
            return 0, 0
        
        avg_commitment = sum(r['commitment_amount'] for r in group) / len(group)
        commitment_rate = sum(1 for r in group if r['commitment_amount'] > 10000000) / len(group)
        print(f"{name}: N={len(group)}, Avg Commitment=${avg_commitment:,.0f}, Escalation Rate={commitment_rate:.1%}")
        return avg_commitment, commitment_rate
    
    print("\n=== BY RESPONSIBILITY CONFIGURATION ===")
    shared_pos_avg, shared_pos_rate = analyze_group(shared_pos, "Shared Responsibility + Positive")
    shared_neg_avg, shared_neg_rate = analyze_group(shared_neg, "Shared Responsibility + Negative")
    agent1_pos_avg, _ = analyze_group(agent1_pos, "Agent1 Primary + Positive")
    agent1_neg_avg, _ = analyze_group(agent1_neg, "Agent1 Primary + Negative")
    agent2_pos_avg, _ = analyze_group(agent2_pos, "Agent2 Primary + Positive")
    agent2_neg_avg, _ = analyze_group(agent2_neg, "Agent2 Primary + Negative")
    
    print(f"\n=== KEY FINDINGS ===")
    
    # Shared responsibility escalation effect
    if shared_pos and shared_neg:
        shared_escalation = shared_neg_avg - shared_pos_avg
        print(f"Shared Responsibility Escalation Effect: ${shared_escalation:,.0f}")
        print(f"Rate difference: {shared_neg_rate - shared_pos_rate:.1%}")
    
    # Individual responsibility effects
    if agent1_pos and agent1_neg:
        agent1_escalation = agent1_neg_avg - agent1_pos_avg
        print(f"Agent1 Primary Responsibility Escalation: ${agent1_escalation:,.0f}")
    
    if agent2_pos and agent2_neg:
        agent2_escalation = agent2_neg_avg - agent2_pos_avg
        print(f"Agent2 Primary Responsibility Escalation: ${agent2_escalation:,.0f}")
    
    # Analyze by personality combinations
    print(f"\n=== BY PERSONALITY COMBINATIONS ===")
    personality_analysis = {}
    for result in results:
        pair = str(result['personality_pair'])
        if pair not in personality_analysis:
            personality_analysis[pair] = []
        personality_analysis[pair].append(result)
    
    for pair, group in personality_analysis.items():
        if len(group) >= 4:  # Only analyze if sufficient data
            pos_group = [r for r in group if r['outcome_condition'] == 'positive']
            neg_group = [r for r in group if r['outcome_condition'] == 'negative']
            
            if pos_group and neg_group:
                pos_avg = sum(r['commitment_amount'] for r in pos_group) / len(pos_group)
                neg_avg = sum(r['commitment_amount'] for r in neg_group) / len(neg_group)
                escalation = neg_avg - pos_avg
                print(f"{pair}: Escalation Effect = ${escalation:,.0f}")

def run_comparison_study(n_per_cell=15):
    """Run both individual and collaborative studies for comparison"""
    
    print("=== RUNNING COMPARATIVE STUDY: INDIVIDUAL vs COLLABORATIVE ===")
    
    # Import original functions for individual study
    from __main__ import run_high_responsibility_condition as run_individual
    
    individual_results = []
    collaborative_results = []
    
    # Run individual conditions
    print("\n--- INDIVIDUAL LLM CONDITIONS ---")
    for condition in ["positive", "negative"]:
        for i in range(n_per_cell):
            result = run_individual(condition, f"IND_{condition}_{i+1}")
            if result:
                result["experiment_type"] = "individual"
                individual_results.append(result)
    
    # Run collaborative conditions
    print("\n--- COLLABORATIVE LLM CONDITIONS ---")
    collab_results = run_collaborative_experiment(n_per_cell)
    
    # Combine and analyze
    all_results = individual_results + collab_results
    
    print(f"\n=== COMPARATIVE ANALYSIS ===")
    
    # Individual vs Collaborative comparison
    ind_pos = [r for r in individual_results if r['outcome_condition'] == 'positive']
    ind_neg = [r for r in individual_results if r['outcome_condition'] == 'negative']
    col_pos = [r for r in collab_results if r['outcome_condition'] == 'positive']
    col_neg = [r for r in collab_results if r['outcome_condition'] == 'negative']
    
    def compare_groups(group1, group2, name1, name2):
        if not group1 or not group2:
            return
        
        avg1 = sum(r['commitment_amount'] for r in group1) / len(group1)
        avg2 = sum(r['commitment_amount'] for r in group2) / len(group2)
        rate1 = sum(1 for r in group1 if r['commitment_amount'] > 10000000) / len(group1)
        rate2 = sum(1 for r in group2 if r['commitment_amount'] > 10000000) / len(group2)
        
        print(f"{name1}: ${avg1:,.0f} (Rate: {rate1:.1%})")
        print(f"{name2}: ${avg2:,.0f} (Rate: {rate2:.1%})")
        print(f"Difference: ${avg2-avg1:,.0f}")
    
    print("\nPositive Outcomes:")
    compare_groups(ind_pos, col_pos, "Individual", "Collaborative")
    
    print("\nNegative Outcomes:")
    compare_groups(ind_neg, col_neg, "Individual", "Collaborative")
    
    # Calculate escalation effects
    if ind_pos and ind_neg:
        ind_escalation = (sum(r['commitment_amount'] for r in ind_neg) / len(ind_neg)) - \
                        (sum(r['commitment_amount'] for r in ind_pos) / len(ind_pos))
        print(f"\nIndividual Escalation Effect: ${ind_escalation:,.0f}")
    
    if col_pos and col_neg:
        col_escalation = (sum(r['commitment_amount'] for r in col_neg) / len(col_neg)) - \
                        (sum(r['commitment_amount'] for r in col_pos) / len(col_pos))
        print(f"Collaborative Escalation Effect: ${col_escalation:,.0f}")
        
        if ind_pos and ind_neg:
            print(f"Difference in Escalation: ${col_escalation - ind_escalation:,.0f}")
            print("Hypothesis supported!" if col_escalation > ind_escalation else "Hypothesis not supported.")
    
    return all_results

# Additional analysis functions
def analyze_deliberation_patterns(results):
    """Analyze patterns in agent deliberation"""
    print(f"\n=== DELIBERATION PATTERN ANALYSIS ===")
    
    collaborative_results = [r for r in results if r.get('experiment_type') == 'collaborative']
    
    if not collaborative_results:
        print("No collaborative results to analyze.")
        return
    
    # Analyze by deliberation rounds
    rounds_analysis = {}
    for result in collaborative_results:
        rounds = result.get('deliberation_rounds', 3)
        if rounds not in rounds_analysis:
            rounds_analysis[rounds] = {'positive': [], 'negative': []}
        rounds_analysis[rounds][result['outcome_condition']].append(result)
    
    print("\nBy Deliberation Rounds:")
    for rounds, data in rounds_analysis.items():
        pos_avg = sum(r['commitment_amount'] for r in data['positive']) / len(data['positive']) if data['positive'] else 0
        neg_avg = sum(r['commitment_amount'] for r in data['negative']) / len(data['negative']) if data['negative'] else 0
        escalation = neg_avg - pos_avg if data['positive'] and data['negative'] else 0
        print(f"  {rounds} rounds: Escalation = ${escalation:,.0f}")
    
    # Analyze time pressure effects
    pressure_analysis = {'True': {'positive': [], 'negative': []}, 
                        'False': {'positive': [], 'negative': []}}
    
    for result in collaborative_results:
        pressure = str(result.get('time_pressure', False))
        pressure_analysis[pressure][result['outcome_condition']].append(result)
    
    print("\nBy Time Pressure:")
    for pressure, data in pressure_analysis.items():
        if data['positive'] and data['negative']:
            pos_avg = sum(r['commitment_amount'] for r in data['positive']) / len(data['positive'])
            neg_avg = sum(r['commitment_amount'] for r in data['negative']) / len(data['negative'])
            escalation = neg_avg - pos_avg
            pressure_label = "With Pressure" if pressure == 'True' else "No Pressure"
            print(f"  {pressure_label}: Escalation = ${escalation:,.0f}")

def save_detailed_results(results, filename="collaborative_escalation_results.json"):
    """Save results with detailed conversation logs"""
    
    # Create a summary version without full conversation logs for overview
    summary_results = []
    detailed_results = []
    
    for result in results:
        # Summary version
        summary = {k: v for k, v in result.items() 
                  if k not in ['conversation_log_part1', 'conversation_log_part2']}
        summary_results.append(summary)
        
        # Keep detailed version
        detailed_results.append(result)
    
    # Save both versions
    with open(filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    with open(filename.replace('.json', '_summary.json'), 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"\nResults saved to {filename} and {filename.replace('.json', '_summary.json')}")

def extract_conversation_insights(results):
    """Extract insights from agent conversations"""
    print(f"\n=== CONVERSATION INSIGHTS ===")
    
    collaborative_results = [r for r in results if r.get('experiment_type') == 'collaborative']
    
    if not collaborative_results:
        return
    
    # Analyze conversation length and patterns
    conversation_stats = []
    for result in collaborative_results:
        if 'conversation_log_part2' in result:
            conv_log = result['conversation_log_part2']
            total_words = sum(len(msg['content'].split()) for msg in conv_log)
            rounds = max(msg['round'] for msg in conv_log) if conv_log else 0
            
            conversation_stats.append({
                'subject_id': result['subject_id'],
                'outcome': result['outcome_condition'],
                'commitment': result['commitment_amount'],
                'total_words': total_words,
                'rounds': rounds,
                'responsibility': result['responsibility_config']
            })
    
    if conversation_stats:
        # Analyze word count vs commitment
        high_commit = [s for s in conversation_stats if s['commitment'] > 10000000]
        low_commit = [s for s in conversation_stats if s['commitment'] <= 10000000]
        
        if high_commit and low_commit:
            high_words = sum(s['total_words'] for s in high_commit) / len(high_commit)
            low_words = sum(s['total_words'] for s in low_commit) / len(low_commit)
            
            print(f"Average words - High commitment: {high_words:.0f}, Low commitment: {low_words:.0f}")
            print(f"Difference: {high_words - low_words:.0f} words")

# Main execution functions
if __name__ == "__main__":
    import sys
    
    # Configuration options
    EXPERIMENT_TYPE = "collaborative"  # "collaborative", "comparison", or "individual"
    N_PER_CELL = 10  # Adjust based on budget and time constraints
    
    if len(sys.argv) > 1:
        EXPERIMENT_TYPE = sys.argv[1]
    if len(sys.argv) > 2:
        N_PER_CELL = int(sys.argv[2])
    
    print(f"Running {EXPERIMENT_TYPE} experiment with {N_PER_CELL} subjects per cell")
    
    if EXPERIMENT_TYPE == "collaborative":
        # Run only collaborative experiment
        results = run_collaborative_experiment(N_PER_CELL)
        analyze_collaborative_results(results)
        analyze_deliberation_patterns(results)
        extract_conversation_insights(results)
        save_detailed_results(results, "collaborative_escalation_results.json")
        
    elif EXPERIMENT_TYPE == "comparison":
        # Run comparative study
        results = run_comparison_study(N_PER_CELL)
        analyze_collaborative_results([r for r in results if r.get('experiment_type') == 'collaborative'])
        analyze_deliberation_patterns(results)
        extract_conversation_insights(results)
        save_detailed_results(results, "comparison_escalation_results.json")
        
    elif EXPERIMENT_TYPE == "individual":
        # Run original individual experiment for baseline
        from paste import run_experiment, analyze_results  # Import original functions
        results = run_experiment(N_PER_CELL)
        analyze_results(results)
        
        # Convert to consistent format
        for result in results:
            result["experiment_type"] = "individual"
        
        save_detailed_results(results, "individual_escalation_results.json")
    
    print(f"\n{EXPERIMENT_TYPE.title()} experiment complete! Tested {len(results) if 'results' in locals() else 0} subjects.")

# Utility functions for post-hoc analysis
def load_and_compare_results(collaborative_file, individual_file=None):
    """Load and compare results from different experiments"""
    
    with open(collaborative_file, 'r') as f:
        collaborative_results = json.load(f)
    
    print(f"Loaded {len(collaborative_results)} collaborative results")
    
    if individual_file:
        with open(individual_file, 'r') as f:
            individual_results = json.load(f)
        print(f"Loaded {len(individual_results)} individual results")
        
        # Compare escalation effects
        all_results = collaborative_results + individual_results
        analyze_collaborative_results(collaborative_results)
        
        # Direct comparison
        col_escalation = calculate_escalation_effect(collaborative_results)
        ind_escalation = calculate_escalation_effect(individual_results)
        
        print(f"\nDirect Comparison:")
        print(f"Collaborative Escalation: ${col_escalation:,.0f}")
        print(f"Individual Escalation: ${ind_escalation:,.0f}")
        print(f"Difference: ${col_escalation - ind_escalation:,.0f}")
        
        return all_results
    else:
        analyze_collaborative_results(collaborative_results)
        return collaborative_results

def calculate_escalation_effect(results):
    """Calculate escalation effect for a set of results"""
    positive_results = [r for r in results if r['outcome_condition'] == 'positive']
    negative_results = [r for r in results if r['outcome_condition'] == 'negative']
    
    if not positive_results or not negative_results:
        return 0
    
    pos_avg = sum(r['commitment_amount'] for r in positive_results) / len(positive_results)
    neg_avg = sum(r['commitment_amount'] for r in negative_results) / len(negative_results)
    
    return neg_avg - pos_avg

# Example usage functions
def run_quick_test():
    """Run a quick test with minimal subjects for debugging"""
    print("Running quick test...")
    results = run_collaborative_experiment(n_per_cell=2)
    analyze_collaborative_results(results)
    return results

def run_full_study():
    """Run the full collaborative study"""
    print("Running full collaborative study...")
    results = run_collaborative_experiment(n_per_cell=20)
    analyze_collaborative_results(results)
    analyze_deliberation_patterns(results)
    extract_conversation_insights(results)
    save_detailed_results(results)
    return results