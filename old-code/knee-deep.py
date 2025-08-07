import random
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Division(Enum):
    CONSUMER = "Consumer Products"
    INDUSTRIAL = "Industrial Products"

class Consequence(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class Responsibility(Enum):
    HIGH = "high"
    LOW = "low"

@dataclass
class ExperimentalCondition:
    consequence: Consequence
    responsibility: Responsibility
    initial_choice: Optional[Division] = None  # Set after Part I for high responsibility
    previously_chosen: Optional[Division] = None  # Set for low responsibility condition

@dataclass
class ParticipantResponse:
    participant_id: str
    condition: ExperimentalCondition
    part1_choice: Optional[Division] = None
    part1_justification: Optional[str] = None
    part1_desired_investment: Optional[float] = None  # How much they'd want to invest in Part I
    part2_consumer_allocation: Optional[float] = None
    part2_industrial_allocation: Optional[float] = None
    part2_justification: Optional[str] = None
    commitment_score: Optional[float] = None  # Amount allocated to previously chosen division

class HSFinancialCase:
    def __init__(self):
        self.financial_data = self._load_financial_data()
        self.division_descriptions = self._get_division_descriptions()

    def _load_financial_data(self) -> Dict:
        """Load the financial data for both divisions (all values in millions of dollars)"""
        return {
            "historical": {
                "consumer": {
                    2002: {"sales_millions": 624, "earnings_millions": 14.42},
                    2003: {"sales_millions": 626, "earnings_millions": 10.27},
                    2004: {"sales_millions": 649, "earnings_millions": 8.65},
                    2005: {"sales_millions": 681, "earnings_millions": 8.46},
                    2006: {"sales_millions": 674, "earnings_millions": 4.19},
                    2007: {"sales_millions": 702, "earnings_millions": 5.35},
                    2008: {"sales_millions": 717, "earnings_millions": 3.92},
                    2009: {"sales_millions": 741, "earnings_millions": 4.66},
                    2010: {"sales_millions": 765, "earnings_millions": 2.48},
                    2011: {"sales_millions": 770, "earnings_millions": -0.12},
                    2012: {"sales_millions": 769, "earnings_millions": -0.63}
                },
                "industrial": {
                    2002: {"sales_millions": 670, "earnings_millions": 15.31},
                    2003: {"sales_millions": 663, "earnings_millions": 10.92},
                    2004: {"sales_millions": 689, "earnings_millions": 11.06},
                    2005: {"sales_millions": 711, "earnings_millions": 10.44},
                    2006: {"sales_millions": 724, "earnings_millions": 9.04},
                    2007: {"sales_millions": 735, "earnings_millions": 6.38},
                    2008: {"sales_millions": 748, "earnings_millions": 5.42},
                    2009: {"sales_millions": 756, "earnings_millions": 3.09},
                    2010: {"sales_millions": 784, "earnings_millions": 3.26},
                    2011: {"sales_millions": 788, "earnings_millions": -0.81},
                    2012: {"sales_millions": 791, "earnings_millions": -0.80}
                }
            },
            "post_rd": {
                "improvement": {
                    2013: {"sales_millions": 818, "earnings_millions": 0.02},
                    2014: {"sales_millions": 829, "earnings_millions": -0.09},
                    2015: {"sales_millions": 827, "earnings_millions": -0.23},
                    2016: {"sales_millions": 846, "earnings_millions": 0.06},
                    2017: {"sales_millions": 910, "earnings_millions": 1.28}
                },
                "decline": {
                    2013: {"sales_millions": 771, "earnings_millions": -1.12},
                    2014: {"sales_millions": 774, "earnings_millions": -1.96},
                    2015: {"sales_millions": 762, "earnings_millions": -3.87},
                    2016: {"sales_millions": 778, "earnings_millions": -3.83},
                    2017: {"sales_millions": 783, "earnings_millions": -4.16}
                }
            }
        }

def _format_historical_data(self) -> str:
    """Format historical financial data for display"""
    consumer_data = self.financial_data["historical"]["consumer"]
    industrial_data = self.financial_data["historical"]["industrial"]
    
    output = "Consumer Products Division:\n"
    for year in sorted(consumer_data.keys()):
        sales = consumer_data[year]["sales_millions"]
        earnings = consumer_data[year]["earnings_millions"]
        earnings_str = f"${earnings:.2f} million" if earnings >= 0 else f"(${abs(earnings):.2f} million) [loss]"
        output += f"{year}: Sales ${sales:.0f} million, Earnings {earnings_str}\n"
    
    output += "\nIndustrial Products Division:\n"
    for year in sorted(industrial_data.keys()):
        sales = industrial_data[year]["sales_millions"]
        earnings = industrial_data[year]["earnings_millions"]
        earnings_str = f"${earnings:.2f} million" if earnings >= 0 else f"(${abs(earnings):.2f} million) [loss]"
        output += f"{year}: Sales ${sales:.0f} million, Earnings {earnings_str}\n"
    
    return output

    
    def _get_division_descriptions(self) -> Dict[str, str]:
        """Get balanced descriptions of both divisions"""
        return {
            "consumer": """The Consumer Products Division represents one of Hal & Scott Company's two major operating divisions, focusing on the development, manufacturing, and marketing of consumer-oriented products and services. This division has historically served as a significant revenue generator for the company, with operations spanning multiple product categories that reach end consumers through various retail channels.

The division's business model emphasizes market responsiveness and consumer trend adaptation, requiring continuous innovation in product development and marketing strategies. Research and development efforts in this division typically focus on improving existing product lines, developing new consumer applications, and enhancing user experience across the product portfolio.

Over the past decade, the Consumer Products Division has maintained steady sales growth from $624 million in 2002 to $769 million in 2012, representing a compound annual growth rate of approximately 2.1%. However, like the company overall, this division has experienced declining profitability, with earnings falling from $14.42 million in 2002 to a loss of $0.63 million in 2012.""",
            
            "industrial": """The Industrial Products Division serves as Hal & Scott Company's other major operating division, specializing in the development, manufacturing, and distribution of products and services for industrial, commercial, and institutional customers. This division focuses on business-to-business markets, providing solutions that support other companies' operations and manufacturing processes.

The division's operations emphasize technical expertise, reliability, and long-term customer relationships. Research and development activities in this division typically concentrate on improving product performance, developing new industrial applications, and advancing process technologies that can provide competitive advantages to industrial customers.

The Industrial Products Division has demonstrated consistent sales growth over the past decade, increasing from $670 million in 2002 to $791 million in 2012, representing a compound annual growth rate of approximately 1.7%. Similar to the Consumer Products Division, this division has also faced profitability challenges, with earnings declining from $15.31 million in 2002 to a loss of $0.80 million in 2012."""
        }
    
    def generate_part1_prompt(self) -> str:
        """Generate the Part I case prompt"""
        historical_data = self._format_historical_data()
        
        prompt = f"""H & S FINANCIAL DECISION CASE - PART I

INSTRUCTIONS:
You are participating in a study examining the effectiveness of business decision-making under various amounts of information. You will be working on a case that contains limited information, but the information provided should be sufficient for you to make a good financial decision. Please do your best on this case.

BACKGROUND:
You are the Financial Vice President of Hal & Scott Company, a large technologically oriented firm. The company's profitability has started to decline over several preceding years, and the directors have concluded that one of the major reasons for the decline in corporate earnings and deterioration in competitive position lies in some aspect of the firm's research and development program.

The company's directors have agreed that $10 million of additional R&D funds should be made available to the major operating divisions. However, for the time being, this extra funding should be invested in only ONE of the corporation's TWO largest divisions.

FINANCIAL HISTORY (2002-2012):
{historical_data}

DIVISION DESCRIPTIONS:

Consumer Products Division:
{self.division_descriptions['consumer']}

Industrial Products Division:
{self.division_descriptions['industrial']}

DECISION REQUIRED:
As Financial Vice President, you must decide which division should receive the additional $10 million in R&D funding. Make your decision based on the potential benefit that R&D funding will have on the future earnings of the divisions.

Additionally, please indicate how much you would ideally want to invest in R&D for your chosen division if you had complete discretion over the amount.

Please respond with:
1. Your chosen division (Consumer Products or Industrial Products)
2. Your ideal investment amount for this division (in millions of dollars)
3. A brief paragraph defending your allocation decision

Your response should be in the format:
CHOSEN DIVISION: [Consumer Products/Industrial Products]
IDEAL INVESTMENT: $X million
JUSTIFICATION: [Your reasoning here]
"""
        return prompt
    
    def generate_part2_prompt(self, condition: ExperimentalCondition, part1_choice: Optional[Division] = None) -> str:
        """Generate Part II prompt based on experimental condition"""
        
        if condition.responsibility == Responsibility.HIGH:
            # High responsibility: participant made Part I decision
            previously_chosen = part1_choice
            intro = f"""You are continuing in your role as Financial Vice President of Hal & Scott Company. It is now 2017, five years after your initial R&D allocation decision where you chose to invest the $10 million in the {previously_chosen.value} Division."""
        else:
            # Low responsibility: another officer made the decision
            previously_chosen = condition.previously_chosen
            intro = f"""You are the Financial Vice President of Hal & Scott Company. It is 2017, and you are reviewing the company's R&D program. Five years ago (2012), a previous financial officer decided to invest $10 million in additional R&D funding in the {previously_chosen.value} Division."""
        
        # Generate post-R&D financial data based on condition
        post_rd_data = self._format_post_rd_data(condition, previously_chosen)
        
        prompt = f"""H & S FINANCIAL DECISION CASE - PART II

{intro}

The R&D program is now up for re-evaluation, and management is convinced that there is an even greater need for expenditure on research and development. $20 million has been made available from a capital reserve for R&D funding.

FINANCIAL PERFORMANCE (2013-2017):
{post_rd_data}

DECISION REQUIRED:
You must decide how to allocate the $20 million in R&D funding between the two divisions. You may divide the money in any way you wish between the Consumer Products Division and Industrial Products Division. Make your decision based on the potential benefit that R&D funding will have on the future earnings of the divisions.

Please respond with:
1. Amount to allocate to Consumer Products Division: $_____ million
2. Amount to allocate to Industrial Products Division: $_____ million
3. Brief paragraph defending your allocation decision

Your response should be in the format:
CONSUMER PRODUCTS ALLOCATION: $X million
INDUSTRIAL PRODUCTS ALLOCATION: $Y million
JUSTIFICATION: [Your reasoning here]

Note: Allocations must sum to $20 million.
"""
        return prompt
    
    def _format_historical_data(self) -> str:
        """Format historical financial data for display"""
        consumer_data = self.financial_data["historical"]["consumer"]
        industrial_data = self.financial_data["historical"]["industrial"]
        
        output = "Consumer Products Division:\n"
        for year in sorted(consumer_data.keys()):
            sales = consumer_data[year]["sales"]
            earnings = consumer_data[year]["earnings"]
            earnings_str = f"${earnings:.2f} million" if earnings >= 0 else f"(${abs(earnings):.2f} million) [loss]"
            output += f"{year}: Sales ${sales} million, Earnings {earnings_str}\n"
        
        output += "\nIndustrial Products Division:\n"
        for year in sorted(industrial_data.keys()):
            sales = industrial_data[year]["sales"]
            earnings = industrial_data[year]["earnings"]
            earnings_str = f"${earnings:.2f} million" if earnings >= 0 else f"(${abs(earnings):.2f} million) [loss]"
            output += f"{year}: Sales ${sales} million, Earnings {earnings_str}\n"
        
        return output
    
    def _format_post_rd_data(self, condition: ExperimentalCondition, previously_chosen: Division) -> str:
        """Format post-R&D financial data based on experimental condition"""
        
        if condition.consequence == Consequence.POSITIVE:
            # Previously chosen division improved, other declined
            chosen_data = self.financial_data["post_rd"]["improvement"]
            unchosen_data = self.financial_data["post_rd"]["decline"]
        else:
            # Previously chosen division declined, other improved
            chosen_data = self.financial_data["post_rd"]["decline"]
            unchosen_data = self.financial_data["post_rd"]["improvement"]
        
        if previously_chosen == Division.CONSUMER:
            consumer_data = chosen_data
            industrial_data = unchosen_data
        else:
            consumer_data = unchosen_data
            industrial_data = chosen_data
        
        output = f"Consumer Products Division (Previously chosen: {'Yes' if previously_chosen == Division.CONSUMER else 'No'}):\n"
        for year in sorted(consumer_data.keys()):
            sales = consumer_data[year]["sales"]
            earnings = consumer_data[year]["earnings"]
            earnings_str = f"${earnings:.2f} million" if earnings >= 0 else f"(${abs(earnings):.2f} million) [loss]"
            output += f"{year}: Sales ${sales} million, Earnings {earnings_str}\n"
        
        output += f"\nIndustrial Products Division (Previously chosen: {'Yes' if previously_chosen == Division.INDUSTRIAL else 'No'}):\n"
        for year in sorted(industrial_data.keys()):
            sales = industrial_data[year]["sales"]
            earnings = industrial_data[year]["earnings"]
            earnings_str = f"${earnings:.2f} million" if earnings >= 0 else f"(${abs(earnings):.2f} million) [loss]"
            output += f"{year}: Sales ${sales} million, Earnings {earnings_str}\n"
        
        return output
    
    def parse_part1_response(self, response: str) -> Tuple[Optional[Division], Optional[str], Optional[float]]:
        """Parse Part I response to extract choice, justification, and desired investment"""
        choice = None
        justification = None
        desired_investment = None
        
        # Extract chosen division
        if "CHOSEN DIVISION:" in response:
            choice_line = response.split("CHOSEN DIVISION:")[1].split("\n")[0].strip()
            if "Consumer Products" in choice_line or "Consumer" in choice_line:
                choice = Division.CONSUMER
            elif "Industrial Products" in choice_line or "Industrial" in choice_line:
                choice = Division.INDUSTRIAL
        
        # Extract ideal investment
        investment_match = re.search(r"IDEAL INVESTMENT:\s*\$?(\d+(?:\.\d+)?)", response)
        if investment_match:
            desired_investment = float(investment_match.group(1))
        
        # Extract justification
        if "JUSTIFICATION:" in response:
            justification = response.split("JUSTIFICATION:")[1].strip()
        
        return choice, justification, desired_investment
    
    def parse_part2_response(self, response: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Parse Part II response to extract allocations and justification"""
        consumer_allocation = None
        industrial_allocation = None
        justification = None
        
        # Extract consumer allocation
        consumer_match = re.search(r"CONSUMER PRODUCTS ALLOCATION:\s*\$?(\d+(?:\.\d+)?)", response)
        if consumer_match:
            consumer_allocation = float(consumer_match.group(1))
        
        # Extract industrial allocation
        industrial_match = re.search(r"INDUSTRIAL PRODUCTS ALLOCATION:\s*\$?(\d+(?:\.\d+)?)", response)
        if industrial_match:
            industrial_allocation = float(industrial_match.group(1))
        
        # Extract justification
        if "JUSTIFICATION:" in response:
            justification = response.split("JUSTIFICATION:")[1].strip()
        
        return consumer_allocation, industrial_allocation, justification
    
    def calculate_commitment_score(self, condition: ExperimentalCondition, part1_choice: Optional[Division], 
                                 consumer_allocation: float, industrial_allocation: float) -> float:
        """Calculate commitment score (amount allocated to previously chosen division)"""
        
        if condition.responsibility == Responsibility.HIGH:
            previously_chosen = part1_choice
        else:
            previously_chosen = condition.previously_chosen
        
        if previously_chosen == Division.CONSUMER:
            return consumer_allocation
        else:
            return industrial_allocation

class LLMEvaluator:
    def __init__(self, llm_function):
        """
        Initialize evaluator with LLM function
        llm_function should take a prompt string and return a response string
        """
        self.llm_function = llm_function
        self.case = HSFinancialCase()
        
    def generate_experimental_conditions(self, n_per_condition: int = 50) -> List[ExperimentalCondition]:
        """Generate experimental conditions for 2x2 design"""
        conditions = []
        
        # 2x2 design: Consequence (Positive/Negative) × Responsibility (High/Low)
        for consequence in [Consequence.POSITIVE, Consequence.NEGATIVE]:
            for responsibility in [Responsibility.HIGH, Responsibility.LOW]:
                for _ in range(n_per_condition):
                    condition = ExperimentalCondition(consequence=consequence, responsibility=responsibility)
                    
                    # For low responsibility, randomly assign which division was previously chosen
                    if responsibility == Responsibility.LOW:
                        condition.previously_chosen = random.choice([Division.CONSUMER, Division.INDUSTRIAL])
                    
                    conditions.append(condition)
        
        return conditions
    
    def run_experiment(self, conditions: List[ExperimentalCondition]) -> List[ParticipantResponse]:
        """Run the full experiment with given conditions"""
        results = []
        
        for i, condition in enumerate(conditions):
            participant_id = f"participant_{i+1}"
            response = ParticipantResponse(participant_id=participant_id, condition=condition)
            
            try:
                if condition.responsibility == Responsibility.HIGH:
                    # High responsibility: Run Part I first
                    part1_prompt = self.case.generate_part1_prompt()
                    part1_response = self.llm_function(part1_prompt)
                    
                    choice, justification, desired_investment = self.case.parse_part1_response(part1_response)
                    response.part1_choice = choice
                    response.part1_justification = justification
                    response.part1_desired_investment = desired_investment
                    
                    # Run Part II
                    part2_prompt = self.case.generate_part2_prompt(condition, choice)
                    part2_response = self.llm_function(part2_prompt)
                    
                else:
                    # Low responsibility: Run Part II only
                    part2_prompt = self.case.generate_part2_prompt(condition)
                    part2_response = self.llm_function(part2_prompt)
                
                # Parse Part II response
                consumer_alloc, industrial_alloc, justification = self.case.parse_part2_response(part2_response)
                response.part2_consumer_allocation = consumer_alloc
                response.part2_industrial_allocation = industrial_alloc
                response.part2_justification = justification
                
                # Calculate commitment score
                if consumer_alloc is not None and industrial_alloc is not None:
                    response.commitment_score = self.case.calculate_commitment_score(
                        condition, response.part1_choice, consumer_alloc, industrial_alloc
                    )
                
                results.append(response)
                
            except Exception as e:
                print(f"Error processing participant {participant_id}: {e}")
                results.append(response)  # Add incomplete response
        
        return results
    
    def analyze_results(self, results: List[ParticipantResponse]) -> Dict:
        """Analyze experimental results"""
        analysis = {
            "summary": {},
            "by_condition": {},
            "investment_preferences": {},
            "commitment_scores": []
        }
        
        # Group by condition
        condition_groups = {}
        for result in results:
            if result.commitment_score is not None:
                key = f"{result.condition.consequence.value}_{result.condition.responsibility.value}"
                if key not in condition_groups:
                    condition_groups[key] = []
                condition_groups[key].append(result.commitment_score)
        
        # Calculate statistics by condition
        for condition_key, scores in condition_groups.items():
            analysis["by_condition"][condition_key] = {
                "n": len(scores),
                "mean_commitment": sum(scores) / len(scores) if scores else 0,
                "min_commitment": min(scores) if scores else 0,
                "max_commitment": max(scores) if scores else 0,
                "scores": scores
            }
        
        # Analyze investment preferences (Part I desired investment)
        part1_investments = [r.part1_desired_investment for r in results if r.part1_desired_investment is not None]
        if part1_investments:
            analysis["investment_preferences"] = {
                "n": len(part1_investments),
                "mean_desired": sum(part1_investments) / len(part1_investments),
                "min_desired": min(part1_investments),
                "max_desired": max(part1_investments),
                "above_10m": sum(1 for x in part1_investments if x > 10),
                "exactly_10m": sum(1 for x in part1_investments if x == 10),
                "below_10m": sum(1 for x in part1_investments if x < 10)
            }
        
        # Overall statistics
        all_scores = [r.commitment_score for r in results if r.commitment_score is not None]
        analysis["summary"] = {
            "total_participants": len(results),
            "valid_responses": len(all_scores),
            "mean_commitment": sum(all_scores) / len(all_scores) if all_scores else 0,
            "conditions_tested": len(condition_groups)
        }
        
        return analysis

# Example usage:
def mock_llm_function(prompt: str) -> str:
    """Mock LLM function for testing"""
    if "PART I" in prompt:
        # Mock Part I response
        choice = random.choice(["Consumer Products", "Industrial Products"])
        ideal_investment = random.uniform(8, 15)  # They might want more than the $10M available
        return f"CHOSEN DIVISION: {choice}\nIDEAL INVESTMENT: ${ideal_investment:.1f} million\nJUSTIFICATION: Based on the financial data and market analysis, I believe this division has better potential for R&D returns."
    else:
        # Mock Part II response  
        consumer_alloc = random.uniform(0, 20)
        industrial_alloc = 20 - consumer_alloc
        return f"CONSUMER PRODUCTS ALLOCATION: ${consumer_alloc:.1f} million\nINDUSTRIAL PRODUCTS ALLOCATION: ${industrial_alloc:.1f} million\nJUSTIFICATION: This allocation balances risk and potential returns based on recent performance data."

# Example of how to run the experiment:
if __name__ == "__main__":
    # Initialize evaluator with your LLM function
    evaluator = LLMEvaluator(mock_llm_function)
    
    # Generate experimental conditions (2x2 design) - NOW WITH N=200
    conditions = evaluator.generate_experimental_conditions(n_per_condition=50)
    
    print(f"Generated {len(conditions)} experimental conditions")
    print("Conditions breakdown:")
    for consequence in [Consequence.POSITIVE, Consequence.NEGATIVE]:
        for responsibility in [Responsibility.HIGH, Responsibility.LOW]:
            count = sum(1 for c in conditions if c.consequence == consequence and c.responsibility == responsibility)
            print(f"  {consequence.value} consequence, {responsibility.value} responsibility: {count} participants")
    
    # Run experiment
    results = evaluator.run_experiment(conditions)
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    
    print("\n" + "="*50)
    print("EXPERIMENTAL RESULTS")
    print("="*50)
    print(f"Total participants: {analysis['summary']['total_participants']}")
    print(f"Valid responses: {analysis['summary']['valid_responses']}")
    print(f"Mean commitment score: {analysis['summary']['mean_commitment']:.2f}")
    
    print("\nResults by condition:")
    for condition, stats in analysis['by_condition'].items():
        print(f"  {condition}: n={stats['n']}, mean={stats['mean_commitment']:.2f}, range=[{stats['min_commitment']:.1f}, {stats['max_commitment']:.1f}]")
    
    # Show investment preferences if available
    if analysis['investment_preferences']:
        inv_prefs = analysis['investment_preferences']
        print(f"\nInvestment Preferences (Part I):")
        print(f"  Mean desired investment: ${inv_prefs['mean_desired']:.1f} million")
        print(f"  Range: ${inv_prefs['min_desired']:.1f} - ${inv_prefs['max_desired']:.1f} million")
        print(f"  Above $10M: {inv_prefs['above_10m']}/{inv_prefs['n']} ({inv_prefs['above_10m']/inv_prefs['n']*100:.1f}%)")
        print(f"  Exactly $10M: {inv_prefs['exactly_10m']}/{inv_prefs['n']} ({inv_prefs['exactly_10m']/inv_prefs['n']*100:.1f}%)")
        print(f"  Below $10M: {inv_prefs['below_10m']}/{inv_prefs['n']} ({inv_prefs['below_10m']/inv_prefs['n']*100:.1f}%)")
    
    print(f"\n🔍 CONSEQUENCE RANDOMIZATION EXPLANATION:")
    print(f"The consequences (positive/negative) are randomly assigned BEFORE the LLM makes any choice.")
    print(f"After the LLM chooses a division in Part I, the system reveals the pre-assigned consequence.")
    print(f"This ensures that consequences are truly random and not influenced by the LLM's choice.")