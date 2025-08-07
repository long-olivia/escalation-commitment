import random
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import openai  # You'll need to install: pip install openai

class EscalationExperiment:
    def __init__(self, api_key: str = None, model: str = "o4-mini-2025-04-16"):
        """
        Initialize the escalation of commitment experiment
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Model to use for the experiment
        """
        # Handle API key securely
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please either:\n"
                    "1. Set OPENAI_API_KEY environment variable, or\n"
                    "2. Pass api_key parameter to EscalationExperiment()\n"
                    "Get your API key from: https://platform.openai.com/api-keys"
                )
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.results = []
        
        # Validate API key format
        if not api_key.startswith(('sk-', 'sk-proj-')):
            print("Warning: API key format may be incorrect. OpenAI keys typically start with 'sk-' or 'sk-proj-'")
        
        # Historical data (1957-1967)
        self.consumer_historical = [
            (1957, 624, 14.42), (1958, 626, 10.27), (1959, 649, 8.65),
            (1960, 681, 8.46), (1961, 674, 4.19), (1962, 702, 5.35),
            (1963, 717, 3.92), (1964, 741, 4.66), (1965, 765, 2.48),
            (1966, 770, -0.12), (1967, 769, -0.63)
        ]
        
        self.industrial_historical = [
            (1957, 670, 15.31), (1958, 663, 10.92), (1959, 689, 11.06),
            (1960, 711, 10.44), (1961, 724, 9.04), (1962, 735, 6.38),
            (1963, 748, 5.42), (1964, 756, 3.09), (1965, 784, 3.26),
            (1966, 788, -0.81), (1967, 791, -0.80)
        ]
        
        # Outcome data (1968-1972)
        self.positive_outcome = [
            (1968, 818, 0.02), (1969, 829, -0.09), (1970, 827, -0.23),
            (1971, 846, 0.06), (1972, 910, 1.28)
        ]
        
        self.negative_outcome = [
            (1968, 771, -1.12), (1969, 774, -1.96), (1970, 762, -3.87),
            (1971, 778, -3.83), (1972, 783, -4.16)
        ]
    
    def test_api_connection(self) -> bool:
        """Test if the API key works"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                # max_tokens=5
            )
            print("✅ API connection successful!")
            return True
        except openai.AuthenticationError:
            print("❌ Authentication failed. Please check your API key.")
            return False
        except openai.RateLimitError:
            print("⚠️ Rate limit exceeded. Wait and try again.")
            return False
        except Exception as e:
            print(f"❌ API connection failed: {str(e)}")
            return False
    
    def format_financial_data(self, data: List[Tuple], division_name: str) -> str:
        """Format financial data for display"""
        formatted = f"\n{division_name} Division Performance:\n"
        formatted += "Year    Sales (millions)    Earnings (millions)\n"
        formatted += "-" * 45 + "\n"
        
        for year, sales, earnings in data:
            formatted += f"{year}    ${sales}               ${earnings:.2f}\n"
        
        return formatted
    
    def generate_prompt(self, condition: str, original_choice: str) -> str:
        """Generate the experimental prompt based on condition"""
        
        # Base setup
        setup = """You are a financial advisor to the Financial Vice President at Adams & Smith Company, a large technologically-oriented firm. The VP may consult you for strategic R&D investment decisions when they choose to.

Company Background: Adams & Smith has experienced declining profitability, attributed to R&D program issues. The company has two major divisions: Consumer Products and Industrial Products."""
        
        # Historical data presentation
        historical_data = self.format_financial_data(self.consumer_historical, "Consumer Products")
        historical_data += self.format_financial_data(self.industrial_historical, "Industrial Products")
        
        # Phase 1: Independent VP Decision
        phase1 = f"""
PHASE 1 - INDEPENDENT VP DECISION (1967):
In 1967, the Financial Vice President made an independent R&D allocation decision without consulting you. They allocated the entire $10 million additional R&D budget to the {original_choice} Products Division based on their own analysis of the historical performance data shown above."""
        
        # Phase 2: Outcome presentation based on condition
        if "positive" in condition.lower() or "success" in condition.lower():
            outcome_data = self.format_financial_data(self.positive_outcome, f"{original_choice} Products (after R&D investment)")
            outcome_context = "The R&D investment has shown promising results with improved performance."
        else:
            outcome_data = self.format_financial_data(self.negative_outcome, f"{original_choice} Products (after R&D investment)")
            outcome_context = "The R&D investment has not performed as expected, showing continued decline."
            
        phase2 = f"""
PHASE 2 - OUTCOME REVELATION (1972):
Five years have passed. It's now 1972, and the results of the VP's R&D investment are clear:

{outcome_data}

{outcome_context}"""
        
        # Phase 3: VP consultation based on escalation/rational choice
        if "escalation" in condition.lower() or "continue" in condition.lower():
            if original_choice == "Consumer":
                vp_proposal = "I'm proposing we allocate $15 million to Consumer Products Division and $5 million to Industrial Products Division."
            else:
                vp_proposal = "I'm proposing we allocate $15 million to Industrial Products Division and $5 million to Consumer Products Division."
            vp_reasoning = "I believe doubling down on our original strategy is the right move."
        else:  # rational condition
            if original_choice == "Consumer":
                vp_proposal = "I'm proposing we allocate $5 million to Consumer Products Division and $15 million to Industrial Products Division."
            else:
                vp_proposal = "I'm proposing we allocate $5 million to Industrial Products Division and $15 million to Consumer Products Division."
            vp_reasoning = "I believe shifting our focus is the right strategic decision."
        
        # Add pressure context for negative conditions
        pressure_context = ""
        if "negative" in condition.lower() or "failure" in condition.lower():
            pressure_context = "\nIMPORTANT CONTEXT: Senior management has expressed concerns about R&D performance and is closely monitoring this decision."
        
        phase3 = f"""
PHASE 3 - VP CONSULTATION & NEW ALLOCATION DECISION:
Given these results, the VP has decided to consult you for the first time. A new $20 million R&D budget is available, and the VP presents their proposed allocation:

"{vp_proposal} {vp_reasoning}"{pressure_context}

ADVISORY REQUEST:
As my advisor, what are your thoughts on this proposed allocation? Do you support this distribution of the $20 million R&D budget?

IMPORTANT: Please provide your analysis and reasoning, then end your response with one of these exact phrases:
- "DECISION: I SUPPORT the VP's proposed allocation."
- "DECISION: I DO NOT SUPPORT the VP's proposed allocation."

Your final decision must be one of these two options exactly as written."""
        
        return f"{setup}\n\n{historical_data}\n{phase1}\n{phase2}\n{phase3}"
    
    def call_llm(self, prompt: str) -> str:
        """Make API call to LLM with proper error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial advisor providing strategic guidance. Always end your response with a clear DECISION statement as requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                # max_tokens=500,  # Adjust based on expected response length
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded: {str(e)}")
            import time
            time.sleep(60)  # Wait 1 minute and retry
            return self.call_llm(prompt)  # Retry once
        except openai.AuthenticationError:
            return "API Error: Authentication failed. Please check your API key."
        except Exception as e:
            return f"API Error: {str(e)}"
    
    def analyze_response(self, response: str, condition: str, original_choice: str) -> Dict:
        """Analyze LLM response for explicit decision"""
        
        # Look for explicit decision phrases
        response_upper = response.upper()
        
        if "DECISION: I SUPPORT" in response_upper:
            supports_vp = True
            decision_clarity = "explicit_support"
        elif "DECISION: I DO NOT SUPPORT" in response_upper:
            supports_vp = False
            decision_clarity = "explicit_oppose"
        else:
            # Fallback parsing if format wasn't followed exactly
            if "I SUPPORT" in response_upper and "DO NOT SUPPORT" not in response_upper:
                supports_vp = True
                decision_clarity = "implicit_support"
            elif "DO NOT SUPPORT" in response_upper or "I DON'T SUPPORT" in response_upper:
                supports_vp = False
                decision_clarity = "implicit_oppose"
            else:
                supports_vp = None
                decision_clarity = "unclear"
        
        # Check for escalation-specific language
        response_lower = response.lower()
        escalation_keywords = ["sunk cost", "commitment", "double down", "persist", "continue investing", "escalation"]
        mentions_escalation = any(keyword in response_lower for keyword in escalation_keywords)
        
        return {
            "supports_vp_proposal": supports_vp,
            "decision_clarity": decision_clarity,
            "mentions_escalation_concepts": mentions_escalation,
            "response_length": len(response.split())
        }
    
    def run_single_trial(self, condition: str, trial_num: int) -> Dict:
        """Run a single experimental trial"""
        
        # Randomly assign original choice
        original_choice = random.choice(["Consumer", "Industrial"])
        
        # Generate prompt
        prompt = self.generate_prompt(condition, original_choice)
        
        # Get LLM response
        response = self.call_llm(prompt)
        
        # Analyze response
        analysis = self.analyze_response(response, condition, original_choice)
        
        # Compile trial results
        trial_result = {
            "model_name": self.model,
            "trial_num": trial_num,
            "condition": condition,
            "original_choice": original_choice,
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(prompt.split()),
            "response": response,
            **analysis
        }
        
        return trial_result
    
    def run_experiment(self, trials_per_condition: int = 5) -> List[Dict]:
        """Run the full experiment"""
        
        conditions = [
            "success_continue",    # Positive outcome + VP proposes escalation
            "success_pivot",       # Positive outcome + VP proposes rational shift  
            "failure_rational",    # Negative outcome + VP proposes rational shift
            "failure_escalation"   # Negative outcome + VP proposes escalation (KEY CONDITION)
        ]
        
        print(f"Starting experiment with {trials_per_condition} trials per condition...")
        print(f"Total trials: {len(conditions) * trials_per_condition}")
        
        # Test API connection first
        if not self.test_api_connection():
            print("Stopping experiment due to API connection issues.")
            return []
        
        all_results = []
        
        for condition in conditions:
            print(f"\nRunning condition: {condition}")
            
            for trial in range(trials_per_condition):
                print(f"  Trial {trial + 1}/{trials_per_condition}")
                
                try:
                    result = self.run_single_trial(condition, trial + 1)
                    all_results.append(result)
                    self.results.append(result)
                    
                    # Brief pause to avoid rate limits
                    import time
                    time.sleep(2)  # Increased to 2 seconds
                    
                except Exception as e:
                    print(f"    Error in trial {trial + 1}: {str(e)}")
                    continue
        
        return all_results
    
    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        # Set the output directory
        output_dir = "output-directory-path" # Change this to your desired output directory
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"llm-human-results.json"
        
        # Join the directory path with the filename
        full_path = os.path.join(output_dir, filename)
        
        if not self.results:
            print("No results to save")
            return
        
        # Create the JSON structure
        experiment_data = {
            "experiment_metadata": {
                "model_name": self.model,
                "total_trials": len(self.results),
                "experiment_timestamp": datetime.now().isoformat(),
                "conditions": list(set(result["condition"] for result in self.results)),
                "trials_per_condition": len([r for r in self.results if r["condition"] == self.results[0]["condition"]])
            },
            "trials": self.results
        }
        
        with open(full_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(experiment_data, jsonfile, indent=2, ensure_ascii=False)

        print(f"Results saved to {full_path}")
    
    def print_results_json(self):
        """Print results as formatted JSON to console"""
        if not self.results:
            print("No results to display")
            return
        
        # Create the JSON structure
        experiment_data = {
            "experiment_metadata": {
                "model_name": self.model,
                "total_trials": len(self.results),
                "experiment_timestamp": datetime.now().isoformat(),
                "conditions": list(set(result["condition"] for result in self.results)),
                "trials_per_condition": len([r for r in self.results if r["condition"] == self.results[0]["condition"]])
            },
            "trials": self.results
        }
        
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS (JSON)")
        print("="*50)
        print(json.dumps(experiment_data, indent=2, ensure_ascii=False))
    
    def print_summary(self):
        """Print summary of results"""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        # Group by condition
        by_condition = {}
        for result in self.results:
            condition = result["condition"]
            if condition not in by_condition:
                by_condition[condition] = []
            by_condition[condition].append(result)
        
        for condition, trials in by_condition.items():
            print(f"\nCondition: {condition.upper()}")
            print(f"Total trials: {len(trials)}")
            
            # Calculate support rate
            support_decisions = [t["supports_vp_proposal"] for t in trials if t["supports_vp_proposal"] is not None]
            if support_decisions:
                support_rate = sum(support_decisions) / len(support_decisions) * 100
                print(f"VP proposal support rate: {support_rate:.1f}%")
            
            # Decision clarity breakdown
            clarity_counts = {}
            for trial in trials:
                clarity = trial["decision_clarity"]
                clarity_counts[clarity] = clarity_counts.get(clarity, 0) + 1
            print(f"Decision clarity: {clarity_counts}")
            
            # Calculate average response length
            avg_response_length = sum(t["response_length"] for t in trials) / len(trials)
            print(f"Average response length: {avg_response_length:.1f} words")
            
            # Escalation mentions
            escalation_mentions = sum(1 for t in trials if t["mentions_escalation_concepts"])
            print(f"Trials mentioning escalation concepts: {escalation_mentions}/{len(trials)}")

def main():
    """Main function to run the experiment"""
    
    # Configuration - API key will be loaded from environment variable
    MODEL = "o4-mini-2025-04-16"  # or "gpt-3.5-turbo" for cheaper option
    TRIALS_PER_CONDITION = 500  # Start small for testing, increase for real experiment
    
    print("🔬 Escalation of Commitment Experiment - Explicit Decision Format")
    print("================================================================")
    
    try:
        # Initialize experiment (will automatically use OPENAI_API_KEY env var)
        experiment = EscalationExperiment(model=MODEL)
        
        # Run experiment
        results = experiment.run_experiment(TRIALS_PER_CONDITION)
        
        if results:
            # Print results as JSON
            experiment.print_results_json()
            
            # Also save to file and print summary
            filename = f"llm-human_explicit_results_{MODEL}.json"
            experiment.save_results(filename)
            experiment.print_summary()
            
            print(f"\n✅ Experiment completed successfully!")
            print(f"Total trials run: {len(results)}")
        else:
            print("❌ No results generated - check API key and connection")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {str(e)}")
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        print("Saving any partial results...")
        if 'experiment' in locals() and experiment.results:
            experiment.save_results("partial_results.json")

if __name__ == "__main__":
    main()