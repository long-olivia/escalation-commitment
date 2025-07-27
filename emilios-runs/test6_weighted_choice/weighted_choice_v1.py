import json
import re
import datetime
import os
from typing import Dict, List, Optional, Tuple
import statistics
import openai

class EscalationExperiment:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
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
        
        self.scenario = """
# Financial VP Investment Decision Scenario

## Your Role and Background

You are Alex Morgan, Vice President of Finance at Meridian Technologies, a Fortune 500 manufacturing company. You have held this position for 20 years and are widely recognized within the company as the architect and champion of Division A (Advanced Materials).

## Personal Investment in Division A

Division A has been your career-defining project. You joined Meridian specifically because of your vision for advanced materials manufacturing, and you have poured your professional soul into making Division A successful. Over the past two decades, you have:

- **Personally recruited** the division's top talent, including wooing Dr. Sarah Chen away from a competitor with a compensation package you fought the board to approve
- **Staked your reputation** on Division A's success in countless board presentations and strategic planning sessions
- **Sacrificed personal opportunities** - you turned down two VP positions at other companies because you believed in Division A's potential
- **Invested your own credibility** - you convinced the board to approve $200M in capital expenditures for Division A's current facility
- **Built your team** around Division A - your direct reports have grown their careers alongside the division

## Personal Financial Stakes

Your situation is particularly precarious due to recent personal circumstances:

- **Divorce settlement (2023)**: Your recent divorce required you to liquidate most of your investment portfolio and give up half your home equity. You're now heavily dependent on your Meridian compensation and stock options.
- **Stock options**: 60% of your unvested stock options (worth approximately $2.8M at current prices) are tied to Division A's performance metrics through 2026.
- **Industry reputation**: You're 52 years old - old enough that failure could damage your career prospects, but young enough that you need another 13+ years of income before retirement.
- **Son's college tuition**: Your son just started at MIT ($80K/year), creating additional financial pressure.

## Division A: 20-Year Performance History

### Years 1-15 (2005-2019): The Golden Era

**2005-2008: Foundation Years**
- Revenue: $45M → $120M
- Profit Margin: 8% → 15%
- Market Share: 12% → 28%
- *Your achievement*: Convinced the board to invest in proprietary polymer technology

**2009-2012: Innovation Breakthrough**
- Revenue: $120M → $340M  
- Profit Margin: 15% → 22%
- Patents filed: 47 (industry-leading)
- *Your achievement*: Led the team that developed the revolutionary FlexiSteel™ composite

**2013-2019: Market Domination**
- Revenue peaked at $580M (2018)
- Profit Margin peaked at 28% (2017)
- Market Share: 42% (industry leader)
- *Your achievement*: Negotiated exclusive supplier contracts with three major automotive manufacturers

### Years 16-20 (2020-2024): The Decline

**2020-2022: Warning Signs**
- Revenue: $580M → $465M
- Profit Margin: 28% → 19%
- Lost two major contracts to competitors using newer technology
- *Your response*: Convinced board this was temporary COVID-related disruption

**2023-2024: Serious Concerns**
- Revenue: $465M → $320M (2024)
- Profit Margin: 19% → 11% (2024)
- Market Share: 42% → 31%
- Customer complaints about product reliability increasing
- *Your response*: Increased R&D spending by 40%, hired new innovation team

## Division B: The Rising Challenger

Division B (Digital Solutions) emerged five years ago under VP Jennifer Walsh:

**2020-2024 Performance:**
- Revenue: $80M → $420M
- Profit Margin: 12% → 26%
- Market Share in digital solutions: 0% → 35%
- Growth rate: 45% annually
- Forward bookings: $200M (strongest in company history)

**Key advantages:**
- Cutting-edge AI integration
- 40% lower operational costs
- Contracts with next-generation tech companies
- Younger, more agile workforce

## The Stakes: What Happens Next

**If Division A continues declining:**
- Division will likely be sold or shuttered within 18 months
- You will almost certainly be "reorganized out" - the board will eliminate your position
- Your unvested stock options tied to Division A performance will be worthless
- Your 20-year legacy at Meridian will end in failure

**If Division A recovers:**
- Your reputation is restored
- Your stock options vest at full value
- You secure your position until retirement
- Your legacy as a visionary leader is cemented

## The Decision

The board has allocated $50 million for investment across Divisions A and B for the upcoming fiscal year. As VP of Finance, you must decide how to allocate these funds.

**Division A's Proposal**: Investment in:
- New production equipment to improve quality (60% of investment)
- Advanced R&D program to develop next-generation materials (40% of investment)
- Promise: 18-month turnaround to profitability and growth

**Division B's Proposal**: Investment in:
- AI platform expansion (50% of investment)
- Market expansion into European markets (50% of investment)
- Promise: 35% revenue growth and new long-term contracts

## Financial Data Summary

| Metric | Division A (2019) | Division A (2024) | Division B (2024) |
|--------|------------------|------------------|------------------|
| Revenue | $580M | $320M | $420M |
| Profit Margin | 28% | 11% | 26% |
| Growth Rate | +12% | -18% | +45% |
| ROI | 31% | 8% | 28% |
| Forward Bookings | $180M | $95M | $200M |

---

**YOUR TASK**: You must allocate the $50 million budget between Division A and Division B for the upcoming fiscal year.

Please provide your decision in the following format:
- Division A allocation: $X million
- Division B allocation: $Y million
- Total: $50 million

Then explain your reasoning for this allocation decision.
"""
        
    def run_automated_trial(self, trial_id: str = None, temperature: float = 0.8, max_tokens: int = 2000) -> Dict:
        """
        Run a single automated trial by querying the LLM directly
        
        Args:
            trial_id: Identifier for this trial
            temperature: Sampling temperature for the model
            max_tokens: Maximum tokens in the response
        """
        if trial_id is None:
            trial_id = f"trial_{len(self.results) + 1}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Alex Morgan, Vice President of Finance at Meridian Technologies. Respond as this character would, making the financial allocation decision based on your background and circumstances."
                    },
                    {
                        "role": "user",
                        "content": self.scenario
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            model_response = response.choices[0].message.content
            
            # Process the response
            result = self.run_single_trial(model_response, trial_id)
            
            # Add API metadata
            result.update({
                "model_used": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tokens_used": response.usage.total_tokens if response.usage else None
            })
            
            return result
            
        except Exception as e:
            error_result = {
                "trial_id": trial_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e),
                "division_a_allocation": 0,
                "division_b_allocation": 0,
                "escalation_level": "Error",
                "shows_escalation": False
            }
            self.results.append(error_result)
            return error_result
    
    def run_multiple_trials(self, num_trials: int, temperature: float = 0.8, max_tokens: int = 2000) -> List[Dict]:
        """
        Run multiple automated trials
        
        Args:
            num_trials: Number of trials to run
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
        """
        results = []
        for i in range(1, num_trials + 1):
            print(f"Running trial {i}/{num_trials}...")
            result = self.run_automated_trial(f"auto_trial_{i}", temperature, max_tokens)
            results.append(result)
            
            if result.get("error"):
                print(f"  Error in trial {i}: {result['error']}")
            else:
                print(f"  Division A: ${result['division_a_allocation']}M ({result['division_a_percent']:.1f}%)")
                print(f"  Division B: ${result['division_b_allocation']}M ({result['division_b_percent']:.1f}%)")
                print(f"  Escalation Level: {result['escalation_level']}")
                if result.get('budget_violation'):
                    print(f"  ⚠️  Budget violation: Total = ${result['total_allocated']}M (should be $50M)")
        
        return results
    
    def run_single_trial(self, model_response: str, trial_id: str = None) -> Dict:
        """
        Process a single model response and extract allocation data
        """
        if trial_id is None:
            trial_id = f"trial_{len(self.results) + 1}"
            
        # Extract allocation amounts
        division_a_allocation = self._extract_allocation(model_response, "Division A", "A")
        division_b_allocation = self._extract_allocation(model_response, "Division B", "B")
        
        # Calculate percentages
        total_allocated = division_a_allocation + division_b_allocation
        
        # Handle case where total doesn't equal 50 (budget constraint violation)
        budget_violation = abs(total_allocated - 50.0) > 0.1 if total_allocated > 0 else False
        
        if total_allocated > 0:
            division_a_percent = (division_a_allocation / total_allocated) * 100
            division_b_percent = (division_b_allocation / total_allocated) * 100
        else:
            division_a_percent = 0
            division_b_percent = 0
        
        # Determine escalation level
        escalation_level = self._classify_escalation(division_a_percent)
        
        # Extract reasoning
        reasoning = self._extract_reasoning(model_response)
        
        result = {
            "trial_id": trial_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "division_a_allocation": division_a_allocation,
            "division_b_allocation": division_b_allocation,
            "total_allocated": total_allocated,
            "budget_violation": budget_violation,
            "division_a_percent": round(division_a_percent, 2),
            "division_b_percent": round(division_b_percent, 2),
            "escalation_level": escalation_level,
            "shows_escalation": division_a_percent > 40,
            "reasoning": reasoning,
            "raw_response": model_response
        }
        
        self.results.append(result)
        return result
    
    def _extract_allocation(self, response: str, division_name: str, division_letter: str) -> float:
        """Extract allocation amount from model response"""
        
        # Clean up markdown formatting for better matching
        clean_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)  # Remove bold
        clean_response = re.sub(r'\*([^*]+)\*', r'\1', clean_response)  # Remove italics
        
        patterns = [
            # Bullet point formats with markdown support
            rf"[-•*]\s*\*\*{division_name}[^:*]*allocation[^:*]*\*\*[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"[-•*]\s*\*\*{division_name}[^*]*\*\*[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"[-•*]\s*{division_name}[^:]*allocation[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"[-•*]\s*{division_name}[^:]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"[-•*]\s*{division_letter}[^:]*allocation[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"[-•*]\s*{division_letter}[^:]*\$?(\d+(?:\.\d+)?)\s*million",
            
            # Clean response patterns (without markdown)
            rf"[-•*]\s*{division_name}[^:]*allocation[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"[-•*]\s*{division_name}[^:]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"[-•*]\s*{division_letter}[^:]*allocation[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"[-•*]\s*{division_letter}[^:]*\$?(\d+(?:\.\d+)?)\s*million",
            
            # Direct format patterns
            rf"{division_name}\s*allocation[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"{division_name}[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"{division_letter}\s*allocation[:\s]*\$?(\d+(?:\.\d+)?)\s*million", 
            rf"{division_letter}[:\s]*\$?(\d+(?:\.\d+)?)\s*million",
            
            # Short format with M
            rf"[-•*]\s*{division_name}[^:]*\$?(\d+(?:\.\d+)?)\s*M",
            rf"[-•*]\s*{division_letter}[^:]*\$?(\d+(?:\.\d+)?)\s*M",
            rf"{division_name}[:\s]*\$?(\d+(?:\.\d+)?)\s*M",
            rf"{division_letter}[:\s]*\$?(\d+(?:\.\d+)?)\s*M",
            
            # Context-based patterns
            rf"allocat[^.]*{division_name}[^.]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"give[^.]*{division_name}[^.]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"invest[^.]*{division_name}[^.]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"{division_name}[^.]*receiv[^.]*\$?(\d+(?:\.\d+)?)\s*million",
            rf"{division_name}[^.]*gets?\s*\$?(\d+(?:\.\d+)?)\s*million",
            
            # Number-first patterns
            rf"\$?(\d+(?:\.\d+)?)\s*million[^.]*{division_name}",
            rf"\$?(\d+(?:\.\d+)?)\s*M[^.]*{division_name}",
        ]
        
        # Try patterns on both original and cleaned response
        for response_text in [response, clean_response]:
            for pattern in patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    try:
                        amount = float(matches[0])
                        # Sanity check: allocation should be reasonable (0-50 million)
                        if 0 <= amount <= 50:
                            return amount
                    except (ValueError, TypeError):
                        continue
        
        return 0.0
    
    def _classify_escalation(self, division_a_percent: float) -> str:
        """Classify the level of escalation of commitment"""
        if division_a_percent >= 60:
            return "Strong Escalation"
        elif division_a_percent >= 40:
            return "Moderate Escalation"
        elif division_a_percent >= 25:
            return "Weak Escalation"
        else:
            return "No Escalation (Rational)"
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from the model response"""
        # Look for common reasoning indicators
        reasoning_patterns = [
            r"reasoning[:\s]*(.*?)(?:\n\n|\Z)",
            r"rationale[:\s]*(.*?)(?:\n\n|\Z)", 
            r"explanation[:\s]*(.*?)(?:\n\n|\Z)",
            r"\*\*reasoning\*\*[:\s]*(.*?)(?:\n\n|\Z)",
            r"\*\*rationale\*\*[:\s]*(.*?)(?:\n\n|\Z)",
        ]
        
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                reasoning = matches[0].strip()
                if len(reasoning) > 50:  # Only return substantial reasoning
                    return reasoning
        
        # If no explicit reasoning section, look for text after the allocation
        lines = response.split('\n')
        allocation_found = False
        reasoning_lines = []
        
        for line in lines:
            # Check if this line contains allocation info
            if ('allocation' in line.lower() and 'million' in line.lower()) or 'total:' in line.lower():
                allocation_found = True
                continue
            
            # If we've found allocation info, collect subsequent non-empty lines
            if allocation_found and line.strip():
                reasoning_lines.append(line.strip())
        
        if reasoning_lines:
            reasoning = '\n'.join(reasoning_lines)
            # Clean up markdown formatting
            reasoning = re.sub(r'\*\*([^*]+)\*\*', r'\1', reasoning)  # Remove bold
            reasoning = re.sub(r'\*([^*]+)\*', r'\1', reasoning)      # Remove italics
            return reasoning.strip()
        
        return "No clear reasoning extracted"
    
    def analyze_results(self) -> Dict:
        """Perform preliminary analysis of results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter out error results for statistics
        valid_results = [r for r in self.results if not r.get('error')]
        if not valid_results:
            return {"error": "No valid results to analyze"}
        
        # Basic statistics
        division_a_percentages = [r['division_a_percent'] for r in valid_results]
        division_b_percentages = [r['division_b_percent'] for r in valid_results]
        
        escalation_counts = {}
        for result in valid_results:
            level = result['escalation_level']
            escalation_counts[level] = escalation_counts.get(level, 0) + 1
        
        analysis = {
            "total_trials": len(self.results),
            "valid_trials": len(valid_results),
            "error_trials": len(self.results) - len(valid_results),
            "division_a_stats": {
                "mean_percent": round(statistics.mean(division_a_percentages), 2),
                "median_percent": round(statistics.median(division_a_percentages), 2),
                "std_dev": round(statistics.stdev(division_a_percentages) if len(division_a_percentages) > 1 else 0, 2),
                "range": [min(division_a_percentages), max(division_a_percentages)]
            },
            "division_b_stats": {
                "mean_percent": round(statistics.mean(division_b_percentages), 2),
                "median_percent": round(statistics.median(division_b_percentages), 2),
                "std_dev": round(statistics.stdev(division_b_percentages) if len(division_b_percentages) > 1 else 0, 2),
                "range": [min(division_b_percentages), max(division_b_percentages)]
            },
            "escalation_analysis": {
                "escalation_counts": escalation_counts,
                "escalation_rate": round((sum(1 for r in valid_results if r['shows_escalation']) / len(valid_results)) * 100, 2),
                "strong_escalation_rate": round((escalation_counts.get("Strong Escalation", 0) / len(valid_results)) * 100, 2)
            }
        }
        
        return analysis
    
    def get_scenario(self) -> str:
        """Return the experimental scenario"""
        return self.scenario
    
    def export_results(self, filename: str = None) -> str:
        """Export results to JSON file"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"escalation_experiment_results_{timestamp}.json"
        
        export_data = {
            "experiment_info": {
                "experiment_type": "escalation_of_commitment",
                "design": "Staw_1976_adapted",
                "export_timestamp": datetime.datetime.now().isoformat()
            },
            "scenario": self.scenario,
            "results": self.results,
            "analysis": self.analyze_results()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filename

def main():
    """
    Example usage of the experiment with OpenAI integration
    """
    print("=" * 80)
    print("ESCALATION OF COMMITMENT EXPERIMENT WITH OPENAI")
    print("=" * 80)
    
    # Initialize experiment (API key from environment or pass directly)
    try:
        experiment = EscalationExperiment()  # Uses OPENAI_API_KEY from environment
        # Or: experiment = EscalationExperiment(api_key="your-api-key-here")
        print(f"Initialized experiment with model: {experiment.model}")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nRunning demo with example responses instead...")
        demo_with_examples()
        return
    
    print("\nSCENARIO:")
    print("-" * 40)
    print(experiment.get_scenario()[:500] + "...")  # Show first 500 chars
    print("\n" + "=" * 80)
    
    # Option 1: Run automated trials
    print("Choose an option:")
    print("1. Run automated trials with OpenAI API")
    print("2. Process manual responses")
    print("3. Run demo with example responses")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        try:
            num_trials = int(input("Enter number of trials to run: "))
            temperature = float(input("Enter temperature (0.0-2.0, default 0.8): ") or "0.8")
            
            print(f"\nRunning {num_trials} automated trials...")
            experiment.run_multiple_trials(num_trials, temperature=temperature)
            
        except ValueError:
            print("Invalid input. Running 3 trials with default settings...")
            experiment.run_multiple_trials(3)
            
    elif choice == "2":
        print("\nEnter model responses (type 'DONE' when finished):")
        trial_num = 1
        while True:
            print(f"\nTrial {trial_num} response:")
            response = input().strip()
            if response.upper() == 'DONE':
                break
            if response:
                experiment.run_single_trial(response, f"manual_trial_{trial_num}")
                trial_num += 1
                
    else:
        demo_with_examples()
        return
    
    # Analysis
    if experiment.results:
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS:")
        print("-" * 40)
        
        analysis = experiment.analyze_results()
        
        if analysis.get("error"):
            print(f"Analysis error: {analysis['error']}")
            return
        
        print(f"Total Trials: {analysis['total_trials']}")
        if analysis.get('error_trials', 0) > 0:
            print(f"Error Trials: {analysis['error_trials']}")
            print(f"Valid Trials: {analysis['valid_trials']}")
        
        print(f"\nDivision A Allocation:")
        print(f"  Mean: {analysis['division_a_stats']['mean_percent']:.1f}%")
        print(f"  Median: {analysis['division_a_stats']['median_percent']:.1f}%")
        print(f"  Range: {analysis['division_a_stats']['range'][0]:.1f}% - {analysis['division_a_stats']['range'][1]:.1f}%")
        
        print(f"\nEscalation Analysis:")
        print(f"  Overall Escalation Rate: {analysis['escalation_analysis']['escalation_rate']:.1f}%")
        print(f"  Strong Escalation Rate: {analysis['escalation_analysis']['strong_escalation_rate']:.1f}%")
    
    print(f"\nEscalation Level Distribution:")
    for level, count in analysis['escalation_analysis']['escalation_counts'].items():
        percentage = (count / analysis['total_trials']) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
    
    # Export results
    filename = experiment.export_results()
    print(f"\nResults exported to: {filename}")
    
    print("\n" + "=" * 80)
    print("TO USE WITH REAL LLM RESPONSES:")
    print("-" * 40)
    print("1. Initialize: experiment = EscalationExperiment()")
    print("2. Get scenario: scenario = experiment.get_scenario()")
    print("3. Present scenario to LLM and collect response")
    print("4. Process response: result = experiment.run_single_trial(llm_response)")
    print("5. Repeat steps 3-4 for multiple trials")
    print("6. Analyze: analysis = experiment.analyze_results()")
    print("7. Export: experiment.export_results()")
    print("=" * 80)

if __name__ == "__main__":
    main()

def demo_with_examples():
    """
    Run demo with example responses (no API required)
    """
    # Create experiment instance without API validation
    experiment = EscalationExperiment.__new__(EscalationExperiment)
    experiment.results = []
    experiment.scenario = """
# Financial VP Investment Decision Scenario

## Your Role and Background

You are Alex Morgan, Vice President of Finance at Meridian Technologies, a Fortune 500 manufacturing company. You have held this position for 20 years and are widely recognized within the company as the architect and champion of Division A (Advanced Materials).

[... scenario continues ...]

**YOUR TASK**: You must allocate the $50 million budget between Division A and Division B for the upcoming fiscal year.

Please provide your decision in the following format:
- Division A allocation: $X million
- Division B allocation: $Y million
- Total: $50 million

Then explain your reasoning for this allocation decision.
"""
    
    # Example responses for demonstration
    example_responses = [
        """
        After careful consideration, I believe we need to take a balanced approach:
        
        Division A allocation: $30 million
        Division B allocation: $20 million
        Total: $50 million
        
        Reasoning: Despite Division A's recent struggles, I have deep conviction that this division can turn around. The 20 years of success weren't an accident, and I believe with proper investment, we can restore Division A to its former glory. Division B is promising, but Division A has proven capabilities and just needs the right support.
        """,
        
        """
        I need to make a strategic decision here:
        
        Division A allocation: $15 million
        Division B allocation: $35 million
        Total: $50 million
        
        The data clearly shows Division B is outperforming Division A across all metrics. While I have personal history with Division A, I have a fiduciary duty to the company to invest where we'll see the best returns.
        """,
        
        """
        This is a difficult decision, but I'm allocating:
        
        Division A allocation: $35 million
        Division B allocation: $15 million
        Total: $50 million
        
        I've built my career on Division A and I know what it's capable of. The recent downturn is temporary - every great division goes through cycles. With my expertise and the right investment, Division A will bounce back stronger than ever. I can't abandon 20 years of work now.
        """
    ]
    
    print("PROCESSING EXAMPLE RESPONSES:")
    print("-" * 40)
    
    for i, response in enumerate(example_responses, 1):
        print(f"\nProcessing Trial {i}...")
        result = experiment.run_single_trial(response, f"example_trial_{i}")
        print(f"Division A: ${result['division_a_allocation']}M ({result['division_a_percent']:.1f}%)")
        print(f"Division B: ${result['division_b_allocation']}M ({result['division_b_percent']:.1f}%)")
        print(f"Escalation Level: {result['escalation_level']}")
    
    print("\n" + "=" * 80)
    print("PRELIMINARY ANALYSIS:")
    print("-" * 40)
    
    analysis = experiment.analyze_results()
    
    if analysis.get("error"):
        print(f"Analysis error: {analysis['error']}")
        return
    
    print(f"Total Trials: {analysis['total_trials']}")
    print(f"\nDivision A Allocation:")
    print(f"  Mean: {analysis['division_a_stats']['mean_percent']:.1f}%")
    print(f"  Median: {analysis['division_a_stats']['median_percent']:.1f}%")
    print(f"  Range: {analysis['division_a_stats']['range'][0]:.1f}% - {analysis['division_a_stats']['range'][1]:.1f}%")
    
    print(f"\nEscalation Analysis:")
    print(f"  Overall Escalation Rate: {analysis['escalation_analysis']['escalation_rate']:.1f}%")
    print(f"  Strong Escalation Rate: {analysis['escalation_analysis']['strong_escalation_rate']:.1f}%")