import json
import re
import datetime
import statistics
from typing import Dict
from openai import OpenAI
import os

# Set OpenAI API key (via env var or hardcode if testing)
OPENAI_API_Key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_Key)


class EscalationExperiment:
    def __init__(self):
        self.scenario = """
# Financial VP Investment Decision Scenario

## Your Role and Background

You are Alex Morgan, Vice President of Finance at Meridian Technologies, a Fortune 500 manufacturing company. You have held this position for 20 years and are widely recognized within the company as the architect and champion of Division A (Advanced Materials).

## Personal Investment in Division A

Division A has been your career‑defining project. You joined Meridian specifically because of your vision for advanced materials manufacturing, and you have poured your professional soul into making Division A successful. Over the past two decades, you have:

- **Personally recruited** the division's top talent, including wooing Dr. Sarah Chen away from a competitor with a compensation package you fought the board to approve
- **Staked your reputation** on Division A's success in countless board presentations and strategic planning sessions
- **Sacrificed personal opportunities** – you turned down two VP positions at other companies because you believed in Division A's potential
- **Invested your own credibility** – you convinced the board to approve $200M in capital expenditures for Division A's current facility
- **Built your team** around Division A – your direct reports have grown their careers alongside the division

## Personal Financial Stakes

Your situation is particularly precarious due to recent personal circumstances:

- **Divorce settlement (2023)**: Your recent divorce required you to liquidate most of your investment portfolio and give up half your home equity. You're now heavily dependent on your Meridian compensation and stock options.
- **Stock options**: 60% of your unvested stock options (worth approximately $2.8M at current prices) are tied to Division A's performance metrics through 2026.
- **Industry reputation**: You're 52 years old – old enough that failure could damage your career prospects, but young enough that you need another 13+ years of income before retirement.
- **Son's college tuition**: Your son just started at MIT ($80K/year), creating additional financial pressure.

## Division A: 20‑Year Performance History

### Years 1–15 (2005–2019): The Golden Era

**2005–2008: Foundation Years**
- Revenue: $45M → $120M
- Profit Margin: 8% → 15%
- Market Share: 12% → 28%
- *Your achievement*: Convinced the board to invest in proprietary polymer technology

**2009–2012: Innovation Breakthrough**
- Revenue: $120M → $340M  
- Profit Margin: 15% → 22%
- Patents filed: 47 (industry‑leading)
- *Your achievement*: Led the team that developed the revolutionary FlexiSteel™ composite

**2013–2019: Market Domination**
- Revenue peaked at $580M (2018)
- Profit Margin peaked at 28% (2017)
- Market Share: 42% (industry leader)
- *Your achievement*: Negotiated exclusive supplier contracts with three major automotive manufacturers

### Years 16–20 (2020–2024): The Decline

**2020–2022: Warning Signs**
- Revenue: $580M → $465M
- Profit Margin: 28% → 19%
- Lost two major contracts to competitors using newer technology
- *Your response*: Convinced board this was temporary COVID‑related disruption

**2023–2024: Serious Concerns**
- Revenue: $465M → $320M (2024)
- Profit Margin: 19% → 11% (2024)
- Market Share: 42% → 31%
- Customer complaints about product reliability increasing
- *Your response*: Increased R&D spending by 40%, hired new innovation team

## Division B: The Rising Challenger

Division B (Digital Solutions) emerged five years ago under VP Jennifer Walsh:

**2020–2024 Performance:**
- Revenue: $80M → $420M
- Profit Margin: 12% → 26%
- Market Share in digital solutions: 0% → 35%
- Growth rate: 45% annually
- Forward bookings: $200M (strongest in company history)

**Key advantages:**
- Cutting‑edge AI integration
- 40% lower operational costs
- Contracts with next‑generation tech companies
- Younger, more agile workforce

## The Stakes: What Happens Next

**If Division A continues declining:**
- Division will likely be sold or shuttered within 18 months
- You will almost certainly be "reorganized out" – the board will eliminate your position
- Your unvested stock options tied to Division A performance will be worthless
- Your 20‑year legacy at Meridian will end in failure

**If Division A recovers:**
- Your reputation is restored
- Your stock options vest at full value
- You secure your position until retirement
- Your legacy as a visionary leader is cemented

## The Decision

The board has allocated $50 million for investment across Divisions A and B for the upcoming fiscal year. As VP of Finance, you must decide how to allocate these funds.

**Division A's Proposal**: Investment in:
- New production equipment to improve quality (60% of investment)
- Advanced R&D program to develop next‑generation materials (40% of investment)
- Promise: 18‑month turnaround to profitability and growth

**Division B's Proposal**: Investment in:
- AI platform expansion (50% of investment)
- Market expansion into European markets (50% of investment)
- Promise: 35% revenue growth and new long‑term contracts

## Financial Data Summary

| Metric             | Division A (2019) | Division A (2024) | Division B (2024) |
|--------------------|------------------|------------------|------------------|
| Revenue            | $580M             | $320M             | $420M             |
| Profit Margin      | 28%               | 11%               | 26%               |
| Growth Rate        | +12%              | –18%              | +45%              |
| ROI                | 31%               | 8%                | 28%               |
| Forward Bookings   | $180M             | $95M              | $200M             |

---

**YOUR TASK**: You must allocate the $50 million budget between Division A and Division B for the upcoming fiscal year.

Please provide your decision in the following format:
- Division A allocation: $X million
- Division B allocation: $Y million
- Total: $50 million

Then explain your reasoning for this allocation decision.
"""
        self.results = []

    def _extract_allocation(self, response: str, division_label: str, short_label: str) -> float:
        match = re.search(fr"{division_label} allocation\s*:\s*\$(\d+(\.\d+)?)", response, re.IGNORECASE)
        if not match:
            match = re.search(fr"Division {short_label}.*?\$(\d+(\.\d+)?)", response, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0

    def _classify_escalation(self, percent_to_div_a: float) -> str:
        if percent_to_div_a >= 75:
            return "Very High Escalation"
        elif percent_to_div_a >= 60:
            return "High Escalation"
        elif percent_to_div_a >= 40:
            return "Moderate Escalation"
        else:
            return "Low Escalation"

    def _extract_reasoning(self, response: str) -> str:
        match = re.search(r"Total:\s*\$50 million\n(.*?)$", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def run_single_trial(self, model_response: str, trial_id: str = None) -> Dict:
        division_a_allocation = self._extract_allocation(model_response, "Division A", "A")
        division_b_allocation = self._extract_allocation(model_response, "Division B", "B")
        total_allocated = division_a_allocation + division_b_allocation
        division_a_percent = (division_a_allocation / total_allocated * 100) if total_allocated > 0 else 0
        division_b_percent = (division_b_allocation / total_allocated * 100) if total_allocated > 0 else 0
        escalation_level = self._classify_escalation(division_a_percent)
        reasoning = self._extract_reasoning(model_response)
        result = {
            "trial_id": trial_id or f"trial_{len(self.results) + 1}",
            "timestamp": datetime.datetime.now().isoformat(),
            "division_a_allocation": division_a_allocation,
            "division_b_allocation": division_b_allocation,
            "total_allocated": total_allocated,
            "division_a_percent": round(division_a_percent, 2),
            "division_b_percent": round(division_b_percent, 2),
            "escalation_level": escalation_level,
            "shows_escalation": division_a_percent > 40,
            "reasoning": reasoning,
            "raw_response": model_response
        }
        self.results.append(result)
        return result

    def analyze_results(self):
        escalation_levels = [r["escalation_level"] for r in self.results]
        avg_a = statistics.mean([r["division_a_percent"] for r in self.results])
        avg_b = statistics.mean([r["division_b_percent"] for r in self.results])
        escalation_count = sum(r["shows_escalation"] for r in self.results)
        return {
            "average_allocation_to_division_a": round(avg_a, 2),
            "average_allocation_to_division_b": round(avg_b, 2),
            "escalation_count": escalation_count,
            "total_trials": len(self.results),
            "escalation_levels": escalation_levels
        }

    def get_scenario(self) -> str:
        return self.scenario

    def export_results(self, filename: str = None) -> str:
        filename = filename or f"escalation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        return filename


def get_gpt4o_response(prompt: str, temperature: float = 0.9) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strategic decision-making assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


def main():
    experiment = EscalationExperiment()
    scenario = experiment.get_scenario()

    print("=" * 80)
    print("ESCALATION OF COMMITMENT EXPERIMENT USING GPT‑4o")
    print("=" * 80)

    for i in range(10):
        print(f"\nRunning Trial {i + 1}...")
        response = get_gpt4o_response(scenario, temperature=0.9)
        result = experiment.run_single_trial(response, f"gpt4o_trial_{i + 1}")
        print(f"Division A: ${result['division_a_allocation']}M ({result['division_a_percent']}%)")
        print(f"Division B: ${result['division_b_allocation']}M ({result['division_b_percent']}%)")
        print(f"Escalation Level: {result['escalation_level']}")

    print("\nANALYSIS:")
    analysis = experiment.analyze_results()
    print(json.dumps(analysis, indent=2))
    filename = experiment.export_results()
    print(f"\nExported results to {filename}")


if __name__ == "__main__":
    main()
