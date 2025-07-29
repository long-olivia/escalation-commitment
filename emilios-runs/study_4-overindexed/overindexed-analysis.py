import json
from collections import Counter
import numpy as np
import os
from datetime import datetime

def load_data(file_path):
    """Load the escalation dataset from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Normalize: if it's a dict, wrap it in a list
    if isinstance(data, dict):
        data = [data]

    print(f"Loaded {len(data)} participant entries.")
    return data

def compute_summary(data):
    """Compute summary statistics across the dataset."""
    escalation_levels_all = []
    escalation_counts = []
    total_trials = []
    allocation_a = []
    allocation_b = []

    for entry in data:
        if not isinstance(entry, dict):
            print(f"Skipping invalid entry: {entry}")
            continue  # skip non-dict entries

        escalation_levels_all.extend(entry.get("escalation_levels", []))
        escalation_counts.append(entry.get("escalation_count", 0))
        total_trials.append(entry.get("total_trials", 0))
        allocation_a.append(entry.get("average_allocation_to_division_a", 0.0))
        allocation_b.append(entry.get("average_allocation_to_division_b", 0.0))

    # Summary stats
    escalation_rate = sum(escalation_counts) / sum(total_trials) if sum(total_trials) > 0 else 0
    level_distribution = Counter(escalation_levels_all)
    avg_alloc_a = np.mean(allocation_a)
    avg_alloc_b = np.mean(allocation_b)

    summary = {
        "Total Participants": len(data),
        "Valid Entries Analyzed": len(escalation_counts),
        "Total Trials": sum(total_trials),
        "Total Escalations": sum(escalation_counts),
        "Escalation Rate": round(escalation_rate, 3),
        "Escalation Level Distribution": dict(level_distribution),
        "Average Allocation to Division A": round(avg_alloc_a, 2),
        "Average Allocation to Division B": round(avg_alloc_b, 2),
    }

    return summary

def print_summary(summary):
    """Print the summary statistics in a readable format."""
    print("\n===== Escalation of Commitment Summary =====")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  - {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")
    print("===========================================\n")

def save_summary_to_file(summary, output_dir, output_filename="study4_summary_results.txt"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        f.write("===== Escalation of Commitment Summary =====\n")
        for k, v in summary.items():
            if isinstance(v, dict):
                f.write(f"{k}:\n")
                for sub_k, sub_v in v.items():
                    f.write(f"  - {sub_k}: {sub_v}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write("===========================================\n")

    print(f"\nSummary saved to: {output_path}")

# ===============================
# MAIN EXECUTION BLOCK
# ===============================
if __name__ == "__main__":
    # Path to your JSON input
    file_path = "/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_4-overindexed/results/overindexed-results_o4_mini_2025_04_16.json"

    # Output directory and filename
    output_dir = "/Users/leo/Documents/GitHub/escalation-commitment/emilios-runs/study_4-overindexed"
    output_filename = "overindexed-analysis-results.txt"

    # Run analysis
    data = load_data(file_path)
    summary_stats = compute_summary(data)
    print_summary(summary_stats)
    save_summary_to_file(summary_stats, output_dir, output_filename)