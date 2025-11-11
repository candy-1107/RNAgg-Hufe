import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_bit_scores(score_file):
    """Loads bit scores from a file, skipping commented lines."""
    scores = {}
    with open(score_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 7:  # Ensure there are enough parts
                seq_name = parts[1]  # seq name is in the second column
                try:
                    score = float(parts[6])  # bit score is in the seventh column
                    scores[seq_name] = score
                except ValueError:
                    # This will handle cases where the score is not a valid float
                    print(f"  - Could not parse score for {seq_name} in {score_file}. Skipping.")
                    continue
    return scores

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Assumes scripts is a subdir of base_dir
    results_dir = os.path.join(base_dir, "results")
    bit_scores_base_dir = os.path.join(results_dir, "bit_scores")
    max_ni_file = os.path.join(results_dir, "max_ni_values.csv")
    output_plot_file = os.path.join(results_dir, "bit_score_vs_max_ni.png")

    # Check if max_ni_values.csv exists
    if not os.path.exists(max_ni_file):
        print(f"Error: Max-NI data file not found at {max_ni_file}")
        print("Please generate this file first.")
        return

    # Load Max-NI data
    max_ni_df = pd.read_csv(max_ni_file)

    # Get unique families from the data
    if 'family' not in max_ni_df.columns:
        print(f"Error: 'family' column not found in {max_ni_file}")
        return
    families = max_ni_df['family'].unique()

    # --- Plotting Configuration ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # Define model names and colors for plotting
    model_mapping = {
        'nuc_unaligned': 'nuc-una',
        'non-nuc_unaligned': 'RNAgg-una',
        'nuc_aligned': 'nuc-ali',
        'non-nuc_aligned': 'RNAgg-ali'
    }
    colors = {
        'nuc-una': 'blue',
        'RNAgg-una': 'green',
        'nuc-ali': 'red',
        'RNAgg-ali': 'purple'
    }

    # --- Loop through each family to create a separate plot ---
    for family in families:
        fig, ax = plt.subplots(figsize=(12, 8))
        print(f"--- Processing family: {family} ---")

        # --- Data Loading and Plotting Loop for models ---
        for model_dir_name, model_plot_name in model_mapping.items():
            print(f"  Processing model: {model_plot_name}...")

            # Filter Max-NI data for the current model and family
            model_max_ni = max_ni_df[(max_ni_df['variant'] == model_plot_name) & (max_ni_df['family'] == family)]
            if model_max_ni.empty:
                print(f"    - No Max-NI data found for model {model_plot_name} in family {family}. Skipping.")
                continue

            # Load bit scores for the specific family of the current model
            score_filename = f"{family}_scores.txt"
            score_file_path = os.path.join(bit_scores_base_dir, model_dir_name, score_filename)

            if not os.path.isfile(score_file_path):
                print(f"    - Bit score file not found for {model_dir_name}/{score_filename}. Skipping.")
                continue

            bit_scores = load_bit_scores(score_file_path)

            if not bit_scores:
                print(f"    - No bit scores found for model {model_plot_name} in family {family}. Skipping.")
                continue

            # Create a copy to avoid SettingWithCopyWarning
            data = model_max_ni.copy()
            data['bit_score'] = data['seq_id'].map(bit_scores)

            # Drop rows where bit score could not be found
            data.dropna(subset=['bit_score'], inplace=True)

            if data.empty:
                print(f"    - No matching bit scores for Max-NI data in model {model_plot_name}, family {family}. Skipping.")
                continue

            # Sort by Max-NI (ascending) for correct plotting of both scatter and line
            data.sort_values(by='max_ni', inplace=True)

            # --- Plotting ---
            # Scatter plot
            ax.scatter(data['max_ni'], data['bit_score'],
                       alpha=0.3, s=15,
                       color=colors[model_plot_name])

            # 100-point moving average, calculated even if data is less than 100 points
            moving_avg = data['bit_score'].rolling(window=100, center=True, min_periods=1).mean()
            ax.plot(data['max_ni'], moving_avg,
                    color=colors[model_plot_name],
                    linewidth=2.0)


        # --- Final Plot Customization for the family ---
        # Create custom legend handles after the loop
        legend_handles = [plt.Line2D([0], [0], color=colors[model_mapping[m]], lw=2, label=model_mapping[m])
                          for m in model_mapping]

        ax.set_title(f'Bit Score vs. Max-NI for {family}', fontsize=16)
        ax.set_xlabel('Max-NI (Maximum Nucleotide Identity)', fontsize=12)
        ax.set_ylabel('Bit Score', fontsize=12)
        ax.invert_xaxis()  # Max-NI decreases from left to right
        ax.legend(handles=legend_handles, title='Variant', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        family_plot_file = os.path.join(results_dir, f"{family}_bit_score_vs_max_ni.png")
        plt.savefig(family_plot_file, dpi=300)
        print(f"  Plot for {family} saved to {family_plot_file}\n")
        plt.close(fig) # Close the figure to free up memory

if __name__ == "__main__":
    main()
