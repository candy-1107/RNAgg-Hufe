import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from Bio import SeqIO
import re

def get_fasta_lengths(file_path):
    """Reads a FASTA file and returns a list of sequence lengths."""
    lengths = []
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            lengths.append(len(record.seq))
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}")
    return lengths

def get_stockholm_lengths(file_path):
    """Reads a Stockholm file and returns a list of sequence lengths (ignoring gaps)."""
    lengths = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('//'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    # Remove gaps and get the length of the sequence
                    seq = parts[1].replace('-', '').replace('.', '')
                    lengths.append(len(seq))
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}")
    return lengths

def main():
    """Main function to generate the length distribution plot."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(base_dir, "output")
    seed_dir = os.path.join(base_dir, "preprocessing", "RfamSeed", "rfam_out", "rfam_stk")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    model_variants = {
        'nuc-una': 'nuc_unaligned',
        'RNAgg-una': 'non-nuc_unaligned',
        'nuc-ali': 'nuc_aligned',
        'RNAgg-ali': 'non-nuc_aligned'
    }

    # Assuming RfamGen data is not available as per file structure, it will be skipped.
    # If RfamGen data exists in a specific directory, add it to model_variants.
    # For example: 'RfamGen': 'path/to/rfamgen_output'

    families = [f"RF{i:05d}" for i in range(1, 11)]
    all_data = []

    # --- Data Collection and Normalization ---
    for family in families:
        print(f"Processing family: {family}...")

        # 1. Get seed lengths and calculate mean for normalization
        seed_stk_file = os.path.join(seed_dir, f"{family}.stk")
        seed_lengths = get_stockholm_lengths(seed_stk_file)
        if not seed_lengths:
            print(f"  - No seed sequences found for {family}. Skipping normalization.")
            continue
        mean_seed_length = sum(seed_lengths) / len(seed_lengths)

        # 2. Add normalized seed lengths to data
        for length in seed_lengths:
            all_data.append({
                'family': family,
                'variant': 'seed',
                'normalized_length': length / mean_seed_length
            })

        # 3. Get generated sequence lengths for each model variant
        for variant_name, variant_dir in model_variants.items():
            fasta_file = os.path.join(output_dir, variant_dir, f"{family}.fasta")
            gen_lengths = get_fasta_lengths(fasta_file)
            if not gen_lengths:
                print(f"  - No generated sequences found for {variant_name} in {family}.")
                continue

            for length in gen_lengths:
                all_data.append({
                    'family': family,
                    'variant': variant_name,
                    'normalized_length': length / mean_seed_length
                })

    if not all_data:
        print("No data collected. Exiting.")
        return

    df = pd.DataFrame(all_data)

    # --- Plotting ---
    sns.set_theme(style="whitegrid")

    # Define the order and colors for the plot
    full_hue_order = ['nuc-una', 'RNAgg-una', 'nuc-ali', 'RNAgg-ali', 'seed', 'RfamGen']
    full_palette = {
        'nuc-una': '#8dd3c7',
        'RNAgg-una': '#ffffb3',
        'nuc-ali': '#bebada',
        'RNAgg-ali': '#fb8072',
        'seed': '#80b1d3',
        'RfamGen': '#fdb462'
    }

    # Filter palette to only the variants we have data for
    final_palette = {k: v for k, v in full_palette.items() if k in df['variant'].unique()}
    final_hue_order = [h for h in full_hue_order if h in df['variant'].unique()]

    # --- Plot for RF00001-RF00005 ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    df1 = df[df['family'].isin([f"RF{i:05d}" for i in range(1, 6)])]
    sns.violinplot(ax=ax1, data=df1, x='family', y='normalized_length', hue='variant',
                   hue_order=final_hue_order, palette=final_palette, inner='box', cut=0)
    ax1.set_ylabel("Normalized length", fontsize=12)
    ax1.set_xlabel(None)

    # Create and place legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend_.remove()
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(final_hue_order), title=None, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path1 = os.path.join(results_dir, "length_distribution_plot_1-5.png")
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"\nPlot 1 saved to {output_path1}")
    plt.close(fig1) # Close the figure to free up memory

    # --- Plot for RF00006-RF00010 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    df2 = df[df['family'].isin([f"RF{i:05d}" for i in range(6, 11)])]
    sns.violinplot(ax=ax2, data=df2, x='family', y='normalized_length', hue='variant',
                   hue_order=final_hue_order, palette=final_palette, inner='box', cut=0)
    ax2.set_ylabel("Normalized length", fontsize=12)
    ax2.set_xlabel(None)

    # Create and place legend
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend_.remove()
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(final_hue_order), title=None, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path2 = os.path.join(results_dir, "length_distribution_plot_6-10.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Plot 2 saved to {output_path2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
