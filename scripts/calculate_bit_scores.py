import os
import subprocess
import pandas as pd
from pathlib import Path

def get_rfam_acc(fasta_file):
    """Extracts Rfam accession number from a FASTA file name."""
    return os.path.basename(fasta_file).split('.')[0]

def calculate_bit_scores_for_family(cm_file, fasta_file, output_dir):
    """
    Calculates bit scores for a given FASTA file using a specific covariance model.
    """
    rfam_acc = get_rfam_acc(fasta_file)
    score_file = os.path.join(output_dir, f"{os.path.basename(fasta_file).replace('.fasta', '')}_scores.txt")

    # Check if scores have already been calculated
    if os.path.exists(score_file):
        print(f"Scores for {fasta_file} already exist. Skipping.")
        return

    # Construct the cmalign command
    command = [
        "cmalign",
        "--sfile", score_file,
        "-o", os.path.join(output_dir, f"{rfam_acc}_aligned.sto"), # Pfam format alignment
        cm_file,
        fasta_file
    ]

    print(f"Running command: {' '.join(command)}")

    try:
        # Execute the command
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully calculated bit scores for {fasta_file}")
    except FileNotFoundError:
        print("Error: 'cmalign' command not found. Make sure Infernal is installed and in your PATH.")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error calculating bit scores for {fasta_file}:")
        print(e.stderr)
        return

def main():
    # Define paths
    project_root = Path(__file__).resolve().parent.parent
    output_base_dir = project_root / "output"
    results_dir = project_root / "results" / "bit_scores"
    cm_dir = project_root / "results" / "alignments" / "cms"

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # List of models to process
    models = ["nuc_unaligned", "non-nuc_unaligned", "nuc_aligned", "non-nuc_aligned"]

    # Iterate over each model
    for model in models:
        model_path = output_base_dir / model
        if not os.path.isdir(model_path):
            print(f"Directory for model {model} not found at {model_path}. Skipping.")
            continue

        # Iterate over each fasta file in the model's directory
        for fasta_filename in os.listdir(model_path):
            if fasta_filename.endswith(".fasta"):
                fasta_file_path = model_path / fasta_filename
                rfam_acc = get_rfam_acc(fasta_file_path)
                cm_file_path = cm_dir / f"{rfam_acc}.cm"

                if not os.path.exists(cm_file_path):
                    print(f"Covariance model for {rfam_acc} not found at {cm_file_path}. Skipping.")
                    continue

                # Define a specific output directory for each model's scores
                model_score_dir = results_dir / model
                os.makedirs(model_score_dir, exist_ok=True)

                calculate_bit_scores_for_family(str(cm_file_path), str(fasta_file_path), str(model_score_dir))

if __name__ == "__main__":
    main()
