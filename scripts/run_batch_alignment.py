#!/usr/bin/env python3
"""
Batch-align generated sequences against Rfam covariance models using cmalign.

This script automates the process of aligning generated sequences from different
models (nuc/non-nuc, aligned/unaligned) back to their respective Rfam
covariance models (CMs).

It assumes that 'cmalign' is available in the system's PATH (e.g., in a WSL/Linux env).
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# --- Configuration ---
# Define project structure and file locations.
# All paths are relative to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input directories
# Assumes Rfam seed alignments are in '.stk' format, used to build CMs.
RFAM_SEED_DIR = PROJECT_ROOT / 'preprocessing' / 'RfamSeed' / 'rfam_out' / 'rfam_stk'
GENERATED_SEQS_DIR = PROJECT_ROOT / 'output'

# Output directory for alignments and CMs
ALIGNMENT_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'alignments'
CM_DIR = ALIGNMENT_OUTPUT_DIR / 'cms'

# Models to process
MODELS = [
    "nuc_aligned",
    "nuc_unaligned",
    "non-nuc_aligned",
    "non-nuc_unaligned",
]

def build_cm_from_stk(stk_file: Path, cm_file: Path):
    """Builds a covariance model from a Stockholm file using 'cmbuild'."""
    if cm_file.exists():
        print(f"CM file already exists, skipping build: {cm_file.name}")
        return

    print(f"Building CM for {stk_file.stem}...")
    try:
        # Ensure the parent directory for the CM file exists
        cm_file.parent.mkdir(parents=True, exist_ok=True)

        command = ['cmbuild', str(cm_file), str(stk_file)]
        # Using capture_output=True to hide the verbose output of cmbuild
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("\n[ERROR] 'cmbuild' command not found.", file=sys.stderr)
        print("Please ensure Infernal is installed and in your PATH (e.g., via Conda in a WSL/Linux environment).", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] cmbuild failed for {stk_file.name} with return code {e.returncode}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during cmbuild: {e}", file=sys.stderr)
        sys.exit(1)

def run_cmalign(cm_file: Path, fasta_file: Path, output_sto_file: Path):
    """Runs cmalign to align sequences to a covariance model."""
    if output_sto_file.exists():
        print(f"Alignment file already exists, skipping alignment: {output_sto_file.name}")
        return

    print(f"Aligning {fasta_file.name}...")
    try:
        # Ensure the output directory exists
        output_sto_file.parent.mkdir(parents=True, exist_ok=True)

        command = ['cmalign', '--outformat', 'Stockholm', str(cm_file), str(fasta_file)]
        with open(output_sto_file, 'w') as f_out:
            subprocess.run(command, check=True, stdout=f_out, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("\n[ERROR] 'cmalign' command not found.", file=sys.stderr)
        print("Please ensure Infernal is installed and in your PATH (e.g., via Conda in a WSL/Linux environment).", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] cmalign failed for {fasta_file.name} with return code {e.returncode}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        # We continue to the next file instead of exiting
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during cmalign: {e}", file=sys.stderr)


def main():
    """Main function to run the batch alignment workflow."""
    parser = argparse.ArgumentParser(description="Batch align generated sequences using cmalign.")
    parser.add_argument(
        '--families',
        type=str,
        default=",".join([f"RF{i:05d}" for i in range(1, 11)]),
        help='Comma-separated list of Rfam family IDs to process (e.g., "RF00001,RF00002").'
    )
    args = parser.parse_args()
    families_to_process = [fam.strip() for fam in args.families.split(',')]

    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Processing {len(families_to_process)} families: {families_to_process}\n")

    # --- Workflow ---
    # 1. Build CM for each family from seed alignment
    print("--- Step 1: Building Covariance Models ---")
    for family_id in families_to_process:
        stk_file = RFAM_SEED_DIR / f"{family_id}.stk"
        cm_file = CM_DIR / f"{family_id}.cm"

        if not stk_file.exists():
            print(f"[WARNING] Seed Stockholm file not found for {family_id}, skipping: {stk_file}", file=sys.stderr)
            continue

        build_cm_from_stk(stk_file, cm_file)

    print("\n--- Step 2: Aligning Generated Sequences ---")
    # 2. Align generated sequences for each model and family
    for model_name in MODELS:
        for family_id in families_to_process:
            cm_file = CM_DIR / f"{family_id}.cm"
            if not cm_file.exists():
                print(f"[WARNING] CM file for {family_id} not found, skipping alignment for this family.", file=sys.stderr)
                continue

            fasta_file = GENERATED_SEQS_DIR / model_name / f"{family_id}.fasta"
            if not fasta_file.exists():
                print(f"[WARNING] Generated FASTA file not found, skipping: {fasta_file}", file=sys.stderr)
                continue

            # Define output path for the alignment
            output_sto_dir = ALIGNMENT_OUTPUT_DIR / model_name
            output_sto_file = output_sto_dir / f"{family_id}_aligned.sto"

            run_cmalign(cm_file, fasta_file, output_sto_file)

    print("\n✅ Batch alignment process finished.")
    print(f"Alignment files are saved in: {ALIGNMENT_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
