#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch RNA Secondary Structure Predictor
========================================
Process multiple FASTA files in batch mode.

Usage:
    python batch_predict.py input_dir/ output_dir/ [options]
    python batch_predict.py input_dir/ output_dir/ --pattern "*.fa" --algorithm nussinov-energy

Author: Auto-generated for RNAgg project
Date: 2025-10-23
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List

# Import predictor from the same directory
try:
    from predict_structure import predict_structures
except ImportError:
    print("Error: Cannot import predict_structure module", file=sys.stderr)
    print("Make sure predict_structure.py is in the same directory", file=sys.stderr)
    sys.exit(1)


def find_fasta_files(input_dir: str, pattern: str = "*.fa") -> List[Path]:
    """
    Find all FASTA files in a directory.

    Args:
        input_dir: Directory to search
        pattern: File pattern (e.g., "*.fa", "*.fasta")

    Returns:
        List of Path objects for FASTA files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Find files matching pattern
    files = list(input_path.glob(pattern))

    # Also try common FASTA extensions
    if not files:
        for ext in ["*.fasta", "*.fna", "*.fa"]:
            files.extend(input_path.glob(ext))

    return sorted(files)


def batch_predict(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.fa",
    algorithm: str = "nussinov-energy",
    min_loop_length: int = 3,
    au_penalty: float = 0.5,
    verbose: bool = False,
    keep_name: bool = True,
):
    """
    Process multiple FASTA files in batch.

    Args:
        input_dir: Input directory containing FASTA files
        output_dir: Output directory for results
        pattern: File pattern to match
        algorithm: Prediction algorithm
        min_loop_length: Minimum hairpin loop size
        au_penalty: Energy penalty for AU pairs
        verbose: Print progress
        keep_name: Keep original filename (change extension to .txt)
    """
    # Find input files
    if verbose:
        print(
            f"Searching for files in {input_dir} matching {pattern}...", file=sys.stderr
        )

    fasta_files = find_fasta_files(input_dir, pattern)

    if not fasta_files:
        print(f"Warning: No FASTA files found in {input_dir}", file=sys.stderr)
        return

    if verbose:
        print(f"Found {len(fasta_files)} file(s)", file=sys.stderr)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output directory: {output_dir}", file=sys.stderr)
        print(f"Algorithm: {algorithm}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

    # Process each file
    success_count = 0
    error_count = 0

    for i, fasta_file in enumerate(fasta_files, 1):
        if verbose:
            print(
                f"\n[{i}/{len(fasta_files)}] Processing: {fasta_file.name}",
                file=sys.stderr,
            )

        # Determine output filename
        if keep_name:
            output_file = output_path / (fasta_file.stem + "_structure.txt")
        else:
            output_file = output_path / f"result_{i:04d}.txt"

        try:
            # Run prediction
            predict_structures(
                fasta_path=str(fasta_file),
                algorithm=algorithm,
                min_loop_length=min_loop_length,
                au_penalty=au_penalty,
                output_path=str(output_file),
                verbose=verbose,
            )

            success_count += 1
            if verbose:
                print(f"  ✓ Saved to: {output_file.name}", file=sys.stderr)

        except Exception as e:
            error_count += 1
            print(f"  ✗ Error processing {fasta_file.name}: {e}", file=sys.stderr)
            if verbose:
                import traceback

                traceback.print_exc()

    # Summary
    if verbose:
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"Batch processing complete!", file=sys.stderr)
        print(f"  Success: {success_count}/{len(fasta_files)}", file=sys.stderr)
        print(f"  Errors:  {error_count}/{len(fasta_files)}", file=sys.stderr)
        print(f"  Output directory: {output_dir}", file=sys.stderr)


def main():
    """Command-line interface for batch prediction."""
    parser = argparse.ArgumentParser(
        description="Batch RNA Secondary Structure Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all .fa files in input/ directory
  python batch_predict.py input/ output/
  
  # Process all .fasta files with energy algorithm
  python batch_predict.py data/ results/ --pattern "*.fasta" --algorithm nussinov-energy
  
  # Verbose output with custom parameters
  python batch_predict.py input/ output/ -v --min-loop-length 4
""",
    )

    parser.add_argument("input_dir", help="Input directory containing FASTA files")

    parser.add_argument("output_dir", help="Output directory for prediction results")

    parser.add_argument(
        "-p", "--pattern", default="*.fa", help="File pattern to match (default: *.fa)"
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        choices=["nussinov", "nussinov-energy"],
        default="nussinov-energy",
        help="Prediction algorithm (default: nussinov-energy)",
    )

    parser.add_argument(
        "-m",
        "--min-loop-length",
        type=int,
        default=3,
        help="Minimum hairpin loop length (default: 3)",
    )

    parser.add_argument(
        "--au-penalty",
        type=float,
        default=0.5,
        help="Energy penalty for AU pairs (default: 0.5)",
    )

    parser.add_argument(
        "--numbered-output",
        action="store_true",
        help="Use numbered output files instead of keeping original names",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run batch prediction
    try:
        batch_predict(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            algorithm=args.algorithm,
            min_loop_length=args.min_loop_length,
            au_penalty=args.au_penalty,
            verbose=args.verbose,
            keep_name=not args.numbered_output,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
