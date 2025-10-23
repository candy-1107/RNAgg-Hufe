#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA Secondary Structure Predictor
==================================
A Python implementation of RNA secondary structure prediction, similar to mxfold2.

Features:
- Multiple algorithms: Nussinov (simple), Nussinov with energy (advanced)
- Automatic encoding detection for input files
- Support for standard FASTA format
- Output in dot-bracket notation
- Configurable parameters

Usage:
    python predict_structure.py predict input.fa [options]
    python predict_structure.py predict input.fa --algorithm nussinov-energy --output result.txt
    python predict_structure.py predict input.fa --min-loop-length 3

Author: Auto-generated for RNAgg project
Date: 2025-10-23
"""

import sys
import os
import argparse
import codecs
from pathlib import Path
from typing import List, Tuple, Optional

# ============================================================================
# Encoding Detection
# ============================================================================


def detect_encoding(path: str, nbytes: int = 4096) -> str:
    """
    Detect file encoding by checking BOM and trying common encodings.
    """
    p = Path(path)
    if not p.exists():
        return "utf-8"

    try:
        with p.open("rb") as fh:
            raw = fh.read(nbytes)
    except Exception:
        return "utf-8"

    # BOM checks
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"

    # Try charset detection libraries if available
    try:
        from charset_normalizer import from_bytes

        result = from_bytes(raw)
        if result:
            best = result.best()
            if best and best.encoding:
                return best.encoding
    except ImportError:
        pass

    try:
        import chardet

        res = chardet.detect(raw)
        if res and res.get("encoding"):
            return res["encoding"]
    except ImportError:
        pass

    # Try common encodings as fallback
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except Exception:
            continue

    return "latin-1"


def open_text(path: str, mode: str = "r") -> object:
    """
    Open text file with automatic encoding detection.
    """
    if "b" in mode:
        return open(path, mode)

    encoding = detect_encoding(path)
    return open(path, mode, encoding=encoding)


# ============================================================================
# FASTA I/O
# ============================================================================


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Read FASTA file and return list of (header, sequence) tuples.
    Handles various encodings automatically.
    """
    records = []
    with open_text(path, "r") as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                # Remove whitespace and convert to uppercase
                seq_lines.append(line.replace(" ", "").replace("\t", "").upper())
        if header is not None:
            records.append((header, "".join(seq_lines)))
    return records


def write_results(
    records: List[Tuple[str, str, str]], output_path: Optional[str] = None
):
    """
    Write prediction results in mxfold2-like format.
    Format: >header\nsequence\ndot-bracket\n
    """
    lines = []
    for header, seq, structure in records:
        lines.append(f">{header}")
        lines.append(seq)
        lines.append(structure)

    output_text = "\n".join(lines)
    if output_text and not output_text.endswith("\n"):
        output_text += "\n"

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        sys.stdout.write(output_text)


# ============================================================================
# RNA Secondary Structure Prediction Algorithms
# ============================================================================


class RNAPredictor:
    """Base class for RNA structure prediction algorithms."""

    # Standard Watson-Crick and wobble pairs
    PAIRS = {("A", "U"), ("U", "A"), ("C", "G"), ("G", "C"), ("G", "U"), ("U", "G")}

    def __init__(self, min_loop_length: int = 3):
        """
        Initialize predictor.

        Args:
            min_loop_length: Minimum number of unpaired bases in a loop (default: 3)
        """
        self.min_loop_length = min_loop_length

    def can_pair(self, base1: str, base2: str) -> bool:
        """Check if two bases can form a pair."""
        return (base1, base2) in self.PAIRS

    def predict(self, seq: str) -> str:
        """
        Predict secondary structure for a sequence.
        Must be implemented by subclasses.

        Args:
            seq: RNA sequence (uppercase)

        Returns:
            Dot-bracket structure notation
        """
        raise NotImplementedError


class NussinovPredictor(RNAPredictor):
    """
    Simple Nussinov algorithm for maximum base-pair prediction.
    No energy model, all valid pairs have equal weight.
    """

    def predict(self, seq: str) -> str:
        """Predict structure using basic Nussinov algorithm."""
        n = len(seq)
        if n == 0:
            return ""

        # DP table and traceback
        dp = [[0] * n for _ in range(n)]
        trace = [[None] * n for _ in range(n)]

        min_dist = self.min_loop_length + 1

        # Fill DP table
        for length in range(1, n):
            for i in range(n - length):
                j = i + length

                # Option 1: j is unpaired
                best = dp[i][j - 1] if j > 0 else 0
                trace_choice = ("skip_j", None)

                # Option 2: j pairs with some k
                if j - i >= min_dist:
                    for k in range(i, j - min_dist + 1):
                        if self.can_pair(seq[k], seq[j]):
                            left = dp[i][k - 1] if k > i else 0
                            middle = dp[k + 1][j - 1] if k + 1 <= j - 1 else 0
                            val = left + 1 + middle
                            if val > best:
                                best = val
                                trace_choice = ("pair", k)

                # Option 3: bifurcation
                for k in range(i, j):
                    val = dp[i][k] + dp[k + 1][j]
                    if val > best:
                        best = val
                        trace_choice = ("bifur", k)

                dp[i][j] = best
                trace[i][j] = trace_choice

        # Traceback to construct structure
        brackets = ["."] * n

        def backtrack(i: int, j: int):
            if i >= j:
                return
            choice = trace[i][j]
            if choice is None:
                return

            typ, k = choice
            if typ == "skip_j":
                backtrack(i, j - 1)
            elif typ == "pair":
                brackets[k] = "("
                brackets[j] = ")"
                if k > i:
                    backtrack(i, k - 1)
                if k + 1 <= j - 1:
                    backtrack(k + 1, j - 1)
            elif typ == "bifur":
                backtrack(i, k)
                backtrack(k + 1, j)

        backtrack(0, n - 1)
        return "".join(brackets)


class NussinovEnergyPredictor(RNAPredictor):
    """
    Enhanced Nussinov algorithm with simple energy scoring.
    Uses a simplified energy model based on base-pair types.
    """

    # Simple energy scores (negative = more stable)
    # These are simplified; real RNA folding uses complex thermodynamic parameters
    PAIR_ENERGY = {
        ("C", "G"): -3.0,  # Most stable
        ("G", "C"): -3.0,
        ("A", "U"): -2.0,
        ("U", "A"): -2.0,
        ("G", "U"): -1.0,  # Wobble pair
        ("U", "G"): -1.0,
    }

    def __init__(self, min_loop_length: int = 3, au_penalty: float = 0.5):
        """
        Initialize energy-based predictor.

        Args:
            min_loop_length: Minimum loop size
            au_penalty: Energy penalty for AU pairs at helix ends
        """
        super().__init__(min_loop_length)
        self.au_penalty = au_penalty

    def get_pair_energy(self, base1: str, base2: str) -> float:
        """Get energy score for a base pair."""
        return self.PAIR_ENERGY.get((base1, base2), 0.0)

    def predict(self, seq: str) -> str:
        """Predict structure using energy-based Nussinov."""
        n = len(seq)
        if n == 0:
            return ""

        # DP table stores minimum energy (more negative = better)
        dp = [[0.0] * n for _ in range(n)]
        trace = [[None] * n for _ in range(n)]

        min_dist = self.min_loop_length + 1

        # Fill DP table (minimizing energy)
        for length in range(1, n):
            for i in range(n - length):
                j = i + length

                # Option 1: j unpaired
                best = dp[i][j - 1] if j > 0 else 0.0
                trace_choice = ("skip_j", None)

                # Option 2: j pairs with k
                if j - i >= min_dist:
                    for k in range(i, j - min_dist + 1):
                        if self.can_pair(seq[k], seq[j]):
                            energy = self.get_pair_energy(seq[k], seq[j])

                            # Apply AU penalty at ends if needed
                            if (seq[k], seq[j]) in [("A", "U"), ("U", "A")]:
                                energy += self.au_penalty

                            left = dp[i][k - 1] if k > i else 0.0
                            middle = dp[k + 1][j - 1] if k + 1 <= j - 1 else 0.0
                            val = left + energy + middle

                            if val < best:  # Lower energy is better
                                best = val
                                trace_choice = ("pair", k)

                # Option 3: bifurcation
                for k in range(i, j):
                    val = dp[i][k] + dp[k + 1][j]
                    if val < best:
                        best = val
                        trace_choice = ("bifur", k)

                dp[i][j] = best
                trace[i][j] = trace_choice

        # Traceback
        brackets = ["."] * n

        def backtrack(i: int, j: int):
            if i >= j:
                return
            choice = trace[i][j]
            if choice is None:
                return

            typ, k = choice
            if typ == "skip_j":
                backtrack(i, j - 1)
            elif typ == "pair":
                brackets[k] = "("
                brackets[j] = ")"
                if k > i:
                    backtrack(i, k - 1)
                if k + 1 <= j - 1:
                    backtrack(k + 1, j - 1)
            elif typ == "bifur":
                backtrack(i, k)
                backtrack(k + 1, j)

        backtrack(0, n - 1)
        return "".join(brackets)


# ============================================================================
# Main Prediction Interface
# ============================================================================


def predict_structures(
    fasta_path: str,
    algorithm: str = "nussinov",
    min_loop_length: int = 3,
    au_penalty: float = 0.5,
    output_path: Optional[str] = None,
    verbose: bool = False,
):
    """
    Main function to predict RNA secondary structures from a FASTA file.

    Args:
        fasta_path: Path to input FASTA file
        algorithm: Algorithm to use ('nussinov' or 'nussinov-energy')
        min_loop_length: Minimum hairpin loop size
        au_penalty: Energy penalty for AU pairs (only for nussinov-energy)
        output_path: Output file path (None = stdout)
        verbose: Print progress messages
    """
    # Read sequences
    if verbose:
        print(f"Reading sequences from {fasta_path}...", file=sys.stderr)

    records = read_fasta(fasta_path)

    if verbose:
        print(f"Found {len(records)} sequence(s)", file=sys.stderr)

    # Select algorithm
    if algorithm == "nussinov":
        predictor = NussinovPredictor(min_loop_length=min_loop_length)
    elif algorithm == "nussinov-energy":
        predictor = NussinovEnergyPredictor(
            min_loop_length=min_loop_length, au_penalty=au_penalty
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if verbose:
        print(f"Using algorithm: {algorithm}", file=sys.stderr)

    # Predict structures
    results = []
    for i, (header, seq) in enumerate(records, 1):
        if verbose:
            print(f"Predicting structure {i}/{len(records)}: {header}", file=sys.stderr)

        # Clean sequence (remove non-ACGU characters)
        clean_seq = "".join(c for c in seq if c in "ACGTU")

        if len(clean_seq) != len(seq):
            if verbose:
                print(
                    f"  Warning: removed {len(seq) - len(clean_seq)} invalid characters",
                    file=sys.stderr,
                )

        if not clean_seq:
            if verbose:
                print(f"  Warning: empty sequence, skipping", file=sys.stderr)
            continue

        structure = predictor.predict(clean_seq)
        results.append((header, clean_seq, structure))

    # Write results
    if verbose:
        print(f"Writing results...", file=sys.stderr)

    write_results(results, output_path)

    if verbose:
        print("Done!", file=sys.stderr)


# ============================================================================
# Command-line Interface
# ============================================================================


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="RNA Secondary Structure Predictor (mxfold2-like)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction with Nussinov algorithm
  python predict_structure.py predict input.fa
  
  # Use energy-based algorithm
  python predict_structure.py predict input.fa --algorithm nussinov-energy
  
  # Save to file with verbose output
  python predict_structure.py predict input.fa -o output.txt -v
  
  # Adjust parameters
  python predict_structure.py predict input.fa --min-loop-length 4 --au-penalty 0.7

Algorithms:
  nussinov         - Simple maximum base-pair prediction (fast)
  nussinov-energy  - Energy-based prediction (more accurate)
""",
    )

    parser.add_argument(
        "command",
        choices=["predict"],
        help='Command to execute (currently only "predict" is supported)',
    )

    parser.add_argument("input", help="Input FASTA file containing RNA sequences")

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
        help="Energy penalty for AU pairs at helix ends (default: 0.5)",
    )

    parser.add_argument("-o", "--output", help="Output file (default: stdout)")

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output to stderr"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Run prediction
    try:
        predict_structures(
            fasta_path=args.input,
            algorithm=args.algorithm,
            min_loop_length=args.min_loop_length,
            au_penalty=args.au_penalty,
            output_path=args.output,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
