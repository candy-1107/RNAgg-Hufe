# -*- coding: utf-8 -*-

import sys, os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts"))
)

import os
import sys
import argparse
import re
import numpy as np
from Bio import AlignIO


# Convert sequence + secondary structure into 11 matrices per record:
# Singles: A, U, G, C, gap ('-')  -> set (i,i) = 1
# Pairs (directional): A-U, U-A, G-C, C-G, G-U, U-G -> set (i,j) = 1 for i<->j
# No OTHER matrix. Unrecognized bases/pairs are skipped.

PAIR_LABELS = ["A-U", "U-A", "G-C", "C-G", "G-U", "U-G"]
SINGLE_LABELS = ["A", "U", "G", "C", "gap"]
ALL_LABELS = SINGLE_LABELS + PAIR_LABELS  # 11 matrices

VALID_BASES = set("AUGC-")
VALID_PAIRS = set([p.replace("-", "") for p in PAIR_LABELS])  # AU, UA, GC, CG, GU, UG


def parse_pairs(ss_list):
    """Return list of (i,j) for base pairs using parentheses () <> {} [] as matching brackets."""
    stack = []
    pairs = []
    opening = set("(<[{")
    closing_map = {
        ")": "(",
        ">": "<",
        "]": "[",
        "}": "{",
    }
    for i, ch in enumerate(ss_list):
        if ch in opening:
            stack.append((ch, i))
        elif ch in closing_map:
            if not stack:
                continue  # malformed, ignore unmatched closer
            # pop until matching opener
            while stack:
                left_ch, left_idx = stack.pop()
                if closing_map[ch] == left_ch:
                    pairs.append((left_idx, i))
                    break
        else:
            # '.' or other symbols are not used for pairing
            pass
    return pairs


def build_matrices(seq_str: str, ss_str: str):
    """Build 11 matrices for one sequence + structure.
    seq_str: aligned/un-aligned sequence including gaps '-'
    ss_str: secondary structure annotation (same length as seq_str)
    Returns dict label->matrix (LxL int8) where L is sequence length.
    """
    L = len(seq_str)
    matrices = {label: np.zeros((L, L), dtype=np.int8) for label in ALL_LABELS}

    # Identify base pairs
    ss_list = list(ss_str)
    pair_indices = parse_pairs(ss_list)
    paired_pos = set()

    for i, j in pair_indices:
        if i < 0 or j >= L:
            continue
        b1, b2 = seq_str[i], seq_str[j]
        # Skip invalids and gaps in pairs
        if b1 not in VALID_BASES or b2 not in VALID_BASES:
            continue
        if b1 == '-' or b2 == '-':
            continue
        pair_key = (b1 + b2).upper()  # e.g. AU
        if pair_key in VALID_PAIRS:
            label = f"{b1}-{b2}".upper()
            matrices[label][i, j] = 1
            paired_pos.add(i)
            paired_pos.add(j)
        # else: non-canonical pair -> skip

    # Singles (unpaired) and gaps
    for i, b in enumerate(seq_str):
        if i in paired_pos:
            continue  # already part of a pair
        if b in "AUGC":
            matrices[b][i, i] = 1
        elif b == "-":
            matrices["gap"][i, i] = 1
        # else: unrecognized base -> skip

    return matrices


def sanitize_record_id(rid: str):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", rid)


# -------- Process Stockholm (.stk/.sto) --------

def process_alignment(stk_file: str, out_dir: str, aggregate: bool = False, save_text: bool = False):
    align = AlignIO.read(stk_file, "stockholm")
    # Find a consensus secondary structure key
    ss_key = None
    for k in ("secondary_structure", "SS_cons", "SS_consensus"):
        if k in align.column_annotations:
            ss_key = k
            break
    if ss_key is None:
        print("No consensus secondary_structure (secondary_structure/SS_cons/SS_consensus) found in Stockholm file.", file=sys.stderr)
        return
    SS_consensus = align.column_annotations[ss_key]
    L = len(SS_consensus)

    os.makedirs(out_dir, exist_ok=True)

    # Pre-init aggregate matrices (zeros)
    aggregate_mats = {label: np.zeros((L, L), dtype=np.int64) for label in ALL_LABELS}

    for record in align:
        seq_str = str(record.seq).upper()
        if len(seq_str) != L:
            print(f"Length mismatch for {record.id}, skipping.", file=sys.stderr)
            continue
        if re.search(r"[^AUGC-]", seq_str):
            print(f"{record.id} contains invalid characters, skipping.", file=sys.stderr)
            continue

        matrices = build_matrices(seq_str, SS_consensus)

        rec_id = sanitize_record_id(record.id)
        rec_dir = os.path.join(out_dir, rec_id)
        os.makedirs(rec_dir, exist_ok=True)

        # Save matrices for this record
        for label, mat in matrices.items():
            np.save(os.path.join(rec_dir, f"{label}.npy"), mat)
            if save_text:
                np.savetxt(os.path.join(rec_dir, f"{label}.txt"), mat, fmt="%d")

        if aggregate:
            for label, mat in matrices.items():
                aggregate_mats[label] += mat

    if aggregate:
        # Save aggregated matrices in one npz file (under out_dir)
        agg_path = os.path.join(out_dir, "aggregated_matrices.npz")
        np.savez(agg_path, **aggregate_mats)
        print(f"Aggregated matrices saved to {agg_path}", file=sys.stderr)


# -------- Process simple text (.txt) --------

def parse_txt_records(txt_file: str):
    """Yield (record_id, seq, struct) from a text file.
    Accepts formats:
      - id\tseq\tstruct
      - id seq struct
      - seq struct (id auto-generated)
    Lines starting with # are ignored.
    """
    with open(txt_file, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 3:
                rid, seq, struct = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                rid, seq, struct = f"rec_{ln}", parts[0], parts[1]
            else:
                print(f"Skip line {ln}: not enough columns.", file=sys.stderr)
                continue
            seq = seq.upper()
            if len(seq) != len(struct):
                print(f"Skip {rid}: seq/struct length mismatch {len(seq)} vs {len(struct)}.", file=sys.stderr)
                continue
            if re.search(r"[^AUGC-]", seq):
                print(f"Skip {rid}: invalid base in sequence.", file=sys.stderr)
                continue
            yield rid, seq, struct


def process_text(txt_file: str, out_root: str, aggregate: bool = False, save_text: bool = False):
    base = os.path.splitext(os.path.basename(txt_file))[0]
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)

    agg = None
    L_ref = None

    for rid, seq, struct in parse_txt_records(txt_file):
        L = len(seq)
        if L_ref is None:
            L_ref = L
            if aggregate:
                agg = {label: np.zeros((L_ref, L_ref), dtype=np.int64) for label in ALL_LABELS}
        elif aggregate and L != L_ref:
            # For simplicity, we aggregate only if all records share the same length
            print(f"Skip aggregation for {rid}: length differs from first record.", file=sys.stderr)
        mats = build_matrices(seq, struct)
        rec_dir = os.path.join(out_dir, sanitize_record_id(rid))
        os.makedirs(rec_dir, exist_ok=True)
        for label, mat in mats.items():
            np.save(os.path.join(rec_dir, f"{label}.npy"), mat)
            if save_text:
                np.savetxt(os.path.join(rec_dir, f"{label}.txt"), mat, fmt="%d")
        if aggregate and L == L_ref:
            for label, mat in mats.items():
                agg[label] += mat

    if aggregate and agg is not None:
        agg_path = os.path.join(out_dir, "aggregated_matrices.npz")
        np.savez(agg_path, **agg)
        print(f"Aggregated matrices saved to {agg_path}", file=sys.stderr)


# -------- Dispatcher & CLI --------

def process_input(path: str, out_dir: str, aggregate: bool = False, save_text: bool = False):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".stk", ".sto"):
        process_alignment(path, out_dir, aggregate=aggregate, save_text=save_text)
    else:
        # default to text parser
        process_text(path, out_dir, aggregate=aggregate, save_text=save_text)


def main():
    parser = argparse.ArgumentParser(description="Convert RNA sequence+structure inputs (.txt or .stk) to 11 matrices per sequence.")
    parser.add_argument("inputs", nargs="+", help="One or more input files: .txt (id seq struct) or .stk/.sto (Stockholm)")
    parser.add_argument("out_dir", help="Output directory root. For .txt, matrices are saved under out_dir/<basename>/<record_id>/.")
    parser.add_argument("--aggregate", action="store_true", help="Save per-input aggregated matrices as aggregated_matrices.npz")
    parser.add_argument("--text", action="store_true", help="Also save .txt matrices for easy viewing")
    args = parser.parse_args()

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"Input not found: {path}", file=sys.stderr)
            continue
        process_input(path, args.out_dir, aggregate=args.aggregate, save_text=args.text)


if __name__ == "__main__":
    main()
