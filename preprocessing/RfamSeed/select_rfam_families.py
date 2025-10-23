# -*- coding: utf-8 -*-
"""
select_rfam_families.py

Usage examples:
  - Extract first 10 families and write their .stk files to the current folder:
      python select_rfam_families.py --count 10

  - Extract a specific list of families (IDs separated by comma):
      python select_rfam_families.py --families RF00001,RF00002

  - Extract and run _make_input_consG4.py on each to produce aligned/unaligned txt outputs:
      python select_rfam_families.py --count 5 --run-cons

This script assumes either individual .stk files for families exist in the current folder
(e.g. RF00001.stk) or that there is a `Rfam.seed` file containing multiple Stockholm
entries separated per family (the script can split that file).

It writes out per-family .stk files into the same folder and (optionally) invokes
_make_input_consG4.py to create the corresponding _unaligned.txt and _aligned.txt files.

The script is intentionally lightweight and uses subprocess to call the local
_make_input_consG4.py script so it runs in the same Python environment.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import re

HERE = Path(__file__).resolve().parent


def list_families_from_rfam_seed(seed_path: Path):
    """Return an ordered list of family IDs found in a multi-family Stockholm Rfam.seed file.
    It looks for lines beginning with '#=GF AC <ID>' (common in Rfam seed files) or
    'RFxxxxx' patterns in headers.
    """
    families = []
    if not seed_path.exists():
        return families
    with seed_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            # look for accession lines
            m = re.search(r"#=GF AC\s+(RF\d{5})", line)
            if m:
                families.append(m.group(1))
    # fallback: try scanning for 'RFxxxxx' anywhere
    if not families:
        with seed_path.open("r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
            found = re.findall(r"RF\d{5}", text)
            # preserve order and uniqueness
            seen = set()
            families = [x for x in found if not (x in seen or seen.add(x))]
    return families


def split_seed_to_stk(seed_path: Path, out_dir: Path):
    """Split a concatenated Stockholm file (Rfam.seed) into per-family .stk files.
    Uses the common delimiter of '//\n' between Stockholm records. It attempts to
    name each output file using '#=GF AC RFxxxxx' header if present, otherwise
    falls back to a numeric index.
    Returns list of Path objects for the written .stk files (in order).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with seed_path.open("r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()

    records = [r.strip() for r in re.split(r"\n//\s*\n", content) if r.strip()]
    written = []
    idx = 1
    for rec in records:
        # find accession
        m = re.search(r"#=GF AC\s+(RF\d{5})", rec)
        if m:
            name = m.group(1)
        else:
            # try to find a header line with ID
            m2 = re.search(r"^\\s*([^\\s]+)\\s+", rec)
            if m2:
                name = m2.group(1)
            else:
                name = f"unknown_{idx:05d}"
        out_path = out_dir.joinpath(f"{name}.stk")
        with out_path.open("w", encoding="utf-8") as out_f:
            out_f.write(rec + "\n//\n")
        written.append(out_path)
        idx += 1
    return written


def run_make_input_consG4(stk_path: Path, out_dir: Path):
    """Invoke the local _make_input_consG4.py script for a given .stk file to
    produce RFxxxxx_unaligned.txt and RFxxxxx_aligned.txt in the same folder.
    Returns the subprocess exit code.
    """
    script = HERE.joinpath("_make_input_consG4.py")
    if not script.exists():
        print(f"Warning: {_make_input_consG4.py if False else script.name} not found; skipping run.")
        return 1
    # Prepare output subdirectories under out_dir
    out_dir = Path(out_dir)
    aligned_dir = out_dir.joinpath("aligned")
    unaligned_dir = out_dir.joinpath("unaligned")
    aligned_dir.mkdir(parents=True, exist_ok=True)
    unaligned_dir.mkdir(parents=True, exist_ok=True)

    stem = stk_path.stem
    out_unaligned = unaligned_dir.joinpath(f"{stem}_unaligned.txt")
    out_aligned = aligned_dir.joinpath(f"{stem}_aligned.txt")

    # call with absolute paths so it works regardless of current working dir
    cmd = [
        sys.executable,
        str(script.resolve()),
        str(stk_path.resolve()),
        str(out_unaligned.resolve()),
        str(out_aligned.resolve()),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"_make_input_consG4.py failed for {stk_path.name}:", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Select and extract families from Rfam.seed or existing .stk files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--count", type=int, help="Number of families to process (first N in Rfam.seed)")
    group.add_argument("--families", type=str, help="Comma-separated list of family IDs to process, e.g. RF00001,RF00002")
    parser.add_argument("--seed", type=str, default="Rfam.seed", help="path to multi-family Rfam.seed file")
    parser.add_argument("--out-dir", type=str, default=".", help="directory to write per-family .stk files and outputs")
    parser.add_argument("--run-cons", action="store_true", help="run _make_input_consG4.py for each extracted .stk")
    args = parser.parse_args()

    seed_path = Path(args.seed)
    out_dir = Path(args.out_dir)

    # If families explicitly provided, honor that order
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
    else:
        families = list_families_from_rfam_seed(seed_path)
        if args.count:
            families = families[: args.count]

    if not families:
        print("No families found. Ensure Rfam.seed exists or provide --families list.")
        return

    # If per-family .stk files already exist in folder, prefer them; otherwise split Rfam.seed
    existing_stk = {}
    for p in HERE.glob("*.stk"):
        existing_stk[p.stem] = p

    written = []
    # If all requested families exist as .stk, just collect them
    missing = [f for f in families if f not in existing_stk]
    if not missing:
        for f in families:
            written.append(existing_stk[f])
    else:
        # attempt to split seed and write all records, then pick requested ones
        if not seed_path.exists():
            print("Some requested .stk files are missing and Rfam.seed not found to split.")
            return
        all_written = split_seed_to_stk(seed_path, out_dir)
        # map by stem
        map_written = {p.stem: p for p in all_written}
        for f in families:
            if f in map_written:
                written.append(map_written[f])
            else:
                print(f"Family {f} not found in Rfam.seed; skipping.")

    print(f"Extracted {len(written)} stk files:")
    for p in written:
        print(" - ", p.name)

    if args.run_cons:
        for p in written:
            print(f"Running _make_input_consG4.py for {p.name} ...")
            rc = run_make_input_consG4(p, out_dir)
            if rc != 0:
                print(f"_make_input_consG4.py returned {rc} for {p.name}")


if __name__ == "__main__":
    main()
