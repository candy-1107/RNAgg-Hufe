# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess


def get_all_fam_ids(seed_path):
    fam_ids = []
    with open(seed_path, encoding="iso-8859-1") as f:
        for line in f:
            if line.startswith("#=GF AC"):
                fam_id = line.split()[2].strip()
                fam_ids.append(fam_id)
    return fam_ids


def ungapped_len(seq):
    return len([c for c in seq if c not in '.-~'])


def filter_stk_by_length(src_stk, target_len, out_stk):
    """Write blocks from src_stk that contain at least one sequence with ungapped length == target_len to out_stk.
    Returns True if any block was written.
    """
    any_written = False
    # read with utf-8 because the generated .stk is written utf-8
    with open(src_stk, 'r', encoding='utf-8') as f:
        block = []
        for line in f:
            block.append(line)
            if line.strip() == '//':
                seqs = {}
                for bl in block:
                    if bl.strip() == '//' or bl.startswith('#'):
                        continue
                    parts = bl.rstrip('\n').split()
                    if len(parts) >= 2:
                        name = parts[0]
                        seq = ''.join(parts[1:])
                        seqs.setdefault(name, '')
                        seqs[name] += seq
                ok = any(ungapped_len(s) == target_len for s in seqs.values())
                if ok:
                    with open(out_stk, 'a', encoding='utf-8') as fout:
                        for bl in block:
                            fout.write(bl)
                    any_written = True
                block = []
        # final block if file doesn't end with //
        if block:
            seqs = {}
            for bl in block:
                if bl.strip() == '//' or bl.startswith('#'):
                    continue
                parts = bl.rstrip('\n').split()
                if len(parts) >= 2:
                    name = parts[0]
                    seq = ''.join(parts[1:])
                    seqs.setdefault(name, '')
                    seqs[name] += seq
            ok = any(ungapped_len(s) == target_len for s in seqs.values())
            if ok:
                with open(out_stk, 'a', encoding='utf-8') as fout:
                    for bl in block:
                        fout.write(bl)
                any_written = True
    return any_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, help="Rfam.seed file path")
    parser.add_argument(
        "--output_stk",
        required=True,
        help="Root output folder (will save .stk under <output_stk>/rfam_stk)",
    )
    parser.add_argument(
        "--output_g4",
        required=True,
        help="Root output folder for G4 files (defaults: unaligned in <output_g4>/rfam_unaligned, aligned in <output_g4>/rfam_aligned)",
    )
    parser.add_argument(
        "--output_unaligned",
        required=False,
        help="Optional: output folder for unaligned .txt files (overrides default <output_g4>/rfam_unaligned)",
        default=None,
    )
    parser.add_argument(
        "--output_aligned",
        required=False,
        help="Optional: output folder for aligned .txt files (overrides default <output_g4>/rfam_aligned)",
        default=None,
    )
    # New options
    parser.add_argument("-n", type=int, default=0, help="Process only first N families (0 = all)")
    parser.add_argument("--target", type=int, default=0, help="Target ungapped sequence length to filter for (0 = no filtering)")
    parser.add_argument("--only-filter", action='store_true', help="Only write filtered .stk, do not run G4 processing")

    args = parser.parse_args()

    # prepare directories
    stk_dir = os.path.join(args.output_stk, "rfam_stk")
    os.makedirs(stk_dir, exist_ok=True)

    una_dir = args.output_unaligned or os.path.join(args.output_g4, "rfam_unaligned")
    ali_dir = args.output_aligned or os.path.join(args.output_g4, "rfam_aligned")
    os.makedirs(una_dir, exist_ok=True)
    os.makedirs(ali_dir, exist_ok=True)

    script_dir = (
        os.path.abspath(os.path.dirname(__file__))
        if "__file__" in globals()
        else os.getcwd()
    )

    fam_ids = get_all_fam_ids(args.seed)
    print(f"Found {len(fam_ids)} families in {args.seed}.")

    processed = 0
    for fam_id in fam_ids:
        if args.n and processed >= args.n:
            break

        stk_path = os.path.join(stk_dir, f"{fam_id}.stk")
        una_path = os.path.join(una_dir, f"{fam_id}_unaligned.txt")
        ali_path = os.path.join(ali_dir, f"{fam_id}_aligned.txt")

        # Step 1: generate .stk file using _makeRfamSeedSto.py
        # Ensure we pass absolute paths to subprocess so the helper script can locate files
        seed_arg = os.path.abspath(args.seed) if not os.path.isabs(args.seed) else args.seed
        stk_path_abs = os.path.abspath(stk_path)
        cmd_sto = [sys.executable, "_makeRfamSeedSto.py", fam_id, seed_arg, "--out_stk", stk_path_abs]
        print(f"Generating {stk_path} ...")
        subprocess.run(cmd_sto, cwd=script_dir, check=True)

        # If target filtering requested, filter generated stk and operate on filtered file
        if args.target and args.target > 0:
            stk_target = os.path.join(stk_dir, f"{fam_id}_{args.target}.stk")
            # remove existing filtered file to avoid appending to old content
            if os.path.exists(stk_target):
                os.remove(stk_target)
            try:
                any_written = filter_stk_by_length(stk_path, args.target, stk_target)
            except Exception as e:
                print(f"Error while filtering {stk_path}: {e}", file=sys.stderr)
                any_written = False

            if any_written:
                print(f"Wrote filtered {stk_target}")
                if not args.only_filter:
                    out_una = os.path.join(una_dir, f"{fam_id}_{args.target}_unaligned.txt")
                    out_ali = os.path.join(ali_dir, f"{fam_id}_{args.target}_aligned.txt")
                    # use absolute paths for helper call
                    stk_target_abs = os.path.abspath(stk_target)
                    out_una_abs = os.path.abspath(out_una)
                    out_ali_abs = os.path.abspath(out_ali)
                    cmd_g4 = [sys.executable, "_make_input_consG4.py", stk_target_abs, out_una_abs, out_ali_abs]
                    print(f"Generating G4 files for {fam_id} (target {args.target}) ...")
                    subprocess.run(cmd_g4, cwd=script_dir, check=True)
                    print(f"Generated {out_una} and {out_ali}")
            else:
                print(f"No blocks with target length {args.target} in {stk_path}; skipping G4 processing.")
        else:
            # no filtering requested: run original behaviour
            print(f"Generating G4 files for {fam_id} ...")
            # pass absolute paths to helper script
            subprocess.run([sys.executable, "_make_input_consG4.py", stk_path_abs, os.path.abspath(una_path), os.path.abspath(ali_path)], cwd=script_dir, check=True)

        processed += 1

    print("All selected families processed.")


if __name__ == "__main__":
    main()
