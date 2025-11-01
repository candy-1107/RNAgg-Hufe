#!/usr/bin/env python3
"""
Batch-generate sequences using trained RNAgg models for multiple RFAM families and 4 variants:
  - nuc vs non-nuc
  - aligned vs unaligned

This script looks for models under `results/<family>/<nuc|non-nuc>/<aligned|unaligned>/` with
names like `model_<family>_<nuc|non-nuc>_<aligned|unaligned>.pth` by default. For each found model it
calls `scripts/RNAgg_generate.py` to produce sequences and writes logs to the same output directory.

Usage (dry-run prints commands):
  python scripts/run_batch_rfam_generate.py --n_families 10 --num 100

To actually run generation (careful: may be long):
  python scripts/run_batch_rfam_generate.py --n_families 10 --num 100 --dry_run False

"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
from typing import List

DEFAULT_MODELS_ROOT = 'results'
DEFAULT_OUTPUT_ROOT = 'output'
DEFAULT_GEN_SCRIPT = os.path.join('scripts', 'RNAgg_generate.py')
NUC_KEYS = ['non-nuc', 'nuc']
ALIGN_KEYS = ['unaligned', 'aligned']


def find_families_in_results(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    fams = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            fams.append(name)
    fams.sort()
    return fams


def find_model_for_combo(root: str, fam: str, nuc_key: str, align_key: str) -> str | None:
    dirpath = os.path.join(root, fam, nuc_key, align_key)
    if not os.path.isdir(dirpath):
        return None
    expected_name = f'model_{fam}_{nuc_key}_{align_key}.pth'
    expected_path = os.path.join(dirpath, expected_name)
    if os.path.isfile(expected_path):
        return expected_path
    # fallback: look for any .pth in dir
    for entry in os.listdir(dirpath):
        if entry.lower().endswith('.pth'):
            return os.path.join(dirpath, entry)
    return None


def build_command(python_exe: str, gen_script: str, num: int, model_path: str, outfile: str, s_bat: int, n_cpu: int, out_fasta: bool, from_emb: bool) -> List[str]:
    cmd = [python_exe, gen_script, str(num), model_path, outfile, '--s_bat', str(s_bat), '--n_cpu', str(n_cpu)]
    if out_fasta:
        cmd.append('--out_fasta')
    if from_emb:
        cmd.append('--from_emb')
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Batch-generate RNA sequences using trained models')
    parser.add_argument('--models_root', default=DEFAULT_MODELS_ROOT, help='root directory where trained models are stored (default: results)')
    parser.add_argument('--out_root', default=DEFAULT_OUTPUT_ROOT, help='root directory to write generated fasta files (default: output)')
    parser.add_argument('--gen_script', default=DEFAULT_GEN_SCRIPT, help='path to RNAgg_generate.py')
    parser.add_argument('--families', help='comma-separated family ids to process')
    parser.add_argument('--n_families', type=int, help='take first N families found under models_root (default 10)')
    parser.add_argument('--num', type=int, default=100, help='number of sequences to generate per model')
    parser.add_argument('--s_bat', type=int, default=100, help='generation batch size (s_bat)')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of CPU to use for generation (RNAgg_generate --n_cpu)')
    # default: write fasta files into output/<category>/ by default; user can disable with --no_out_fasta
    parser.add_argument('--out_fasta', dest='out_fasta', action='store_true', help='output fasta from generator')
    parser.add_argument('--no_out_fasta', dest='out_fasta', action='store_false', help='do not output fasta')
    parser.set_defaults(out_fasta=True)
    parser.add_argument('--from_emb', action='store_true', help='pass embeddings file to generator (then --num is treated as path)')
    parser.add_argument('--python_exe', default=sys.executable, help='python executable to run generation script')
    parser.add_argument('--dry_run', type=lambda x: (str(x).lower() not in ('0','false')), default=True, help='if True only print commands')

    args = parser.parse_args()

    # Ensure models root and output root exist (create if missing)
    for _dir in (args.models_root, args.out_root):
        try:
            if _dir:
                os.makedirs(_dir, exist_ok=True)
        except Exception:
            # If creation fails, continue and let later checks/reporting handle it
            pass

    if args.families:
        families = [f.strip() for f in args.families.split(',') if f.strip()]
    else:
        fams = find_families_in_results(args.models_root)
        if args.n_families:
            families = fams[:args.n_families]
        else:
            families = fams[:10]

    if not families:
        print('No families found under', args.models_root, file=sys.stderr)
        sys.exit(1)

    print('Families to process:', families)
    print('Generator script:', args.gen_script)
    print('Models root:', args.models_root)
    print('Output root:', args.out_root)
    print('Dry run:', args.dry_run)

    for fam in families:
        for nuc_key in NUC_KEYS:
            for align_key in ALIGN_KEYS:
                model_path = find_model_for_combo(args.models_root, fam, nuc_key, align_key)
                if not model_path:
                    print(f'[WARN] model for {fam} {nuc_key} {align_key} not found; skipping')
                    continue
                # decide outfile and outdir: write into output/<category>/ with one fasta per family
                # categories: non-nuc_unaligned, non-nuc_aligned, nuc_unaligned, nuc_aligned
                if nuc_key == 'nuc' and align_key == 'unaligned':
                    category = 'nuc_unaligned'
                elif nuc_key == 'nuc' and align_key == 'aligned':
                    category = 'nuc_aligned'
                elif nuc_key == 'non-nuc' and align_key == 'unaligned':
                    category = 'non-nuc_unaligned'
                else:
                    category = 'non-nuc_aligned'

                out_root = args.out_root if args.out_root else DEFAULT_OUTPUT_ROOT
                outdir = os.path.join(out_root, category)
                os.makedirs(outdir, exist_ok=True)
                # one fasta per family named <family>.fasta (unless out_fasta disabled)
                outfile = os.path.join(outdir, f'{fam}.fasta' if args.out_fasta else f'generated_{fam}_{nuc_key}_{align_key}.txt')
                logpath = os.path.join(outdir, f'generate_{fam}_{nuc_key}_{align_key}.log')
                cmd = build_command(args.python_exe, args.gen_script, args.num, model_path, outfile, args.s_bat, args.n_cpu, args.out_fasta, args.from_emb)
                if args.dry_run:
                    print('DRY:', ' '.join(cmd))
                    print('  -> log:', logpath)
                else:
                    print('RUN:', ' '.join(cmd))
                    print('  -> log:', logpath)
                    with open(logpath, 'w', encoding='utf-8') as lf:
                        try:
                            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f'[ERROR] generation failed for {fam} {nuc_key} {align_key}: returncode={e.returncode}', file=sys.stderr)

    print('Batch generation finished.')


if __name__ == '__main__':
    main()
