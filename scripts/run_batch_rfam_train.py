#!/usr/bin/env python3
"""
Batch-run RNAgg training for multiple RFAM families and 4 variants:
  - nuc vs non-nuc
  - aligned vs unaligned

The script finds input txt files under `preprocessing/RfamSeed/rfam_out/rfam_txt/` by default,
or you can pass explicit family IDs. For safety the default is --dry_run which prints commands
instead of executing training (training is usually long).

Example (dry run):
  python scripts/run_batch_rfam_train.py --n 10

Example (actually run training for first 10 families, epoch=200):
  python scripts/run_batch_rfam_train.py --n 10 --dry_run False --epoch 200

Logs are stored in each output directory as train.log
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
from typing import List

DEFAULT_ALIGNED_DIR = os.path.join('preprocessing', 'RfamSeed', 'rfam_out', 'rfam_aligned')
DEFAULT_UNALIGNED_DIR = os.path.join('preprocessing', 'RfamSeed', 'rfam_out', 'rfam_unaligned')
DEFAULT_SCRIPT = os.path.join('scripts', 'RNAgg_train.py')
DEFAULT_OUTROOT = 'results'

SUFFIXES = {
    'unaligned': '_unaligned.txt',
    'aligned': '_aligned.txt'
}

NUC_KEYS = ['non-nuc', 'nuc']
ALIGN_KEYS = ['unaligned', 'aligned']


def find_families_from_dirs(input_dirs: List[str]) -> List[str]:
    """Scan multiple input dirs for txt files and return sorted unique family ids."""
    fams = []
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            continue
        for n in os.listdir(input_dir):
            if not n.lower().endswith('.txt'):
                continue
            base = n[:-4]
            if '_' in base:
                fam = base.split('_')[0]
            else:
                fam = base
            if fam not in fams:
                fams.append(fam)
    fams.sort()
    return fams


def resolve_input_file_for_align(aligned_dir: str, unaligned_dir: str, fam: str, align: str) -> str | None:
    """Return path to input file for a family and alignment mode from the correct directory."""
    suffix = SUFFIXES.get(align)
    if suffix is None:
        return None
    if align == 'aligned':
        input_dir = aligned_dir
    else:
        input_dir = unaligned_dir
    candidates = [os.path.join(input_dir, f"{fam}{suffix}"),
                  os.path.join(input_dir, f"{fam}_{align}.txt"),
                  os.path.join(input_dir, f"{fam}.txt")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def make_outdir(root: str, fam: str, nuc_key: str, align_key: str) -> str:
    path = os.path.join(root, fam, nuc_key, align_key)
    os.makedirs(path, exist_ok=True)
    return path


def build_command(python_exe: str, train_script: str, input_path: str, outdir: str, model_name: str,
                  png_prefix: str, epoch: int, d_rep: int, save_ongoing: int, lr: float, beta: float,
                  nuc_only: bool, act_fname: str | None) -> List[str]:
    cmd = [python_exe, train_script, input_path,
           '--out_dir', outdir,
           '--model_fname', model_name,
           '--png_prefix', png_prefix,
           '--epoch', str(epoch),
           '--d_rep', str(d_rep),
           '--save_ongoing', str(save_ongoing),
           '--lr', str(lr),
           '--beta', str(beta)]
    if nuc_only:
        cmd.append('--nuc_only')
    if act_fname:
        cmd.extend(['--act_fname', act_fname])
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Batch-run RNAgg training for RFAM txt inputs')
    parser.add_argument('--aligned_dir', default=DEFAULT_ALIGNED_DIR, help='directory with aligned txt inputs')
    parser.add_argument('--unaligned_dir', default=DEFAULT_UNALIGNED_DIR, help='directory with unaligned txt inputs')
    parser.add_argument('--script', default=DEFAULT_SCRIPT, help='path to RNAgg_train.py script')
    parser.add_argument('--out_root', default=DEFAULT_OUTROOT, help='root results directory')
    parser.add_argument('--families', help='comma-separated family ids (e.g. RF00001,RF00002)')
    parser.add_argument('--n', type=int, help='take first n families discovered in input_dir')
    parser.add_argument('--epoch', type=int, default=200, help='epochs to train')
    parser.add_argument('--d_rep', type=int, default=8, help='latent dim')
    parser.add_argument('--save_ongoing', type=int, default=0, help='save ongoing every N epochs (0 to disable)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='beta')
    parser.add_argument('--act_fname', default=None, help='optional activity file to enable act model')
    parser.add_argument('--python_exe', default=sys.executable, help='python executable to run training script')
    parser.add_argument('--dry_run', type=lambda x: (str(x).lower() not in ('0','false')), default=True,
                        help='if True (default) only print commands; set to False to actually run')

    args = parser.parse_args()

    # determine families
    if args.families:
        families = [f.strip() for f in args.families.split(',') if f.strip()]
    else:
        fams = find_families_from_dirs([args.aligned_dir, args.unaligned_dir])
        if args.n:
            families = fams[:args.n]
        else:
            families = fams[:10]

    if not families:
        print('No families found or supplied. Check --input_dir or --families.', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(families)} families: {families}')
    print(f"Training script: {args.script}")
    print(f"Aligned inputs dir: {args.aligned_dir}")
    print(f"Unaligned inputs dir: {args.unaligned_dir}")
    print(f"Out root: {args.out_root}")
    print('Dry run mode:' , args.dry_run)

    for fam in families:
        for nuc_key in NUC_KEYS:
            for align_key in ALIGN_KEYS:
                input_path = resolve_input_file_for_align(args.aligned_dir, args.unaligned_dir, fam, align_key)
                if input_path is None:
                    src_dir = args.aligned_dir if align_key == 'aligned' else args.unaligned_dir
                    print(f'[WARN] input for {fam} {align_key} not found under {src_dir}; skipping')
                    continue
                outdir = make_outdir(args.out_root, fam, nuc_key, align_key)
                model_name = f'model_{fam}_{nuc_key}_{align_key}.pth'
                png_prefix = f'{fam}_{nuc_key}_{align_key}_'
                nuc_only = (nuc_key == 'nuc')
                cmd = build_command(args.python_exe, args.script, input_path, outdir, model_name, png_prefix,
                                    args.epoch, args.d_rep, args.save_ongoing, args.lr, args.beta, nuc_only, args.act_fname)
                # log path
                log_path = os.path.join(outdir, 'train.log')
                if args.dry_run:
                    print('DRY:', ' '.join(cmd))
                    print('  -> log:', log_path)
                else:
                    print('RUN:', ' '.join(cmd))
                    print('  -> log:', log_path)
                    with open(log_path, 'w', encoding='utf-8') as logf:
                        try:
                            # run and redirect stdout/stderr to log file
                            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f'[ERROR] training failed for {fam} {nuc_key} {align_key}: returncode={e.returncode}', file=sys.stderr)
                            # continue to next combination

    print('Batch finished.')


if __name__ == '__main__':
    main()
