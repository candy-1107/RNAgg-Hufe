# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess
import re

def get_all_fam_ids(seed_path):
    fam_ids = []
    with open(seed_path, encoding='iso-8859-1') as f:
        for line in f:
            if line.startswith('#=GF AC'):
                fam_id = line.split()[2].strip()
                fam_ids.append(fam_id)
    return fam_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, help='Rfam.seed file path')
    parser.add_argument('--output_stk', required=True, help='Output folder for .stk files')
    parser.add_argument('--output_g4', required=True, help='Output folder for G4 files')
    args = parser.parse_args()

    os.makedirs(args.output_stk, exist_ok=True)
    os.makedirs(args.output_g4, exist_ok=True)

    script_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
    fam_ids = get_all_fam_ids(args.seed)
    print(f"Found {len(fam_ids)} families.")

    for fam_id in fam_ids:
        stk_path = os.path.join(args.output_stk, f"{fam_id}.stk")
        una_path = os.path.join(args.output_g4, f"{fam_id}_unaligned.txt")
        ali_path = os.path.join(args.output_g4, f"{fam_id}_aligned.txt")
        # Step 1: 生成 .stk 文件
        cmd_sto = [sys.executable, '_makeRfamSeedSto.py', fam_id, args.seed, '--out_stk', stk_path]
        print(f"Generating {stk_path} ...")
        subprocess.run(cmd_sto, cwd=script_dir, check=True)
        # Step 2: 生成 G4语法相关文件
        cmd_g4 = [sys.executable, '_make_input_consG4.py', stk_path, una_path, ali_path]
        print(f"Generating G4 files for {fam_id} ...")
        subprocess.run(cmd_g4, cwd=script_dir, check=True)
    print("All families processed.")

if __name__ == '__main__':
    main()
