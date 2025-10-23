# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
import chardet

# add project scripts dir to sys.path and fallback to ~/pyscript
here = Path(__file__).resolve().parent
scripts_dir = (here / ".." / ".." / "scripts").resolve()
proj_scripts = scripts_dir if scripts_dir.exists() else here
if str(proj_scripts) not in sys.path:
    sys.path.insert(0, str(proj_scripts))
fallback = Path.home() / "pyscript"
if str(fallback) not in sys.path and fallback.exists():
    sys.path.append(str(fallback))
# sys.path.append("/home/terai/pyscript")
# import basic
# from Bio import SeqIO
# from Bio.SeqRecord import SeqRecord
# import re
# import csv
# import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# from sklearn import svm
# from scipy.stats import pearsonr
# import RNA


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw = f.read(4096)
    result = chardet.detect(raw)
    return result['encoding']


def main(args: dict):
    encoding = detect_encoding(args.act)
    df = pd.read_table(args.act, index_col=0, header=None, sep=" ", encoding=encoding)
    # template = "TAATACGACTCACTATAGGGTCGNNNNNNNATAATCGCGTGGATATGGCACGCAAGTTTCTACCGGGCACCGTAAATGTCCGACTGGAGCCGTTCGGGCGGCTATAAACAGACCTCAGGCCCGAAGCGTGGCGGCACCTGCCGCCGGTGGTAAAAAAGATCGGAAGAGCACACGTCT".replace('T','U')
    template = "GGGTCGNNNNNNNATAATCGCGTGGATATGGCACGCAAGTTTCTACCGGGCACCGTAAATGTCCGACTGGAGCCGTTCGGGCGGCTATAAACAGACCTCAGGCCCGAAGCGTGGCGGCACCTGCCGCCGGTGGTAAAAAAGATCGGAAGAGCACACGTCT".replace(
        "T", "U"
    )

    for mix in df.index:
        seq = template.replace("NNNNNNN", mix)
        print(f">{mix}")
        print(seq)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("act", help="the act_apta_norm.txt file")
    args = parser.parse_args()

    main(args)
