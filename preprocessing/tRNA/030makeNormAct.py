# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path

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
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import svm
# from scipy.stats import pearsonr
# import RNA
import numpy as np

try:
    import utils_gg as utils
except Exception:
    utils = None


def main(args: dict):

    sid_list = []
    fit_list = []
    if utils is not None:
        fh = utils.open_text(args.data, "r")
    else:
        fh = open(args.data, "r", encoding="utf-8")
    with fh as f:
        for line in f:
            line = line.replace("\n", "")
            if line[0] == "#" or line[0:3] == "Num":
                continue
            else:
                num, pos, nuc, seq, fit, perc2 = line.split("\t")
                sid = "-".join([num, pos, nuc])
                sid = sid.replace(" ", "+")
                # print(f"{sid} {fit}")
                sid_list.append(sid)
                fit_list.append(float(fit))

    max_fit = np.max(fit_list)
    min_fit = np.min(fit_list)

    norm_fit_list = [(x - min_fit) / (max_fit - min_fit) for x in fit_list]

    # max_fit = np.max(norm_fit_list)
    # min_fit = np.min(norm_fit_list)
    # print(max_fit, min_fit)

    for sid, norm_fit in zip(sid_list, norm_fit_list):
        print(sid, norm_fit)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="The FitnessData.txt file")
    args = parser.parse_args()

    main(args)
