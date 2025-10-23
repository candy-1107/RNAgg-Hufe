# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path
import chardet
import codecs
# add project scripts dir to sys.path and fallback to ~/pyscript
here = Path(__file__).resolve().parent
proj_scripts = here
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


def detect_encoding(file_path: str, nbytes: int = 4096) -> str:
    """Robust detection with fallbacks. Returns a sane encoding string."""
    p = Path(file_path)
    if not p.exists():
        return "utf-8"
    try:
        with p.open("rb") as f:
            raw = f.read(nbytes)
    except Exception:
        return "utf-8"

    # BOM checks
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"

    # try charset_normalizer first
    try:
        from charset_normalizer import from_bytes

        result = from_bytes(raw)
        if result:
            best = result.best()
            if best and best.encoding:
                return best.encoding
    except Exception:
        pass

    # try chardet fallback
    try:
        res = chardet.detect(raw)
        if res and res.get("encoding"):
            return res.get("encoding")
    except Exception:
        pass

    # last-resort guesses
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except Exception:
            continue
    return "latin-1"


def main(args):
    encoding = detect_encoding(args.raw)
    df = pd.read_table(args.raw, sep=" ", index_col=0, header=None, encoding=encoding)
    max_val = df[1].max()
    min_val = df[1].min()
    for mut in df.index:
        val = df.loc[mut][1]
        norm_val = (val - min_val) / (max_val - min_val)
        print(mut, norm_val)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("raw", help="the act_apt_raw.txt file")
    args = parser.parse_args()

    main(args)
