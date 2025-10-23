import os
import sys
import torch
import codecs
from pathlib import Path


def detect_encoding(path: str, nbytes: int = 4096) -> str:
    """
    Try to detect encoding by inspecting BOM and by trying a list of common encodings.
    This is a lightweight fallback when charset detection libs are not available.
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

    # Prefer external libraries for detection if available
    try:
        # charset_normalizer is more modern and often preinstalled
        from charset_normalizer import from_bytes

        result = from_bytes(raw)
        if result:
            best = result.best()
            if best and best.encoding:
                return best.encoding
    except Exception:
        pass
    try:
        import chardet

        res = chardet.detect(raw)
        if res and res.get("encoding"):
            return res.get("encoding")
    except Exception:
        pass

    # Try common encodings as last resort
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except Exception:
            continue
    return "latin-1"


def open_text(
    path: str,
    mode: str = "r",
    encoding: str = None,
    errors: str = "strict",
    newline=None,
):
    """
    Open text file with automatic encoding detection when encoding is not provided.
    Keeps binary modes untouched.
    """
    # If binary mode, delegate to built-in open
    if "b" in mode:
        return open(path, mode)

    if encoding is None:
        enc = detect_encoding(path)
    else:
        enc = encoding
    return open(path, mode, encoding=enc, errors=errors, newline=newline)


def readInput(fname: str):
    d_seq = {}
    d_ss = {}
    with open_text(fname) as f:
        for line in f:
            line = line.rstrip("\n")
            items = line.split()
            if len(items) < 3:
                continue
            sid, seq, ss = items[0:3]
            d_seq[sid] = seq
            d_ss[sid] = ss
    return d_seq, d_ss


def get_token2idx(nuc_letters):  # Letterと行番号の辞書を作成
    d = {}
    for i, x in enumerate(nuc_letters):
        d[x] = i
    return d


def readAct(fname):
    sid2act = {}
    with open_text(fname) as f:
        for line in f:
            line = line.rstrip("\n")
            items = line.split()
            if len(items) < 2:
                continue
            sid, act = items[0], items[1]
            try:
                act = float(act)
            except Exception:
                try:
                    act = float(act.replace(",", "."))
                except Exception:
                    act = float("nan")
            sid2act[sid] = act
    return sid2act


class Dataset:
    def __init__(self, input_mat, sid_list, act_list):  # input_data is list of tensor
        self.data = input_mat
        self.sid_list = sid_list
        self.act_list = torch.tensor(act_list, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index], self.sid_list[index], self.act_list[index]

    def __len__(self):
        return len(self.data)
