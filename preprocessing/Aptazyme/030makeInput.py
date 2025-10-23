# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path
import codecs

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


# helper: lightweight encoding detection when utils_gg is not available
def detect_encoding(path: str, nbytes: int = 4096) -> str:
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

    # try charset_normalizer
    try:
        from charset_normalizer import from_bytes

        result = from_bytes(raw)
        if result:
            best = result.best()
            if best and best.encoding:
                return best.encoding
    except Exception:
        pass

    # try chardet
    try:
        import chardet

        res = chardet.detect(raw)
        if res and res.get("encoding"):
            return res.get("encoding")
    except Exception:
        pass

    # fallback to common encodings
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except Exception:
            continue
    return "latin-1"


def main(args):
    sid2act = {}
    try:
        import utils_gg as utils
    except Exception:
        utils = None
    if utils is not None:
        fh = utils.open_text(args.act, "r")
    else:
        enc = detect_encoding(args.act)
        fh = open(args.act, "r", encoding=enc)
    with fh as f:
        for line in f:
            line = line.replace("\n", "")
            sid, act = line.split(" ")
            try:
                act = float(act)
            except Exception:
                try:
                    act = float(act.replace(",", "."))
                except Exception:
                    act = float('nan')
            sid2act[sid] = act

    if utils is not None:
        fh2 = utils.open_text(args.ss, "r")
    else:
        enc2 = detect_encoding(args.ss)
        fh2 = open(args.ss, "r", encoding=enc2)
    with fh2 as f:
        for line in f:
            line = line.replace("\n", "")
            if not line:
                continue
            if line[0] == ">":
                sid = line[1:]
                seq = next(f).replace("\n", "")
                items = next(f).replace("\n", "").split(" ")
                ss = items[0]
                if sid in sid2act:
                    print(sid, seq, ss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("ss", help="sequence and ss predicted by centroidfold")
    parser.add_argument("act", help="activity information")
    args = parser.parse_args()

    main(args)
