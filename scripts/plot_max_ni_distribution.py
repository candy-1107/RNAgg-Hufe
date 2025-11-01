#!/usr/bin/env python3
"""
Compute Max-NI (maximum nucleotide identity) of generated sequences against their training sets
and plot the distribution across RFAM families and generator variants.

Definition used:
- For a generated sequence g and each training sequence t, we compute a global edit distance
  (Levenshtein: insertion/deletion/substitution cost = 1). The identity is defined as:
      NI(g, t) = 1 - edit_distance(g, t) / max(len(g), len(t))
- Max-NI of g is max_t NI(g, t) over all training sequences in that family.

Why Levenshtein? It approximates global alignment identity while being robust to small
insertions/deletions and not requiring external tools. It's JIT-accelerated with numba here
for speed. If numba is unavailable, we gracefully fall back to a pure-Python implementation
(which may be slower).

Input locations (relative to repo root):
- Training sequences (unaligned): preprocessing/RfamSeed/rfam_out/rfam_unaligned/{FAM}_unaligned.txt
  Lines look like: "<idx> <sequence> <dot-bracket-structure>". We parse the 2nd token.
  If missing, we fall back to aligned txt and strip '-' gaps.
- Generated sequences (FASTA): output/{variant}/{FAM}.fasta
  Available variants in this script:
    nuc_unaligned, non-nuc_unaligned, nuc_aligned, non-nuc_aligned

Outputs:
- results/max_ni_values.csv  (long-form table: family, variant, seq_id, max_ni, seq_len)
- results/max_ni_boxplot.png (grouped boxplot per family and variant)

Usage (examples):
  python scripts/plot_max_ni_distribution.py
  python scripts/plot_max_ni_distribution.py --families RF00001,RF00002
  python scripts/plot_max_ni_distribution.py --variants nuc_unaligned,non-nuc_unaligned
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import math

import matplotlib.pyplot as plt

# Optional deps: seaborn for nicer plots; numpy+numba for speed
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:  # pragma: no cover
    _HAS_SEABORN = False

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False

try:
    from numba import njit  # type: ignore
    import numba.typed as _nb_typed  # type: ignore
    NumbaList = _nb_typed.List  # type: ignore
    _HAS_NUMBA = True and _HAS_NUMPY  # require numpy for numba path
except Exception:  # pragma: no cover
    _HAS_NUMBA = False
    NumbaList = None  # type: ignore

# ---------------- Paths -----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_UNALIGNED_DIR = PROJECT_ROOT / 'preprocessing' / 'RfamSeed' / 'rfam_out' / 'rfam_unaligned'
TRAIN_ALIGNED_DIR = PROJECT_ROOT / 'preprocessing' / 'RfamSeed' / 'rfam_out' / 'rfam_aligned'
OUTPUT_DIR = PROJECT_ROOT / 'output'
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOT_PATH = RESULTS_DIR / 'max_ni_boxplot.png'
CSV_PATH = RESULTS_DIR / 'max_ni_values.csv'

# Variant label -> subdirectory under output/
VARIANTS = {
    'nuc-una': 'nuc_unaligned',
    'RNAgg-una': 'non-nuc_unaligned',
    'nuc-ali': 'nuc_aligned',
    'RNAgg-ali': 'non-nuc_aligned',
    'RfamGen': 'RfamGen',  # optional; will be ignored if not present
}

# Colors tuned to roughly match the example figure
PALETTE = {
    'nuc-una': '#8dd3c7',
    'RNAgg-una': '#ffffb3',
    'nuc-ali': '#bebada',
    'RNAgg-ali': '#fb8072',
    'RfamGen': '#FDB462',
}

# ---------------- Utilities -----------------
_ALLOWED = set('ACGU')
_TRANS = str.maketrans({'T': 'U', 'N': 'N'})  # T->U, keep N; other letters are removed later

def normalize_seq(s: str) -> str:
    """Uppercase, map T->U, strip gaps/spaces, keep only A/C/G/U (drop others)."""
    s = s.strip().upper().translate(_TRANS)
    s = s.replace('-', '').replace('.', '').replace(' ', '')
    # keep only A/C/G/U; optionally we could keep N and count matches as 0
    return ''.join(ch for ch in s if ch in _ALLOWED)


def read_training_sequences(family: str) -> List[str]:
    """Read training sequences for a family. Prefer unaligned txt; fallback to aligned (strip gaps)."""
    seqs: List[str] = []
    una = TRAIN_UNALIGNED_DIR / f"{family}_unaligned.txt"
    ali = TRAIN_ALIGNED_DIR / f"{family}_aligned.txt"

    path = None
    if una.exists():
        path = una
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Expect: index seq structure
                parts = line.split()
                if len(parts) < 2:
                    continue
                seqs.append(normalize_seq(parts[1]))
    elif ali.exists():
        path = ali
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                seqs.append(normalize_seq(parts[1]))
    else:
        raise FileNotFoundError(f"Training inputs not found for {family}: {una} or {ali}")

    # Filter out empties
    return [s for s in seqs if s]


def read_fasta_sequences(path: Path) -> List[Tuple[str, str]]:
    """Minimal FASTA reader -> list of (id, seq)."""
    if not path.exists():
        return []
    records: List[Tuple[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        header = None
        seq_parts: List[str] = []
        for line in f:
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    seq = normalize_seq(''.join(seq_parts))
                    if seq:
                        records.append((header, seq))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        # last record
        if header is not None:
            seq = normalize_seq(''.join(seq_parts))
            if seq:
                records.append((header, seq))
    return records


# --------------- Levenshtein distance (numba-accelerated) ---------------
if _HAS_NUMBA:
    @njit(cache=True)
    def _levenshtein_uint8(a, b) -> int:
        la = a.shape[0]
        lb = b.shape[0]
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = np.empty(lb + 1, dtype=np.int32)
        curr = np.empty(lb + 1, dtype=np.int32)
        for j in range(lb + 1):
            prev[j] = j
        for i in range(1, la + 1):
            curr[0] = i
            ai = a[i - 1]
            for j in range(1, lb + 1):
                cost = 0 if ai == b[j - 1] else 1
                ins = curr[j - 1] + 1
                dele = prev[j] + 1
                subs = prev[j - 1] + cost
                # min of three
                if ins < dele:
                    m = ins
                else:
                    m = dele
                if subs < m:
                    m = subs
                curr[j] = m
            # swap rows
            tmp = prev
            prev = curr
            curr = tmp
        return int(prev[lb])

    @njit(cache=True)
    def _max_identity_to_training(gen, train_list) -> float:
        best = 0.0
        lg = gen.shape[0]
        for k in range(len(train_list)):
            t = train_list[k]
            lt = t.shape[0]
            d = _levenshtein_uint8(gen, t)
            denom = lg if lg > lt else lt
            ident = 1.0 - (d / denom)
            if ident > best:
                best = ident
        return best
else:
    # Pure-Python fallback (slower)
    def _levenshtein_py(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        curr = [0] * (lb + 1)
        for i in range(1, la + 1):
            curr[0] = i
            ai = a[i - 1]
            for j in range(1, lb + 1):
                cost = 0 if ai == b[j - 1] else 1
                ins = curr[j - 1] + 1
                dele = prev[j] + 1
                subs = prev[j - 1] + cost
                curr[j] = min(ins, dele, subs)
            prev, curr = curr, prev
        return prev[lb]

# mapping A/C/G/U to uint8 codes for numba version
if _HAS_NUMBA:
    _CODE = {ord('A'): np.uint8(0), ord('C'): np.uint8(1), ord('G'): np.uint8(2), ord('U'): np.uint8(3)}

    def seq_to_uint8(s: str):
        arr = np.empty(len(s), dtype=np.uint8)
        for i, ch in enumerate(s):
            arr[i] = _CODE.get(ord(ch), np.uint8(255))  # 255 will never match A/C/G/U
        return arr


# ----------------- Core workflow -----------------

def compute_max_ni_for_family(family: str, variant_map: Dict[str, str]) -> List[Dict[str, object]]:
    """Compute Max-NI for all generated sequences of a family across variants.

    Returns a list of dict rows: {family, variant, seq_id, max_ni, seq_len}
    """
    train_seqs = read_training_sequences(family)
    if not train_seqs:
        print(f"[WARN] No training sequences for {family}", file=sys.stderr)
        return []

    # Prepare training sequences for fast distance computation
    rows: List[Dict[str, object]] = []

    if _HAS_NUMBA:
        train_numba = NumbaList()
        for s in train_seqs:
            train_numba.append(seq_to_uint8(s))
    else:
        train_numba = None  # type: ignore

    for label, subdir in variant_map.items():
        # Support a lowercase fallback directory name for convenience
        fasta_path = OUTPUT_DIR / subdir / f"{family}.fasta"
        if not fasta_path.exists():
            alt = OUTPUT_DIR / subdir.lower() / f"{family}.fasta"
            fasta_path = alt if alt.exists() else fasta_path
        recs = read_fasta_sequences(fasta_path)
        if not recs:
            print(f"[INFO] No generated FASTA for {family} {label}: {fasta_path}")
            continue
        for rid, seq in recs:
            if not seq:
                continue
            if _HAS_NUMBA:
                gen = seq_to_uint8(seq)
                max_ni = float(_max_identity_to_training(gen, train_numba))
            else:
                # slow fallback
                best = 0.0
                Lg = max(1, len(seq))
                for ts in train_seqs:
                    d = _levenshtein_py(seq, ts)
                    denom = max(Lg, len(ts))
                    ident = 1.0 - (d / denom)
                    if ident > best:
                        best = ident
                max_ni = float(best)
            rows.append({
                'family': str(family),
                'variant': str(label),
                'seq_id': str(rid),
                'max_ni': float(max_ni),
                'seq_len': int(len(seq)),
            })
    return rows


def plot_boxplot(rows: List[Dict[str, object]], families: List[str], variant_order: List[str]) -> None:
    if not rows:
        print('[ERROR] No data to plot.')
        return

    # Convert to arrays grouped for plotting
    # data[fam][variant] -> list of values
    data: Dict[str, Dict[str, List[float]]] = {fam: {v: [] for v in variant_order} for fam in families}
    for r in rows:
        fam = str(r['family'])
        var = str(r['variant'])
        val = float(r['max_ni'])
        if fam in data and var in data[fam]:
            data[fam][var].append(val)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if _HAS_SEABORN:
        import pandas as pd  # pandas is not strictly required for fallback plotting
        df = []
        for fam in families:
            for var in variant_order:
                for v in data[fam][var]:
                    df.append({'family': fam, 'variant': var, 'max_ni': float(v)})
        df = pd.DataFrame(df)
        plt.style.use('seaborn-v0_8-paper')
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.boxplot(
            data=df,
            x='family', y='max_ni', hue='variant',
            hue_order=variant_order,
            palette=PALETTE,
            fliersize=0, linewidth=1.2, ax=ax,
        )
        ax.set_xlabel('')
        ax.set_ylabel('Max-NI')
        ax.set_ylim(0.3, 1.05)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='', loc='upper left')
        plt.tight_layout()
        fig.savefig(PLOT_PATH, dpi=300)
        plt.close(fig)
    else:
        # Plain matplotlib grouped boxplots (no numpy dependency)
        fig, ax = plt.subplots(figsize=(15, 6))
        n_groups = len(families)
        n_vars = len(variant_order)
        box_width = 0.15
        group_width = n_vars * box_width + 0.1
        x_positions = [gi * (group_width + 0.35) for gi in range(n_groups)]
        for vi, var in enumerate(variant_order):
            xs = []
            series = []
            for gi, fam in enumerate(families):
                xs.append(x_positions[gi] + vi * box_width)
                series.append(data[fam][var] if data[fam][var] else [math.nan])
            # Create a boxplot per family for this variant
            bp = ax.boxplot(series, positions=xs, widths=box_width * 0.9, patch_artist=True,
                            showfliers=False)
            color = PALETTE.get(var, None)
            for patch in bp['boxes']:
                if color:
                    patch.set_facecolor(color)
            for median in bp['medians']:
                median.set_color('black')
        # X ticks
        ax.set_xticks([x_positions[gi] + (n_vars - 1) * box_width / 2 for gi in range(n_groups)])
        ax.set_xticklabels(families, rotation=0)
        ax.set_ylabel('Max-NI')
        ax.set_ylim(0.3, 1.05)
        # Legend
        handles = [plt.Line2D([0], [0], color=PALETTE.get(v, 'gray'), lw=10) for v in variant_order]
        ax.legend(handles, variant_order, title='')
        plt.tight_layout()
        fig.savefig(PLOT_PATH, dpi=300)
        plt.close(fig)

    print(f"Saved plot -> {PLOT_PATH}")


def save_csv(rows: List[Dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Save a simple CSV without pandas
    with open(CSV_PATH, 'w', encoding='utf-8') as f:
        f.write('family,variant,seq_id,max_ni,seq_len\n')
        for r in rows:
            fam = str(r['family'])
            var = str(r['variant'])
            rid = str(r['seq_id'])
            max_ni = float(r['max_ni'])
            slen = int(r.get('seq_len', 0))
            f.write(f"{fam},{var},{rid},{max_ni:.6f},{slen}\n")
    print(f"Saved CSV -> {CSV_PATH}")


# ----------------- CLI -----------------

def main():
    parser = argparse.ArgumentParser(description='Compute Max-NI for generated sequences and plot distribution.')
    parser.add_argument('--families', type=str,
                        default=','.join([f"RF{i:05d}" for i in range(1, 11)]),
                        help='Comma-separated RFAM IDs (default: RF00001..RF00010)')
    parser.add_argument('--variants', type=str,
                        default=','.join(VARIANTS.keys()),
                        help='Comma-separated variant labels (use keys: ' + ','.join(VARIANTS.keys()) + ')')
    parser.add_argument('--min_len', type=int, default=1, help='Ignore sequences shorter than this length')
    args = parser.parse_args()

    families = [x.strip() for x in args.families.split(',') if x.strip()]
    var_labels = [x.strip() for x in args.variants.split(',') if x.strip()]
    variant_map = {k: VARIANTS[k] for k in var_labels if k in VARIANTS}
    if not variant_map:
        print('[ERROR] No valid variants specified.', file=sys.stderr)
        sys.exit(1)

    all_rows: List[Dict[str, object]] = []
    print('--- Computing Max-NI ---')
    for fam in families:
        try:
            rows = compute_max_ni_for_family(fam, variant_map)
            # Honor min_len by filtering on the generated sequence length
            if args.min_len > 1:
                rows = [r for r in rows if int(r.get('seq_len', 0)) >= args.min_len]
            all_rows.extend(rows)
        except Exception as e:
            print(f"[ERROR] {fam}: {e}", file=sys.stderr)

    if not all_rows:
        print('[ERROR] No data collected; nothing to plot.', file=sys.stderr)
        sys.exit(1)

    # Save CSV and plot
    save_csv(all_rows)
    plot_boxplot(all_rows, families, list(variant_map.keys()))


if __name__ == '__main__':
    main()
