#!/usr/bin/env python3
"""
A lightweight replacement for `mxfold2 predict` for use when mxfold2 is unavailable.
This script implements the Nussinov algorithm to predict an RNA secondary structure (dot-bracket).

Usage:
    python mxfold2_predict.py predict input.fa > ss_stub.txt

Output format (per sequence):
    >seq_id
    SEQUENCE
    DOT-BRACKET

Limitations:
- This is a simple DP-based predictor (Nussinov) and is NOT as accurate as mxfold2.
- No energy model; base-pair scoring is uniform and GU pairs are allowed.

"""
import sys
import argparse
from typing import List, Tuple

PAIRS = {('A','U'),('U','A'),('C','G'),('G','C'),('G','U'),('U','G')}

def read_fasta(path: str) -> List[Tuple[str,str]]:
    records = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            header = None
            seq_lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None:
                        records.append((header, ''.join(seq_lines)))
                    header = line[1:]
                    seq_lines = []
                else:
                    seq_lines.append(line.upper())
            if header is not None:
                records.append((header, ''.join(seq_lines)))
    except UnicodeDecodeError:
        # try with latin-1 as a fallback
        with open(path, 'r', encoding='latin-1') as f:
            header = None
            seq_lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None:
                        records.append((header, ''.join(seq_lines)))
                    header = line[1:]
                    seq_lines = []
                else:
                    seq_lines.append(line.upper())
            if header is not None:
                records.append((header, ''.join(seq_lines)))
    return records


def nussinov(seq: str) -> str:
    n = len(seq)
    if n == 0:
        return ''
    dp = [[0]*(n) for _ in range(n)]
    trace = [[None]*n for _ in range(n)]
    # fill
    for l in range(1, n):  # length of subsequence
        for i in range(0, n-l):
            j = i + l
            best = dp[i][j-1]
            trace_choice = ('skip_j', None)
            # pair j with k
            for k in range(i, j):
                if (seq[k], seq[j]) in PAIRS:
                    left = dp[i][k-1] if k-1 >= i else 0
                    right = dp[k+1][j-1] if k+1 <= j-1 else 0
                    val = left + 1 + right
                    if val > best:
                        best = val
                        trace_choice = ('pair', k)
            # bifurcation
            for k in range(i, j):
                val = dp[i][k] + dp[k+1][j]
                if val > best:
                    best = val
                    trace_choice = ('bifur', k)
            dp[i][j] = best
            trace[i][j] = trace_choice
    # reconstruct
    brackets = ['.']*n
    def backtrack(i: int, j: int):
        if i >= j:
            return
        choice = trace[i][j]
        if choice is None:
            return
        typ, k = choice
        if typ == 'skip_j':
            backtrack(i, j-1)
        elif typ == 'pair':
            # pair k with j
            brackets[k] = '('
            brackets[j] = ')'
            # left
            if k-1 >= i:
                backtrack(i, k-1)
            if k+1 <= j-1:
                backtrack(k+1, j-1)
        elif typ == 'bifur':
            backtrack(i, k)
            backtrack(k+1, j)

    backtrack(0, n-1)
    return ''.join(brackets)


def predict_file(path: str):
    records = read_fasta(path)
    out_lines = []
    for header, seq in records:
        struct = nussinov(seq)
        out_lines.append(f">{header}")
        out_lines.append(seq)
        out_lines.append(struct)
    sys.stdout.write('\n'.join(out_lines))


def main():
    parser = argparse.ArgumentParser(description='mxfold2-like stub predictor (Nussinov)')
    parser.add_argument('command', choices=['predict'], help='command: predict')
    parser.add_argument('input', help='input fasta file')
    args = parser.parse_args()
    if args.command == 'predict':
        predict_file(args.input)

if __name__ == '__main__':
    main()

