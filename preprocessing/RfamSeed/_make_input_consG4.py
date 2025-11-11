# -*- coding: utf-8 -*-

import os
import sys
import argparse
import re
import numpy as np
from Bio import AlignIO
import chardet

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
)
# sys.path.append(os.environ['HOME'] + "/pyscript")
sys.path.append(os.path.join(os.path.expanduser("~"), "pyscript"))
# sys.path.append("/home/terai/pyscript")

# from SS2shape2 import generate_rule_G4b, SCFGParseError
from SS2shape3 import generate_rule_G4b, SCFGParseError, getBPpos_ij


def getBP(ss: list):
    """
    解析二级结构字符串，返回碱基对列表。
    """
    bp_list = []
    stack = []
    for i, s in enumerate(ss):
        if s in "(<{[":
            stack.append(i)
        elif s in ")>}]":
            left = stack.pop()
            bp_list.append((left, i))
        elif s in ".,:_-~":
            pass
        elif s in "AaBb":  # pseudo knot
            pass
        else:
            print(f"Unexpected SS annotation ({s})", file=sys.stderr)
            exit(0)
    return bp_list


# 合法的碱基对
valid_pair = ["AU", "UA", "GC", "CG", "GU", "UG"]


def main(args: dict):
    """
    处理Stockholm格式文件，输出未对齐和对齐的序列及二级结构。
    """
    # 先打开输出文件
    with open(args.outfile_una, mode="w") as f_una, open(
            args.outfile_ali, mode="w"
    ) as f_ali:
        # 读取Stockholm文件,自动检测编码格式,写入utf-8临时文件，再用AlignIO读取
        def detect_encoding(file_path):
            with open(file_path, "rb") as f:
                raw = f.read(4096)
            result = chardet.detect(raw)
            return result["encoding"]

        encoding = detect_encoding(args.stk)
        with open(args.stk, encoding=encoding) as handle:
            align = AlignIO.read(handle, "stockholm")

        SS = align.column_annotations["secondary_structure"]
        SS_list = list(SS)

        # 获取共通二级结构的碱基对
        bp_list = getBP(SS_list)

        num_pairs = {}
        sid = 0
        max_len, min_len = -1e10, 1e10
        save_nr_set = set()

        for record in align:
            # 检查是否有非法碱基
            pattern = r"[^AUGC-]"
            seq_str = str(record.seq).upper()
            matches = re.findall(pattern, seq_str)
            if matches:
                print(
                    f"{record.id} contains invalid letters {matches}.", file=sys.stderr
                )
                continue

            seq_list = list(seq_str)

            # 检查bp_list中的碱基对是否合法
            bp_list_pass = []
            for bp in bp_list:
                pair = seq_list[bp[0]] + seq_list[bp[1]]
                if pair in valid_pair:
                    bp_list_pass.append(bp)
                    num_pairs[pair] = num_pairs.get(pair, 0) + 1

            # 构建新的二级结构注释
            SS_list_pass = ["." for _ in SS]
            for bp in bp_list_pass:
                SS_list_pass[bp[0]] = "("
                SS_list_pass[bp[1]] = ")"

            seq_nr, ss_nr = [], []
            for i, base in enumerate(seq_list):
                if base != "-":
                    seq_nr.append(base)
                    if SS_list_pass[i] in "().":
                        ss_nr.append(SS_list_pass[i])
                    else:
                        print(
                            f"Unexpected SS annotation ({SS_list_pass[i]})",
                            file=sys.stderr,
                        )
                        exit(0)

            seq_str = "".join(seq_nr)
            ss_str = "".join(ss_nr)
            ali_seq_str = "".join(seq_list)
            ali_ss_str = "".join(SS_list_pass)

            # 检查G4 grammar解析
            bp = getBPpos_ij(ss_str)
            rule = []
            try:
                generate_rule_G4b(
                    0, rule, seq_str.lower(), ss_str, bp, (0, len(ss_str) - 1), "S"
                )
            except SCFGParseError:
                print(f"SCFGParse Error found in {record.id}.", file=sys.stderr)
                continue

            max_len = max(max_len, len(seq_str))
            min_len = min(min_len, len(seq_str))

            if seq_str in save_nr_set:
                print(f"Duplication of sequence {record.id}.", file=sys.stderr)
                continue
            save_nr_set.add(seq_str)

            print(f"{sid} {seq_str} {ss_str}", file=f_una)
            print(f"{sid} {ali_seq_str} {ali_ss_str}", file=f_ali)
            sid += 1

        # 输出所有碱基对的统计
        ordered = sorted(num_pairs.items(), key=lambda x: x[1], reverse=True)
        sum_pairs = np.sum(list(num_pairs.values()))
        for pair, num in ordered:
            print(pair, num / sum_pairs, file=sys.stderr)

        print(
            f"max len = {max_len}, min len = {min_len}, file = {args.stk}",
            file=sys.stderr,
        )
    print(
        f"max len = {max_len}, min len = {min_len}, file = {args.stk}", file=sys.stderr
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Stockholm file and output unaligned/aligned sequence and structure."
    )
    parser.add_argument("stk", help="stockholm format file")
    parser.add_argument("outfile_una", help="unaligned input file name")
    parser.add_argument("outfile_ali", help="aligned input file name")
    # parser.add_argument('--aligned', action='store_true', help='make aligned sequence data')
    args = parser.parse_args()
    main(args)
