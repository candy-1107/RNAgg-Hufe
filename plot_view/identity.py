# def calc_nucleotide_identity(sto_file):
#     rf_seq = ""  # Rfam 参考序列（#=GC RF 行）
#     gen_seqs = {}  # 生成序列：{序列ID: 序列}
#
#     # 读取 STO 文件，提取核心序列
#     with open(sto_file, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("//"):
#                 continue
#             # 提取 Rfam 参考序列（#=GC RF）
#             if line.startswith("#=GC RF"):
#                 rf_seq = line.split(maxsplit=2)[2].upper()  # 转大写，统一比较
#             # 提取生成序列（Generated_ 开头）
#             elif line.startswith("Generated_"):
#                 seq_id, seq = line.split(maxsplit=1)
#                 gen_seqs[seq_id] = seq.upper()  # 转大写，统一比较
#
#     # 校验必要数据
#     if not rf_seq:
#         raise ValueError("STO 文件中未找到 #=GC RF 参考序列行")
#     if not gen_seqs:
#         raise ValueError("STO 文件中未找到 Generated_ 开头的生成序列")
#
#     # 计算每条生成序列的一致性
#     for seq_id, seq in gen_seqs.items():
#         match = 0
#         total = 0
#         # 按位点一一比对（STO 格式保证序列长度一致）
#         for rf_base, gen_base in zip(rf_seq, seq):
#             # 有效位点：双方都不是 gap（-/.），且是有效碱基（A/U/G/C）
#             if rf_base not in "-." and gen_base not in "-.":
#                 if rf_base in "AUGC" and gen_base in "AUGC":
#                     total += 1
#                     if rf_base == gen_base:
#                         match += 1
#         # 计算一致性（避免除零）
#         identity = (match / total) * 100 if total > 0 else 0.0
#         print(f"{seq_id}\t一致性: {identity:.2f}%")
#
#
# # 调用函数（替换为你的 STO 文件路径）
# calc_nucleotide_identity("../data/gen_sto/RF00001.sto")


import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 第一步：计算全局核苷酸一致性 --------------------------
sto_path = "../data/gen_sto/RF00001.sto"  # 替换为你的STO文件路径
rf_reference = ""
generated_seqs = {}

# 提取并拼接全局序列
with open(sto_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        # 拼接Rfam参考序列（处理多行拆分）
        if line.startswith("#=GC RF"):
            rf_reference += line.split(maxsplit=2)[2].upper()
        # 拼接生成序列（处理多行拆分）
        elif line.startswith("Generated_"):
            seq_id, seq_part = line.split(maxsplit=1)
            generated_seqs[seq_id] = generated_seqs.get(seq_id, "") + seq_part.upper()

# 计算每条序列的一致性并存储
identity_dict = {}
for seq_id, full_seq in generated_seqs.items():
    match = 0
    total = 0
    for rf_base, gen_base in zip(rf_reference, full_seq):
        if rf_base not in "-." and gen_base not in "-." and rf_base in "AUGC" and gen_base in "AUGC":
            total += 1
            if rf_base == gen_base:
                match += 1
    identity_dict[seq_id] = (match / total) * 100 if total > 0 else 0.0

# -------------------------- 第二步：生成可视化视图 --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # 适配英文标签，避免乱码
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  # 2行1列布局，上下两个图

# 1. 柱状图：每条序列的一致性对比
seq_ids = list(identity_dict.keys())
identities = list(identity_dict.values())

# 优化柱状图可读性（序列太多时x轴标签旋转）
ax1.bar(range(len(seq_ids)), identities, color='#2E86AB', alpha=0.7, edgecolor='#1A5276')
ax1.set_xlabel('Generated Sequences', fontsize=12)
ax1.set_ylabel('Nucleotide Identity (%)', fontsize=12)
ax1.set_title('Global Nucleotide Identity of Each Generated Sequence', fontsize=14, pad=20)
ax1.set_xticks(range(len(seq_ids)))
ax1.set_xticklabels([id.split('_')[1] for id in seq_ids], rotation=45, ha='right')  # x轴只显示序号（如998、999）
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 100)  # 一致性范围固定0-100%

# 2. 直方图：一致性分布趋势（看整体集中情况）
ax2.hist(identities, bins=15, color='#A23B72', alpha=0.7, edgecolor='#6B2446')
ax2.set_xlabel('Nucleotide Identity (%)', fontsize=12)
ax2.set_ylabel('Number of Sequences', fontsize=12)
ax2.set_title('Distribution of Global Nucleotide Identities', fontsize=14, pad=20)
ax2.grid(axis='y', alpha=0.3)
ax2.axvline(np.mean(identities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(identities):.1f}%')
ax2.legend()

# 调整布局，避免标签重叠
plt.tight_layout()

# 保存图片（推荐PNG格式，高清无失真）或直接显示
plt.savefig('../data/identity_visualization.png', dpi=300, bbox_inches='tight')
plt.show()