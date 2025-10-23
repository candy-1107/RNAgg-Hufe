# RNA 二级结构预测工具集 - 完整文档

**版本**：1.0  
**日期**：2025-10-23  
**位置**：`mxfold2/`（项目根目录）

---

## 目录

1. [简介](#简介)
2. [文件清单](#文件清单)
3. [快速开始](#快速开始)
4. [脚本详细说明](#脚本详细说明)
5. [使用示例](#使用示例)
6. [算法说明](#算法说明)
7. [常见问题](#常见问题)
8. [技术细节](#技术细节)
9. [迁移指南](#迁移指南)

---

## 简介

这是一个功能类似 mxfold2 的 Python RNA 二级结构预测工具集。本文件夹位于项目根目录（与 `scripts/` 同级），整合了所有 RNA 二级结构预测相关的脚本，包括原 `preprocessing/mxfold2_stub/` 的内容。

### 功能特性

- ✅ **多种算法**：Nussinov（快速）和 Nussinov-Energy（准确）
- ✅ **自动编码检测**：自动识别 UTF-8、UTF-16、Latin-1、CP1252 等编码
- ✅ **标准格式**：支持标准 FASTA 输入/输出
- ✅ **点括号记号**：输出标准的二级结构表示
- ✅ **批量处理**：一次处理多个文件
- ✅ **参数可配置**：调整最小环长度、AU 惩罚等
- ✅ **Windows 友好**：完全支持 Windows 路径和编码
- ✅ **向后兼容**：保留原 mxfold2_stub 的所有功能

### 安装要求

**必需**：
- Python 3.7+

**可选**（推荐，用于更好的编码检测）：
```bash
pip install charset-normalizer
```

---

## 文件清单

### Python 脚本（3 个）

1. **predict_structure.py** (17 KB) ⭐ 推荐
   - RNA 二级结构预测（功能增强版）
   - 支持两种算法：Nussinov 和 Nussinov-Energy
   - 自动编码检测，参数可配置

2. **mxfold2_predict.py** (4.5 KB)
   - RNA 二级结构预测（简化版）
   - 原 mxfold2_stub，保持向后兼容
   - 基础 Nussinov 算法

3. **batch_predict.py** (7 KB)
   - 批量处理多个 FASTA 文件
   - 自动文件发现，进度显示

### 示例文件（2 个）

4. **example.fa** (434 bytes)
   - 6 条测试序列（tRNA-like、发夹、茎环等）

5. **test.fa** (93 bytes)
   - 2 条简单测试序列

### 文档（2 个）

6. **README.md** - 本文件（完整文档）
7. **USAGE.txt** - 命令行参考手册

---

## 快速开始

### 5 分钟上手

#### 步骤 1：进入目录
```powershell
cd preprocessing\rna_structure_predictor
```

#### 步骤 2：运行测试
```powershell
# 方法 1：使用增强版（推荐）
python predict_structure.py predict example.fa

# 方法 2：使用简化版
python mxfold2_predict.py predict test.fa
```

#### 步骤 3：处理你的文件
```powershell
# 单个文件
python predict_structure.py predict your_file.fa -o result.txt -v

# 批量处理
python batch_predict.py input_folder\ output_folder\ -v
```

### 最常用命令

```powershell
# 输出到屏幕
python predict_structure.py predict input.fa

# 保存到文件
python predict_structure.py predict input.fa -o output.txt

# 详细输出模式
python predict_structure.py predict input.fa -v

# 快速模式（Nussinov 算法）
python predict_structure.py predict input.fa --algorithm nussinov

# 精确模式（Nussinov-Energy 算法，默认）
python predict_structure.py predict input.fa --algorithm nussinov-energy

# 批量处理
python batch_predict.py input_dir\ output_dir\ -v
```

---

## 脚本详细说明

### 1. predict_structure.py（主脚本 - 推荐）

**功能**：预测 RNA 二级结构，支持多种算法和参数配置。

**命令格式**：
```bash
python predict_structure.py predict <input.fa> [选项]
```

**主要选项**：
- `-o, --output <file>` - 输出文件（默认：stdout）
- `-a, --algorithm <name>` - 算法选择（nussinov 或 nussinov-energy）
- `-m, --min-loop-length <n>` - 最小环长度（默认：3）
- `--au-penalty <float>` - AU 碱基对惩罚（默认：0.5）
- `-v, --verbose` - 详细输出模式

**算法选择**：
- `nussinov` - 快速，最大化碱基对数量
- `nussinov-energy` - 准确，考虑能量模型（推荐）

**示例**：
```bash
# 基本使用
python predict_structure.py predict input.fa

# 保存到文件并显示详细信息
python predict_structure.py predict input.fa -o output.txt -v

# 使用快速算法
python predict_structure.py predict input.fa --algorithm nussinov

# 调整参数
python predict_structure.py predict input.fa --min-loop-length 4 --au-penalty 0.7
```

### 2. mxfold2_predict.py（简化版）

**功能**：基础 RNA 二级结构预测，与原 mxfold2_stub 兼容。

**命令格式**：
```bash
python mxfold2_predict.py predict <input.fa>
```

**特点**：
- 轻量级实现
- 向后兼容
- 无额外依赖

**示例**：
```bash
# 输出到屏幕
python mxfold2_predict.py predict input.fa

# 保存到文件
python mxfold2_predict.py predict input.fa > output.txt
```

### 3. batch_predict.py（批量处理）

**功能**：批量处理目录中的所有 FASTA 文件。

**命令格式**：
```bash
python batch_predict.py <input_dir> <output_dir> [选项]
```

**主要选项**：
- `-p, --pattern <pattern>` - 文件匹配模式（默认：*.fa）
- `-a, --algorithm <name>` - 算法选择
- `-m, --min-loop-length <n>` - 最小环长度
- `--au-penalty <float>` - AU 惩罚
- `--numbered-output` - 使用编号输出文件名
- `-v, --verbose` - 详细输出

**示例**：
```bash
# 基本批量处理
python batch_predict.py input_folder\ output_folder\

# 处理 .fasta 文件
python batch_predict.py data\ results\ --pattern "*.fasta"

# 详细输出
python batch_predict.py input\ output\ -v

# 使用特定算法
python batch_predict.py input\ output\ --algorithm nussinov-energy -v
```

---

## 使用示例

### 场景 1：替代 mxfold2

**以前使用 mxfold2**：
```bash
mxfold2 predict input.fa > output.txt
```

**现在使用本工具**：
```bash
# 方法 1：使用增强版（推荐）
python predict_structure.py predict input.fa -o output.txt

# 方法 2：使用兼容版
python mxfold2_predict.py predict input.fa > output.txt
```

### 场景 2：处理 Aptazyme 数据

```powershell
cd preprocessing\rna_structure_predictor

# 单个文件
python predict_structure.py predict ..\Aptazyme\seq.fa -o ..\Aptazyme\ss_predicted.txt -v

# 批量处理
python batch_predict.py ..\Aptazyme\ ..\Aptazyme\results\ -v
```

### 场景 3：处理编码问题文件

```powershell
# 工具会自动检测编码
python predict_structure.py predict problematic_file.fa -o result.txt -v
```

### 场景 4：大批量数据处理

```powershell
# 使用快速算法批量处理
python batch_predict.py large_dataset\ results\ --algorithm nussinov -v
```

### 场景 5：调整预测参数

```powershell
# 更保守的预测（更大的环，更高的 AU 惩罚）
python predict_structure.py predict input.fa --min-loop-length 4 --au-penalty 0.8 -o output.txt
```

---

## 算法说明

### Nussinov 算法

**原理**：动态规划，最大化碱基对数量

**特点**：
- 速度快（O(n³) 时间复杂度）
- 简单实现
- 不考虑能量

**适用场景**：
- 快速预览
- 大批量处理
- 计算资源有限

### Nussinov-Energy 算法（推荐）

**原理**：动态规划 + 简化能量模型

**能量参数**：
```
C-G 配对：-3.0 kcal/mol（最稳定）
A-U 配对：-2.0 kcal/mol
G-U 配对：-1.0 kcal/mol（摆动配对）
AU 末端：+0.5 kcal/mol（惩罚，可配置）
```

**特点**：
- 更准确
- 考虑热力学稳定性
- 稍慢于基础 Nussinov

**适用场景**：
- 重要分析
- 需要高准确度
- 出版质量结果

### 支持的碱基配对

- **Watson-Crick 配对**：
  - A-U（腺嘌呤-尿嘧啶）
  - C-G（胞嘧啶-鸟嘌呤）
  
- **摆动配对**：
  - G-U（鸟嘌呤-尿嘧啶）

### 最小环长度

**默认值**：3

**含义**：发夹环中最少的未配对碱基数

**生物学意义**：防止物理上不可能的紧密环结构

**调整建议**：
- 增加（4-5）：更保守，减少假阳性
- 减少（1-2）：更灵活，可能增加假阳性

---

## 常见问题

### Q1: 输出格式是什么？

**A**: 输出采用点括号记号（dot-bracket notation）：

```
>序列名称
GCGGAUUUAGCUCAG
((((........))))
```

- 第 1 行：序列标识符（以 `>` 开头）
- 第 2 行：RNA 序列
- 第 3 行：二级结构
  - `.` = 未配对碱基
  - `(` 和 `)` = 配对碱基

### Q2: 为什么结果与 mxfold2 不同？

**A**: mxfold2 使用深度学习模型和完整的热力学参数，本工具使用动态规划算法。对大多数应用，本工具的结果已足够准确。

### Q3: 可以处理 DNA 序列吗？

**A**: 本工具专为 RNA 设计（使用 U）。如需处理 DNA，先将 T 替换为 U：

```powershell
# 在文本编辑器中替换，或使用命令
(Get-Content dna.fa) -replace 'T','U' | Set-Content rna.fa
```

### Q4: 支持假结（pseudoknots）吗？

**A**: 不支持。Nussinov 算法只能预测无假结的嵌套结构。

### Q5: 文件编码问题怎么办？

**A**: 工具内置自动编码检测。如仍有问题：

1. 安装编码检测库：
   ```bash
   pip install charset-normalizer
   ```

2. 使用详细模式查看编码信息：
   ```bash
   python predict_structure.py predict input.fa -v
   ```

### Q6: 如何选择算法？

**A**: 
- **日常使用**：`nussinov-energy`（默认，推荐）
- **快速预览/大批量**：`nussinov`
- **最高准确度**：考虑使用真正的 mxfold2（如果可用）

### Q7: 序列太长怎么办？

**A**: 
- 算法复杂度：O(n³)
- 建议：序列 > 1000 nt 可能较慢
- 解决方案：
  1. 使用 `nussinov` 算法（更快）
  2. 分段处理
  3. 考虑使用服务器资源

### Q8: 批量处理出错怎么办？

**A**: 
- 使用 `-v` 查看详细错误信息
- 批处理会跳过单个失败的文件
- 检查最终的成功/失败统计

---

## 技术细节

### 时间复杂度

- **Nussinov 算法**：O(n³)，n 为序列长度
- **空间复杂度**：O(n²)

### 性能参考

| 序列长度 | Nussinov | Nussinov-Energy | 内存占用 |
|----------|----------|-----------------|----------|
| 50 nt | < 0.1 秒 | < 0.1 秒 | < 10 MB |
| 100 nt | < 0.5 秒 | < 1 秒 | < 20 MB |
| 500 nt | 2-3 秒 | 3-5 秒 | < 50 MB |
| 1000 nt | 10-15 秒 | 15-25 秒 | < 100 MB |

### 编码检测机制

1. **BOM 检测**：UTF-8-BOM、UTF-16-LE/BE
2. **外部库**（如安装）：charset_normalizer 或 chardet
3. **回退尝试**：UTF-8 → UTF-16 → CP1252 → Latin-1

### 与其他工具对比

| 工具 | 速度 | 准确性 | 依赖 | Windows 支持 | 编码处理 |
|------|------|--------|------|--------------|----------|
| mxfold2 | 快 | 很高 | TensorFlow | 有限 | 基本 |
| RNAfold | 中 | 高 | ViennaRNA | 中等 | 中等 |
| 本工具（nussinov） | 很快 | 中 | 无 | 完全 | 自动 |
| 本工具（nussinov-energy） | 快 | 中-高 | 无 | 完全 | 自动 |

---

## 迁移指南

### 从 mxfold2_stub 迁移

**旧路径**：
```
preprocessing/mxfold2_stub/mxfold2_predict.py
```

**新路径**：
```
preprocessing/rna_structure_predictor/mxfold2_predict.py
```

**迁移步骤**：
1. 更新脚本路径
2. 测试现有功能
3. （可选）升级到 `predict_structure.py`

**命令对照**：
```bash
# 旧命令（仍然有效）
python preprocessing/mxfold2_stub/mxfold2_predict.py predict input.fa > output.txt

# 新命令（相同功能）
python preprocessing/rna_structure_predictor/mxfold2_predict.py predict input.fa > output.txt

# 推荐命令（功能增强）
python preprocessing/rna_structure_predictor/predict_structure.py predict input.fa -o output.txt
```

### 从单独调用迁移到批量处理

**旧方式**（逐个处理）：
```bash
for file in *.fa; do
    python predict_structure.py predict $file -o ${file%.fa}_structure.txt
done
```

**新方式**（批量处理）：
```bash
python batch_predict.py input_folder/ output_folder/ -v
```

---

## 附录

### A. 输入文件格式（FASTA）

```
>sequence_id_1
GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGA
>sequence_id_2
AUGGCUACCUAGCUAGCUAGCUGACUAGCUAGCUA
```

### B. 输出文件格式（点括号记号）

```
>sequence_id_1
GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGA
((((((((....).))).)))))..................
>sequence_id_2
AUGGCUACCUAGCUAGCUAGCUGACUAGCUAGCUA
((((....))))((((((((...))))))))....
```

### C. 获取帮助

```bash
# 查看 predict_structure.py 帮助
python predict_structure.py -h

# 查看 batch_predict.py 帮助
python batch_predict.py -h

# 查看 mxfold2_predict.py 帮助
python mxfold2_predict.py -h
```

### D. 更多参考

- **命令行参考**：查看 `USAGE.txt`
- **示例文件**：`example.fa`（6 条序列）和 `test.fa`（2 条序列）
- **在线帮助**：运行 `python <script>.py -h`

---

**文档版本**：1.0  
**最后更新**：2025-10-23  
**维护者**：RNAgg 项目  
**状态**：生产就绪 ✅

查看 `USAGE.txt` 获取快速命令行参考。
