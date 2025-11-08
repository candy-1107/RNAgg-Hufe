## 快速上手（对 AI 开发者）

本仓库实现了基于 VAE 的 RNA 序列生成与可视化工具（RNAgg）。下面 20-50 行指令帮助你快速理解代码结构、常见约定与可安全修改的热点文件。

### 核心组件（大局观）
- `scripts/`：主用例脚本（训练/生成/嵌入/可视化），例如 `RNAgg_train.py`, `RNAgg_generate.py`, `gg_get_embedding.py`, `gg_draw_latent.py`。
- `scripts/RNAgg_VAE.py`：模型定义（MLP_VAE / MLP_VAE_REGRE）。不要随意改变序列化字典的键名（见下文模型 checkpoint 约定）。
- `scripts/Binary_matrix.py`：把（序列 + 二级结构）转换为输入矩阵：one-hot (核苷) + grammar 矩阵（G_DIM=11）。
- `scripts/utils_gg.py`：轻量 I/O、Dataset 封装与 token 映射（重要的 helper）。
- `scripts/SS2shape3.py`：从二级结构生成 grammar 规则（`getBPpos_ij`, `generate_rule_G4b`），Binary_matrix 依赖它构造 grammar 矩阵。
- `preprocessing/`：数据预处理脚本与 usage 文档（Aptazyme、RfamSeed、tRNA 等），提供如何构建 `sample_input.txt` / activity 文件的示例流程。

### 运行与工作流（可直接引用到命令）
- 安装依赖：`pip install -r requirements.txt`（项目在 Linux、Python 3.10 上测试；CUDA 11.8 为可选加速环境）。
- 训练示例：
  - `python scripts/RNAgg_train.py ../example/sample_input.txt --out_dir ./out`  # 生成 model_RNAgg.pth
  - 若考虑 activity：`python scripts/RNAgg_train.py ../example/sample_input.txt --act_fname ../example/sample_act.txt`
- 生成示例：
  - `python scripts/RNAgg_generate.py 10 model_RNAgg.pth output.txt`  # 生成 10 个序列
- 嵌入与绘图：
  - `python scripts/gg_get_embedding.py ../example/sample_input.txt model_RNAgg.pth emb.pickle`
  - `python scripts/gg_draw_latent.py emb.pickle [--color_fname ../example/sample_act.txt]`

### 文件/格式约定（必须严格遵守）
- 输入文件（`sample_input.txt`）: 三列以空格分隔：`id sequence secondary_structure`。参见 `example/sample_input.txt`。
- activity 文件（可选，`--act_fname`）: 两列：`id activity_value`（float）。参见 `example/sample_act.txt`。
- 填充与字符：序列右填充字符 `'x'`；二级结构右填充 `'.'`。Vocabulary 恒为 `ACGU-x`（scripts 中常量 NUC_LETTERS）。
- Binary 矩阵：每条记录被转换为 shape=(max_len, word_size + G_DIM) 的矩阵（若使用 `--nuc_only` 则只有 one-hot，不含 grammar 部分）。G_DIM 当前硬编码为 11（见 `scripts/Binary_matrix.py`）。
- 模型 checkpoint（torch.save）字典中必须包含键：`model_state_dict`, `d_rep`, `max_len`, `type`（'org' 或 'act'）, `nuc_only`（'yes'/'no'）。不要改动这些键，否则训练/生成流程会断裂。

### 常见陷阱与代码风格约定
- `RNAgg_train.py` 会把 `nuc_only` 转换为字符串 `'yes'/'no'` 存入 checkpoint；许多模块依赖该字符串比较。
- `Binary_matrix.makeMatrix` 会跳过包含邻位配对的序列（检测字符串 `'()'`），会在 stderr 打印并继续。若修改过滤规则，请保证下游 index/padding 兼容。
- `utils_gg.Dataset.__getitem__` 返回 `(tensor, sid, act_tensor)`；训练循环假设第三项始终存在（用 NaN 填充时也可）。修改 Dataset 接口必须同步更新 `RNAgg_train.py` 的 data-loading 代码。
- 并行/生成相关：`RNAgg_generate.py` 在生成时用 joblib 并行 (`n_jobs=args.n_cpu`)；如果改动数据序列化（pickle），请保持兼容性。

### 修改建议（给 AI 的具体小任务）
- 若要添加新 CLI 参数：在 `RNAgg_train.py` / `RNAgg_generate.py` 的 argparse 中添加，并在文件顶部文档字符串中补充使用示例。
- 若要支持不同 padding 字符或 vocabulary：修改 `utils_gg.get_token2idx` 与 `Binary_matrix` 中的填充与 one-hot 构造，确保 `nuc_only` 分支一致。
- 若要增加单元测试：优先对 `utils_gg.readInput`、`utils_gg.readAct`、`Binary_matrix.getGramMatrix` 与 `SS2shape3.getBPpos_ij` 写快速 pytest 测试（输入/输出可用 example/ 和 preprocessing/ 下的小样例）。

### 编辑/审查注意点（必须检查）
1. 不要重命名或移除 checkpoint 中的键（见上文）。
2. 保持命令行接口向后兼容（参数默认值、语义不要突然变化）。
3. 当更改 max_len/padding 逻辑时，确保训练与生成都能读取相同的 `max_len` 来重建模型输入/输出维度。

如果以上内容有不清楚或需要补充的部分，请告诉我你想先改哪一块（例如：为 `Binary_matrix` 增加单元测试，或为生成流程增加 JSON 输出），我会基于该目标继续迭代指令文档或直接修改代码。
