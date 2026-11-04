# HEA-LTC-BM (manual branch)

Machine-learning-guided screening of high-entropy alloys (HEAs) with high
bulk modulus (B) and lattice thermal conductivity (κ_l), combining a
Mix-GNN property-prediction model pre-trained on universal materials
databases (Materials Project and AFLOW) with target-space partitioning,
pre-testing and multi-objective optimization (NSGA-II).

This `manual` branch preserves the original project code, written
independently by the authors: the model architecture, training and
inference pipelines, feature engineering, composition screening and
optimization.

The `main` branch contains the complete project files, including
AI-assisted data-visualization and packaging scripts in addition to
everything here.

## Repository structure

- `code/` — all project scripts
  - `code/mixmodel.py`, `code/mixmodel_bulk_modulus.py` — Mix-GNN model
    (CGCNN branch + MLP feature branch, element-wise product fusion)
  - `code/cgcnn_data_processing.py`, `code/mlp_data_processing.py`,
    `code/data_combine.py`, `code/features_normalization*.py` — feature
    engineering and dataset assembly
  - `code/training/` — model training scripts
  - `code/alloy_test_bulk_modulus.py`, `code/alloy_test_thermal_conduct.py`,
    `code/pure_metal_test.py` — inference with the pre-trained models
  - `code/delta_filtering.py`, `code/vec_filtering.py`,
    `code/modify_ratio_for_vec.py`, `code/delta_H/` — solid-solution
    screening criteria (δ, ΔH_mix, VEC)
  - `code/nsga_2_*.py`, `code/bayes_optimization.py` — composition
    optimization (NSGA-II / TPE)
  - `code/painting_figures/` — figure scripts and figure assets
  - `code/materials_id_in_data_sets/materials_ids_in_datasets.xls` — MP and
    AFLOW identifiers of all training-database entries
- `pre_trained_models/` — trained Mix-GNN weights for bulk modulus and
  lattice thermal conductivity, with the corresponding feature
  normalization statistics (`normalization.pt`)
- `framework/` — primitive-cell POSCAR templates of pure metals
  (fcc / bcc / hcp) used to build substitutional alloy models

## Notes on specific files

- `pre_trained_models/*/result/*/test_loader.pt` — serialized test sets.
  The complete datasets are listed in
  `code/materials_id_in_data_sets/materials_ids_in_datasets.xls`; the
  training split can be recovered as the complement of the test split.
- Alloy POSCAR files (optimization candidates and DFT-validation
  structures) and the source data of the article's figures and tables are
  hosted on Zenodo: https://doi.org/10.5281/zenodo.23938398.
- Optimization scripts expect pure-metal supercell templates under
  `{bi,ter,quater,qui}_alloy/{fcc,bcc,hcp}/POSCAR_{Fe,W,Zr}_{fcc,bcc,hcp}`;
  such templates are generated from the primitive cells in `framework/`
  (the scripts call `make_supercell` after loading a template).
- Data paths in the scripts are placeholders (`./data/...`); set them to
  your local data locations before running.

## Requirements

See `requirements.txt`. The models were trained with PyTorch 2.1 (CUDA
12.1) and PyTorch Geometric 2.4; inference also runs on CPU-only builds.

## Citation

Citation details will be added upon publication.

---

# HEA-LTC-BM（manual 分支）

本项目利用机器学习筛选兼具高体积模量（B）与高晶格热导率（κ_l）的高熵合金
（HEA）：先在通用材料数据库（Materials Project 与 AFLOW）上预训练 Mix-GNN
性质预测模型，再结合目标空间分区、预测试与多目标优化（NSGA-II）进行成分筛选。

本 `manual` 分支保留了项目原始代码，全部由作者独立完成，包括模型结构、
训练与推理流程、特征工程、成分筛选与优化等主干部分。

`main` 分支包含完整的工程文件，除本分支内容外，还包含 AI 辅助完成的数据
可视化与项目封装代码。

## 仓库结构

- `code/`——项目全部脚本
  - `code/mixmodel.py`、`code/mixmodel_bulk_modulus.py`——Mix-GNN 模型
    （CGCNN 分支 + MLP 特征分支，逐元素乘积融合）
  - `code/cgcnn_data_processing.py`、`code/mlp_data_processing.py`、
    `code/data_combine.py`、`code/features_normalization*.py`——特征工程
    与数据集组装
  - `code/training/`——模型训练脚本
  - `code/alloy_test_bulk_modulus.py`、`code/alloy_test_thermal_conduct.py`、
    `code/pure_metal_test.py`——基于预训练模型的推理
  - `code/delta_filtering.py`、`code/vec_filtering.py`、
    `code/modify_ratio_for_vec.py`、`code/delta_H/`——固溶体筛选判据
    （δ、ΔH_mix、VEC）
  - `code/nsga_2_*.py`、`code/bayes_optimization.py`——成分优化
    （NSGA-II / TPE）
  - `code/painting_figures/`——绘图脚本与图件资源
  - `code/materials_id_in_data_sets/materials_ids_in_datasets.xls`——训练
    数据库全部条目的 MP 与 AFLOW 编号
- `pre_trained_models/`——体积模量与晶格热导率的 Mix-GNN 训练权重，附对应
  的特征归一化统计量（`normalization.pt`）
- `framework/`——用于构建替位合金模型的纯金属单胞 POSCAR 模板
  （fcc / bcc / hcp）

## 特定文件说明

- `pre_trained_models/*/result/*/test_loader.pt`——序列化的测试集。完整数据
  集见 `code/materials_id_in_data_sets/materials_ids_in_datasets.xls`；训练集
  划分可由完整数据集对测试集取补集恢复。
- 合金 POSCAR 文件（优化候选结构与 DFT 验证结构）以及文章图表的源数据托管于
  Zenodo：https://doi.org/10.5281/zenodo.23938398。
- 优化脚本需要 `{bi,ter,quater,qui}_alloy/{fcc,bcc,hcp}/POSCAR_{Fe,W,Zr}_{fcc,bcc,hcp}`
  形式的纯金属超胞模板；该类模板可由 `framework/` 中的单胞生成（脚本在读取
  模板后会调用 `make_supercell`）。
- 脚本中的数据路径为占位符（`./data/...`），运行前请改为本地实际数据路径。

## 运行环境

见 `requirements.txt`。模型训练使用 PyTorch 2.1（CUDA 12.1）与 PyTorch
Geometric 2.4；推理在 CPU 版本上也可运行。

## 引用

引用信息将在文章正式发表后补充。
