# -*- coding: utf-8 -*-
# 说明：集中管理所有可配置参数。你只需在 train.py 中实例化并修改这些字段。

from typing import Dict, List, Tuple, Optional
import torch


class Config:
    # ========== 目录与输出 ==========
    data_dir: str = "data"                 # 数据根目录（相对路径）
    processed_subdir: str = "processed_data"  # 输出目录（相对 data_dir）
    save_outputs: bool = True              # 是否保存模型/指标/处理数据/图像
    save_processed_dataset: bool = True    # 是否导出 processed_reactions.csv

    # ========== 随机性与设备 ==========
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 数据文件与列名 ==========
    reactions_csv: str = "reactions.csv"   # 反应表文件名
    imine_feat_csv: str = "Imine.csv"
    thiol_feat_csv: str = "Thiol.csv"
    catalyst_feat_csv: str = "Catalyst.csv"
    # ... 原有字段 ...
    # 可选：自定义本次运行的名字，便于识别（若为空将用时间戳+哈希）
    run_name: str = ""

    # 反应表列名（忽略大小写/空格；读取时统一为小写）
    rxn_col_thiol: str = "thiol"
    rxn_col_imine: str = "imine"
    rxn_col_catalyst: str = "catalyst"
    label_col: str = "output"

    # 三个特征表的“标识列”名（区分大小写，按原样匹配）
    id_col_imine: str = "name"
    id_col_thiol: str = "name"
    id_col_catalyst: str = "name"

    # 若 100 维特征列不是“除 id 列以外的所有列”，可显式指定（否则自动推断）
    feat_cols_imine: Optional[List[str]] = None
    feat_cols_thiol: Optional[List[str]] = None
    feat_cols_catalyst: Optional[List[str]] = None

    # 标签与特征缩放
    normalize_label: bool = False          # 若 output 已是 [0,1]，建议 False
    standardize_features: bool = True      # 节点特征标准化开关
    feature_scale_mode: str = "zscore"     # "zscore" 或 "minmax"

    # ========== 图结构（固定三角形）==========
    node_positions: Dict[int, Tuple[float, float]] = {
        0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 1.0)}
    node_labels: Dict[int, str] = {0: "Imine", 1: "Thiol", 2: "Catalyst"}
    input_dim: int = 100

    # ========== 训练超参 ==========
    epochs: int = 600
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4            # L2 正则
    train_ratio: float = 0.8
    num_workers: int = 0

    # 修改开始
    # 固定样本数划分设置（优先级高于按比例划分）
    # True 时按固定样本数划分；False 时按比例（train_ratio）划分
    use_fixed_split: bool = True
    # 训练/验证集的固定样本数（建议与数据总量匹配，例如 600/475）
    train_count: int = 600
    val_count: int = 475
    # 修改结束

    # 损失函数
    use_huber: bool = True                # True 使用 HuberLoss，False 使用 MSELoss
    huber_delta: float = 1.0

    # 学习率调度与早停
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20
    early_stopping: bool = True
    early_stop_patience: int = 50

    # ========== 模型结构 ==========
    gat_hidden: int = 128
    gat_heads: int = 2
    dropout: float = 0.3
    pooling: str = "gap+gmp"              # "gap" / "gmp" / "gap+gmp"
    mlp_sizes: List[int] = [256, 128]

    # ========== 解释与可视化 ==========
    do_explain_node: bool = True          # 节点遮盖重要性
    do_explain_edge: bool = True          # 边遮盖重要性
    do_explain_feature: bool = True       # 特征维度遮盖重要性
    topk_feat: int = 10                   # 每个节点输出前K重要特征
    do_graph_viz: bool = True             # 画三角形图
    do_train_curves: bool = True          # 训练曲线图
