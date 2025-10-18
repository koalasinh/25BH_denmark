# -*- coding: utf-8 -*-
# 说明：
# - 训练入口，顶部集中配置参数。
# - 执行步骤：读取数据 -> 构图 -> 训练（含早停/调度）-> 评估 -> 解释（遮盖）-> 可视化
# - 所有输出写入 data/processed_data 下（可在 sr_config.Config 中修改）

import torch
import json

from sr_config import Config
from sr_trainer import Trainer
from sr_utils import OutputManager
from sr_viz import Visualizer
from sr_explain import Explainer

train = True
retain = False
# retain_data = "data/processed_data/runs/20250101-120000-ab12cd34/config.json"
retain_data = "data/processed_data/runs/20250101-120000-ab12cd34/config.json"

if train:
    # ========= 需要你设置/确认的输入参数（中文提示） =========
    cfg = Config()

    # 1) 数据与输出目录（推荐将你的 CSV 放在 data/ 下）
    cfg.data_dir = "data"
    cfg.processed_subdir = "processed_data"
    cfg.save_outputs = True
    cfg.save_processed_dataset = True

    # 2) 文件名与列名（若你的反应表为 denmark.csv，请改为同名）
    # 包含: thiol, imine, catalyst, output
    cfg.reactions_csv = "denmark.csv"
    cfg.imine_feat_csv = "Imine_1cal_2iloc_3select.csv"
    cfg.thiol_feat_csv = "Thiol_1cal_2iloc_3select.csv"
    cfg.catalyst_feat_csv = "Catalyst_1cal_2iloc_3select.csv"

    # 反应表列名（忽略大小写/空格，会统一转为小写匹配）
    cfg.rxn_col_thiol = "thiol"
    cfg.rxn_col_imine = "imine"
    cfg.rxn_col_catalyst = "catalyst"
    cfg.label_col = "output"

    # 特征表的“标识列”（区分大小写，按原样匹配）
    cfg.id_col_imine = "Imine"
    cfg.id_col_thiol = "Thiol"
    cfg.id_col_catalyst = "Catalyst"

    # 如果 100 维特征列不是“除 id 列以外的所有列”，可在此显式指定列表；否则保持 None 自动推断
    cfg.feat_cols_imine = None
    cfg.feat_cols_thiol = None
    cfg.feat_cols_catalyst = None

    # 3) 标签与特征缩放设置
    # 若 output 本身已是 [0,1]，建议关闭 normalize_label
    cfg.normalize_label = False
    cfg.standardize_features = True
    cfg.feature_scale_mode = "zscore"  # "zscore" 或 "minmax"

    # 4) 三角形图的三个节点位置（训练不使用，仅用于可视化）
    cfg.node_positions = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 1.0)}
    cfg.node_labels = {0: "Imine", 1: "Thiol", 2: "Catalyst"}
    cfg.input_dim = 100

    # 5) 训练与优化超参（如何设置：请参考每项注释）
    # 随机种子：用于数据划分、初始化等。为了可复现，建议固定不变；若要多跑几次统计均值，可改为不同值（如 0, 1, 2...）。
    cfg.seed = 42
    # 设备：建议优先使用 GPU（cuda），若没有则使用 cpu。无需手动修改，默认会自动检测。
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # 训练总轮数：配合早停使用。若早停开启，实际训练到验证集长期不提升就会提前停止。
    # 建议范围：300~1000（样本较小可用 600 左右；如果不开早停，不要设置过大以免浪费时间）
    cfg.epochs = 600
    # 批大小（batch size）：影响训练稳定性与速度。显存允许下可适度增大（如 128/256）。
    # 越大统计越稳，但可能欠拟合；越小噪声大但可能更易跳出局部最优。常用 64/128。
    cfg.batch_size = 64
    # 学习率：训练收敛的步长。过大可能不收敛，过小训练慢。
    # 建议范围：1e-4 ~ 3e-3。一般先用 3e-4；若验证不提升可尝试 1e-4 或配合调度器自动调整。
    cfg.lr = 2e-4
    # L2 正则（权重衰减）：对抗过拟合，提高泛化能力。数值越大正则越强。
    # 建议范围：1e-5 ~ 5e-4，起步 1e-4；若明显过拟合（Train Loss 远低于 Val Loss）可稍增大。
    cfg.weight_decay = 1e-4
    # 训练集比例：用于训练/验证划分。推荐 0.7~0.9 之间。
    # 注意：本项目使用“分箱分层”划分，以保证回归标签分布在训练/验证集一致。
    cfg.train_ratio = 0.8
    # 使用固定样本数划分（优先于按比例）
    cfg.use_fixed_split = True
    cfg.train_count = 600   # 训练集固定样本数
    cfg.val_count = 475     # 验证集固定样本数
    # DataLoader 的工作线程数：Windows 下一般设 0，Linux/服务器可设为 CPU 核数的 1/2 或更高以加速数据读取。
    cfg.num_workers = 0
    # 使用 HuberLoss（对异常值更稳健）还是 MSELoss（标准回归损失）。
    # 如果数据中存在少量异常点或误差分布有长尾，建议 True（使用 Huber），否则 False（使用 MSE）。
    cfg.use_huber = True
    # HuberLoss 的 delta：越大越接近 L2，越小越接近 L1。常用 1.0；如果异常值特别多可适当减小。
    cfg.huber_delta = 1.0
    # 学习率调度器（ReduceLROnPlateau）：当验证集 Loss 长期不下降时自动降低学习率。
    # True：开启；False：关闭。建议开启，能提升稳定性。
    cfg.use_scheduler = True
    # 当 Plateau 发生时学习率降低的倍数（factor）：新的 lr = 旧的 lr * factor。0.5 表示减半。
    # 若想更细力度降低，可设 0.3；若想更快衰减，可设 0.1。
    cfg.scheduler_factor = 0.5
    # 调度器容忍轮数（patience）：验证集 Loss 连续多少个 epoch 未提升才降低学习率。
    # 建议 10~50 之间。数据噪声大时适当加大（如 20）。
    cfg.scheduler_patience = 18
    # 早停（Early Stopping）：验证集 Loss 在一定轮数内没有提升就提前停止，避免无效训练。
    # 建议开启（True）。若追求极致收敛可关闭，但训练时间会更长。
    cfg.early_stopping = True
    # 早停容忍轮数（patience）：验证集 Loss 连续多少个 epoch 未提升就停止训练。
    # 建议 30~100 之间。与 epochs 搭配使用，通常设为 50 比较稳。
    cfg.early_stop_patience = 80

    # 6) 模型结构（此处也附简短说明，便于你根据表现微调）
    # GAT 隐藏通道数（每头输出的通道数 out_channels，拼接后通道数为 gat_hidden * gat_heads）
    # 建议 64/128/256。样本规模较小建议 128；若过拟合明显可降到 64。
    cfg.gat_hidden = 256
    # GAT 多头注意力的头数：2/4 常见。头数越多，表达力越强但计算量增加。
    cfg.gat_heads = 4
    # Dropout：防止过拟合。建议 0.2~0.5。若过拟合明显可适当增大。
    cfg.dropout = 0.3
    # 池化方式：
    # - "gap"：全局平均池化（稳定，三节点时常作为基线）
    # - "gmp"：全局最大池化（对极值敏感，可能更噪）
    # - "gap+gmp"：均值与最大拼接（维度翻倍，通常效果更好；需注意安装 torch-scatter 可提升速度）
    cfg.pooling = "gap+gmp"
    # MLP 头的层宽列表：池化后接的全连接层结构。可根据过拟合/欠拟合情况调整层数与宽度。
    # 小模型可设 [256, 128]；若欠拟合（能力不够）可适度增大如 [512, 256, 128]。
    cfg.mlp_sizes = [512, 256, 128]

    # 7) 解释与可视化开关（用于分析模型和生成论文/报告素材）
    # 是否计算“节点遮盖”重要性：将某个节点的全部特征置零，观察 RMSE 增量，评估该节点对预测的影响。
    # True：计算并输出 CSV；False：跳过以节省时间。
    cfg.do_explain_node = True
    # 是否计算“边遮盖”重要性：去除某条无向边（两条有向边共同去除），观察 RMSE 增量。
    # 三角形图共有 3 条无向边；True：计算并输出 CSV；False：跳过。
    cfg.do_explain_edge = True
    # 是否计算“特征维度遮盖”重要性：对每个节点的每个特征维度单独置零，观察 RMSE 增量。
    # 计算量较大（3 * input_dim 次评估），验证集较大会耗时较长；建议训练稳定后再开启。
    cfg.do_explain_feature = True
    # 每个节点输出前 K 个最重要特征维度（基于 RMSE 增量排序），用于快速浏览。
    # 建议 5~20 之间。过大意义不大，过小可能遗漏信息。
    cfg.topk_feat = 10
    # 是否绘制三角形图结构（节点坐标与标签）并保存 PNG。用于论文图/报告展示。
    cfg.do_graph_viz = True
    # 是否绘制训练曲线图：train/val loss、val R2 随 epoch 的变化。便于判断过拟合与收敛情况。
    cfg.do_train_curves = True

    # ========= 启动训练 =========
    trainer = Trainer(cfg)
    trainer.fit()
    metrics = trainer.evaluate()
    # 解释
    if cfg.do_explain_node or cfg.do_explain_edge or cfg.do_explain_feature:
        explainer = Explainer(trainer.model, trainer.val_loader, trainer.full_dataset_ref,
                              outman=trainer.outman, cfg=cfg)
        # 将 Explainer 改成可选接收 runman，并用 runman 保存
        # explainer = Explainer(..., runman=trainer.runman)
        # 然后在 explainer 内部用 self.runman.save_csv_local(...) 保存到本次 run 目录
        # 为了不大改动，你也可以直接把返回的 DataFrame 手动保存：
        if cfg.do_explain_node:
            df = explainer.node_importance()
            trainer.runman.save_csv_local("explain_node_importance.csv", df)
        if cfg.do_explain_edge:
            df = explainer.edge_importance()
            trainer.runman.save_csv_local("explain_edge_importance.csv", df)
        if cfg.do_explain_feature:
            df = explainer.feature_importance(topk=cfg.topk_feat)
            trainer.runman.save_csv_local("explain_feature_importance.csv", df)

    # 可视化
    if cfg.do_graph_viz or cfg.do_train_curves:
        viz = Visualizer(lambda name: trainer.runman.path(name))
        if cfg.do_graph_viz:
            viz.plot_triangle_graph(
                cfg.node_positions, cfg.node_labels, filename="triangle_graph.png")
        if cfg.do_train_curves:
            history = {"train_loss": trainer.history["train_loss"],
                       "val_loss": trainer.history["val_loss"],
                       "val_r2": trainer.history["val_r2"]}
            viz.plot_training_curves(history, filename="training_curves.png")

    def load_config_and_run(config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = Config()
        # 将 JSON 中的键值回填到 cfg
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        # 可选：为了复现，保持相同 seed；也可在这里覆盖 device 等
        trainer = Trainer(cfg)
        trainer.fit()
        trainer.evaluate()

if retain:
    load_config_and_run(retain_data)

    # ========= 可选：遮盖解释 =========
    if cfg.do_explain_node or cfg.do_explain_edge or cfg.do_explain_feature:
        outman = OutputManager(
            cfg.data_dir, cfg.processed_subdir, cfg.save_outputs)
        explainer = Explainer(trainer.model, trainer.val_loader,
                              trainer.full_dataset_ref, outman, cfg)
        if cfg.do_explain_node:
            explainer.node_importance()
        if cfg.do_explain_edge:
            explainer.edge_importance()
        if cfg.do_explain_feature:
            explainer.feature_importance(topk=cfg.topk_feat)

    # ========= 可选：可视化 =========
    if cfg.do_graph_viz or cfg.do_train_curves:
        outman = OutputManager(
            cfg.data_dir, cfg.processed_subdir, cfg.save_outputs)
        viz = Visualizer(outman.path)
        if cfg.do_graph_viz:
            viz.plot_triangle_graph(
                cfg.node_positions, cfg.node_labels, filename="triangle_graph.png")
        if cfg.do_train_curves:
            # 使用训练器缓存的曲线（trainer.fit 已记录）
            history = {
                "train_loss": trainer.history["train_loss"],
                "val_loss": trainer.history["val_loss"],
                "val_r2": trainer.history["val_r2"]
            }
            viz.plot_training_curves(history, filename="training_curves.png")

    # 运行完成后，你将在 data/processed_data 下看到：
    # - processed_reactions.csv, label_meta.json
    # - best_model.pt, metrics.json, training_history.json
    # - explain_node_importance.csv, explain_edge_importance.csv, explain_feature_importance.csv（取决于开关）
    # - triangle_graph.png, training_curves.png（取决于开关）
