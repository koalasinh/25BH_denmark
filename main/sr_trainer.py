# -*- coding: utf-8 -*-
# 说明：训练器，包含分箱分层划分、早停、学习率调度、评估与曲线输出。

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sr_config import Config
from sr_utils import OutputManager, compute_metrics, RunManager
from sr_data import ReactionDataset
from sr_model import GNNModel


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.outman = OutputManager(
            cfg.data_dir, cfg.processed_subdir, cfg.save_outputs)

        full_dataset = ReactionDataset(
            cfg, outman=self.outman if cfg.save_outputs else None)
        n_total = len(full_dataset)
        assert n_total > 1, "数据集样本太少。"
        # 新增：run 管理器
        self.runman = RunManager(self.outman, cfg)
        self.runman.save_config()  # 运行开始时立即保存本次配置快照

        # 修改开始
        # 分箱分层划分（对缩放空间标签分箱）
        # y_all = np.array([float(full_dataset.get(i).y.item())
        #                  for i in range(n_total)])
        # num_bins = min(10, max(2, int(np.sqrt(n_total))))  # 根号N个桶，更稳
        # bins = np.linspace(y_all.min(), y_all.max() + 1e-9, num_bins + 1)
        # y_binned = np.digitize(y_all, bins) - 1
        # y_binned = np.clip(y_binned, 0, num_bins - 1)

        # idx_all = np.arange(n_total)
        # stratify = y_binned if n_total >= num_bins else None
        # idx_tr, idx_va = train_test_split(
        #     idx_all,
        #     test_size=(1.0 - cfg.train_ratio),
        #     random_state=cfg.seed,
        #     stratify=stratify
        # )

        # 修改结束
        # 修改开始
        # 构造用于分层的“分箱标签”（将连续标签切成若干桶）
        y_all = np.array([float(full_dataset.get(i).y.item())
                         for i in range(n_total)])
        num_bins = min(10, max(2, int(np.sqrt(n_total))))  # 根号N个桶，更稳
        bins = np.linspace(y_all.min(), y_all.max() + 1e-9, num_bins + 1)
        y_binned = np.digitize(y_all, bins) - 1
        y_binned = np.clip(y_binned, 0, num_bins - 1)

        idx_all = np.arange(n_total)

        if self.cfg.use_fixed_split:
            # 按固定样本数划分
            t_count = int(self.cfg.train_count)
            v_count = int(self.cfg.val_count)
            # 兜底修正：如果总数与设置不一致，自动调整验证集数量
            if t_count + v_count > n_total:
                v_count = n_total - t_count
                print(f"[数据划分] 训练/验证数量之和超过总样本，自动将验证数调整为 {v_count}。")
            # 若验证数为 0 或训练数为 0，抛错提示
            assert t_count > 0 and v_count > 0, f"固定划分无效：train_count={t_count}, val_count={v_count}"
            # 使用 StratifiedShuffleSplit 按“桶标签”分层划分（整数样本数量）
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=t_count, test_size=v_count, random_state=self.cfg.seed
            )
            # 注意：StratifiedShuffleSplit 的 y 需要是离散标签，我们传入 y_binned
            (idx_tr, idx_va), = sss.split(idx_all, y_binned)
        else:
            # 按比例划分（原逻辑），保留以备不时之需
            from sklearn.model_selection import train_test_split
            stratify = y_binned if n_total >= num_bins else None
            idx_tr, idx_va = train_test_split(
                idx_all,
                test_size=(1.0 - self.cfg.train_ratio),
                random_state=self.cfg.seed,
                stratify=stratify
            )

        self.full_dataset_ref = full_dataset
        self.train_set = torch.utils.data.Subset(full_dataset, idx_tr)
        self.val_set = torch.utils.data.Subset(full_dataset, idx_va)
        # 修改结束

        self.full_dataset_ref = full_dataset
        self.train_set = torch.utils.data.Subset(full_dataset, idx_tr)
        self.val_set = torch.utils.data.Subset(full_dataset, idx_va)

        self.train_loader = DataLoader(self.train_set, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers)
        self.val_loader = DataLoader(self.val_set, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers)

        self.model = GNNModel(
            input_dim=cfg.input_dim,
            gat_hidden=cfg.gat_hidden,
            gat_heads=cfg.gat_heads,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
            mlp_sizes=cfg.mlp_sizes
        ).to(cfg.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.use_huber:
            self.criterion = nn.HuberLoss(delta=cfg.huber_delta)
        else:
            self.criterion = nn.MSELoss()

        self.scheduler = None
        if cfg.use_scheduler:
            # try:
            #     self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         self.optimizer, mode="min", factor=cfg.scheduler_factor,
            #         patience=cfg.scheduler_patience, verbose=True  # 新版支持
            #     )
            # except TypeError:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=cfg.scheduler_factor,
                patience=cfg.scheduler_patience  # 旧版不支持 verbose
            )

        # 日志缓存（用于可视化）
        self.history = {"train_loss": [], "val_loss": [], "val_r2": []}

    def _run_one_epoch(self, loader, train: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        device = self.cfg.device
        self.model.train() if train else self.model.eval()

        epoch_loss = 0.0
        preds, gts = [], []

        for batch in loader:
            batch = batch.to(device)
            if train:
                self.optimizer.zero_grad()
            out = self.model(batch)
            y = batch.y.view(-1)
            loss = self.criterion(out, y)
            if train:
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
            preds.append(out.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())

        epoch_loss /= len(loader.dataset)
        y_pred = np.concatenate(preds, axis=0) if len(
            preds) > 0 else np.array([])
        y_true = np.concatenate(gts, axis=0) if len(gts) > 0 else np.array([])
        return epoch_loss, y_true, y_pred

    def fit(self):
        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        patience = self.cfg.early_stop_patience

        for ep in range(1, self.cfg.epochs + 1):
            tr_loss, ytr, ptr = self._run_one_epoch(
                self.train_loader, train=True)
            va_loss, yva, pva = self._run_one_epoch(
                self.val_loader, train=False)

            tr_metrics = compute_metrics(ytr, ptr)
            va_metrics = compute_metrics(yva, pva)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["val_r2"].append(va_metrics["R2"])

            print(f"Epoch [{ep:03d}/{self.cfg.epochs}] "
                  f"TrainLoss={tr_loss:.6f} ValLoss={va_loss:.6f} "
                  f"Train(RMSE={tr_metrics['RMSE']:.4f}, R2={tr_metrics['R2']:.4f}) "
                  f"Val(RMSE={va_metrics['RMSE']:.4f}, R2={va_metrics['R2']:.4f})")

            # 调度器
            if self.scheduler is not None:
                prev_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                self.scheduler.step(va_loss)
                new_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                # 如有变化则打印
                for i, (old, new) in enumerate(zip(prev_lrs, new_lrs)):
                    if new < old:
                        print(
                            f"[LR Plateau] param_group {i}: lr {old:.2e} -> {new:.2e}")

            # 保存最佳
            improved = va_loss < best_val - 1e-6
            if improved:
                best_val = va_loss
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1

            # 早停
            if self.cfg.early_stopping and bad_epochs >= patience:
                print(f"[早停] 验证集 {patience} 个 epoch 未提升，提前停止。")
                break

        if best_state is not None and self.cfg.save_outputs:
            self.model.load_state_dict(best_state)
            # 保存最佳模型到本次 run 目录
            self.runman.save_model(best_state)
            self.outman.save_model("best_model.pt", best_state)

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        device = self.cfg.device
        all_true, all_pred = [], []
        for batch in self.val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                out = self.model(batch).detach().cpu().numpy()
            y = batch.y.view(-1).detach().cpu().numpy()
            all_true.append(y)
            all_pred.append(out)

        y_true_scaled = np.concatenate(all_true, axis=0)
        y_pred_scaled = np.concatenate(all_pred, axis=0)

        # 若训练做了标签归一化，评估时反归一
        if self.full_dataset_ref._label_minmax is not None:
            y_true = self.full_dataset_ref.inverse_transform_label(
                y_true_scaled)
            y_pred = self.full_dataset_ref.inverse_transform_label(
                y_pred_scaled)
        else:
            y_true, y_pred = y_true_scaled, y_pred_scaled

        # ... 组装 y_true/y_pred（缩放空间）并反归一化到原尺度 ...
        metrics = compute_metrics(y_true, y_pred)
        print("Validation metrics (original scale):", metrics)
        if self.cfg.save_outputs:
            # 保存最终指标、训练曲线到本次 run 目录
            self.runman.save_metrics(metrics)
            hist = {
                "train_loss": self.history["train_loss"],
                "val_loss": self.history["val_loss"],
                "val_r2": self.history["val_r2"]
            }
            self.runman.save_history(hist)
            # 追加到 runs_index.csv（总表）
            self.runman.append_runs_index(metrics)
            # 可选：导出逐样本验证集预测对比（原始尺度）
            import pandas as pd
            df_pred = pd.DataFrame({
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_pred - y_true
            })
            self.runman.save_csv_local("val_predictions.csv", df_pred)

            self.outman.save_json("metrics.json", metrics)
            # 同时保存训练曲线（供可视化模块使用）
            hist = {
                "train_loss": self.history["train_loss"],
                "val_loss": self.history["val_loss"],
                "val_r2": self.history["val_r2"]
            }
            self.outman.save_json("training_history.json", hist)
        return metrics
