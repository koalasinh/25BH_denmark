# -*- coding: utf-8 -*-
# 说明：遮盖解释（节点/边/特征维度重要性），在验证集上通过遮盖后 RMSE 增量衡量重要性。
# 使用方法：
# - 在训练完成后，构建 Explainer(model, val_loader, dataset_ref, outman, cfg)
# - 按需调用 node_importance()/edge_importance()/feature_importance() 即可
# - 结果将写入 data/processed_data 下的 CSV 文件

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd

from sr_config import Config
from sr_utils import OutputManager, compute_metrics


class Explainer:
    def __init__(self, model: torch.nn.Module, val_loader: DataLoader,
                 dataset_ref, outman: OutputManager, cfg: Config):
        self.model = model
        self.val_loader = val_loader
        self.ds = dataset_ref
        self.outman = outman
        self.cfg = cfg

    # ================= 基础评估（RMSE） =================
    def _eval_rmse(self) -> float:
        self.model.eval()
        device = self.cfg.device
        ys, ps = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                out = self.model(batch).detach().cpu().numpy()
                y = batch.y.view(-1).detach().cpu().numpy()
                ys.append(y)
                ps.append(out)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)
        if self.ds._label_minmax is not None:
            ys = self.ds.inverse_transform_label(ys)
            ps = self.ds.inverse_transform_label(ps)
        m = compute_metrics(ys, ps)
        return m["RMSE"]

    # ================= 节点重要性（整节点遮盖） =================
    def node_importance(self) -> pd.DataFrame:
        """
        对每个节点做遮盖：将该节点的全部特征置零，计算 RMSE 增量。
        返回 DataFrame，并保存到 explain_node_importance.csv
        """
        base_rmse = self._eval_rmse()
        results = []
        for node in [0, 1, 2]:
            inc = self._rmse_with_node_mask(node) - base_rmse
            results.append(
                {"node": node, "importance_rmse_increase": float(inc)})
        df = pd.DataFrame(results).sort_values(
            "importance_rmse_increase", ascending=False)
        self.outman.save_csv("explain_node_importance.csv", df)
        return df

    def _rmse_with_node_mask(self, node_idx: int) -> float:
        self.model.eval()
        device = self.cfg.device
        ys, ps = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                # 克隆特征并遮盖指定节点（每个图3个节点）
                x = batch.x.clone()
                num_graphs = batch.num_graphs
                idxs = torch.arange(num_graphs, device=device) * 3 + node_idx
                x[idxs] = 0.0
                batch_masked = Data(
                    x=x, edge_index=batch.edge_index, y=batch.y, batch=batch.batch)
                out = self.model(batch_masked).detach().cpu().numpy()
                y = batch.y.view(-1).detach().cpu().numpy()
                ys.append(y)
                ps.append(out)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)
        if self.ds._label_minmax is not None:
            ys = self.ds.inverse_transform_label(ys)
            ps = self.ds.inverse_transform_label(ps)
        m = compute_metrics(ys, ps)
        return m["RMSE"]

    # ================= 边重要性（去除无向边） =================
    def edge_importance(self) -> pd.DataFrame:
        """
        新的边集合：
        - 无向对：(0,1) -> 一次同时去掉 0->1 和 1->0
        - 定向边：2->0、2->1 -> 分别单独去掉
        """
        base_rmse = self._eval_rmse()
        rows = []

        # 先评估无向对 (0,1)
        inc_01 = self._rmse_without_edges(
            remove_pairs=[(0, 1)], remove_directed=[]) - base_rmse
        rows.append({"edge": "0-1 (undirected)",
                    "importance_rmse_increase": float(inc_01)})

        # 再评估定向边 2->0、2->1
        for (u, v) in [(2, 0), (2, 1)]:
            inc = self._rmse_without_edges(
                remove_pairs=[], remove_directed=[(u, v)]) - base_rmse
            rows.append(
                {"edge": f"{u}->{v}", "importance_rmse_increase": float(inc)})

        df = pd.DataFrame(rows).sort_values(
            "importance_rmse_increase", ascending=False)
        self.outman.save_csv("explain_edge_importance.csv", df)
        return df

    def _rmse_without_edges(self, remove_pairs, remove_directed) -> float:
        """
        - remove_pairs: 列表，元素为 (u,v)，表示去除无向对 u<->v（两条方向边一起去掉）
        - remove_directed: 列表，元素为 (u,v)，表示去除单条有向边 u->v
        """
        self.model.eval()
        device = self.cfg.device
        ys, ps = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                ei = batch.edge_index
                src, dst = ei[0], ei[1]
                keep = torch.ones(ei.size(1), dtype=torch.bool, device=device)
                num_g = batch.num_graphs
                for g in range(num_g):
                    # 去除无向对
                    for (u, v) in remove_pairs:
                        gu, gv = g*3 + u, g*3 + v
                        mask_uv = (src == gu) & (dst == gv)
                        mask_vu = (src == gv) & (dst == gu)
                        keep = keep & (~mask_uv) & (~mask_vu)
                    # 去除单向边
                    for (u, v) in remove_directed:
                        gu, gv = g*3 + u, g*3 + v
                        mask = (src == gu) & (dst == gv)
                        keep = keep & (~mask)
                masked_ei = ei[:, keep]
                out = self.model(Data(x=batch.x, edge_index=masked_ei, y=batch.y, batch=batch.batch))\
                    .detach().cpu().numpy()
                y = batch.y.view(-1).detach().cpu().numpy()
                ys.append(y)
                ps.append(out)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)
        if self.ds._label_minmax is not None:
            ys = self.ds.inverse_transform_label(ys)
            ps = self.ds.inverse_transform_label(ps)
        m = compute_metrics(ys, ps)
        return m["RMSE"]

    def _rmse_without_undirected_edge(self, u: int, v: int) -> float:
        """
        在验证集上去除无向边 (u, v)（即去掉 u->v 和 v->u 两条有向边），评估 RMSE
        """
        self.model.eval()
        device = self.cfg.device
        ys, ps = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                ei = batch.edge_index
                src, dst = ei[0], ei[1]
                # 构造需去除的全局边集合：对于每个图 g，去除 (g*3+u, g*3+v) 与 (g*3+v, g*3+u)
                num_graphs = batch.num_graphs
                # 初始化保留掩码
                keep = torch.ones(ei.size(1), dtype=torch.bool, device=device)
                # 逐图过滤（每图只有6条边，代价很小）
                for g in range(num_graphs):
                    gu, gv = g*3 + u, g*3 + v
                    mask_uv = (src == gu) & (dst == gv)
                    mask_vu = (src == gv) & (dst == gu)
                    keep = keep & (~mask_uv) & (~mask_vu)
                masked_ei = ei[:, keep]
                batch_masked = Data(
                    x=batch.x, edge_index=masked_ei, y=batch.y, batch=batch.batch)
                out = self.model(batch_masked).detach().cpu().numpy()
                y = batch.y.view(-1).detach().cpu().numpy()
                ys.append(y)
                ps.append(out)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)
        if self.ds._label_minmax is not None:
            ys = self.ds.inverse_transform_label(ys)
            ps = self.ds.inverse_transform_label(ps)
        m = compute_metrics(ys, ps)
        return m["RMSE"]

    # ================= 特征维度重要性（单节点单特征遮盖） =================
    def feature_importance(self, topk: int = 10) -> pd.DataFrame:
        """
        对每个节点的每个特征维度，置零该维度，计算 RMSE 增量；返回按重要性排序的前 K（每节点）。
        保存到 explain_feature_importance.csv
        注意：此操作计算量较大（3*input_dim 次评估），建议在验证集不大时使用。
        """
        base_rmse = self._eval_rmse()
        input_dim = self.cfg.input_dim
        rows: List[Dict] = []
        for node_idx in [0, 1, 2]:
            for j in range(input_dim):
                inc = self._rmse_with_feature_mask(node_idx, j) - base_rmse
                rows.append({"node": node_idx, "feature_index": j,
                            "importance_rmse_increase": float(inc)})
        df = pd.DataFrame(rows)
        # 每个节点各取 topk
        tops = []
        for node_idx in [0, 1, 2]:
            sub = df[df["node"] == node_idx].sort_values(
                "importance_rmse_increase", ascending=False)
            tops.append(sub.head(topk))
        out_df = pd.concat(tops, axis=0, ignore_index=True)
        self.outman.save_csv("explain_feature_importance.csv", out_df)
        return out_df

    def _rmse_with_feature_mask(self, node_idx: int, feat_j: int) -> float:
        self.model.eval()
        device = self.cfg.device
        ys, ps = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                x = batch.x.clone()
                num_graphs = batch.num_graphs
                idxs = torch.arange(num_graphs, device=device) * \
                    3 + node_idx  # 每图该节点的行索引
                x[idxs, feat_j] = 0.0
                batch_masked = Data(
                    x=x, edge_index=batch.edge_index, y=batch.y, batch=batch.batch)
                out = self.model(batch_masked).detach().cpu().numpy()
                y = batch.y.view(-1).detach().cpu().numpy()
                ys.append(y)
                ps.append(out)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)
        if self.ds._label_minmax is not None:
            ys = self.ds.inverse_transform_label(ys)
            ps = self.ds.inverse_transform_label(ps)
        m = compute_metrics(ys, ps)
        return m["RMSE"]
