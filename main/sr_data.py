# -*- coding: utf-8 -*-
# 说明：三角形图构建与数据集，包含反应表鲁棒读取、特征数值化与标准化、处理后数据导出。

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from sr_config import Config
from sr_utils import resolve_in_data_dir, OutputManager


class GraphBuilder:
    @staticmethod
    def build_edge_index() -> torch.Tensor:
        # 节点定义：
        # 0: Imine, 1: Thiol, 2: Catalyst
        # 连接规则：
        # - 催化剂单向指向 0 和 1：2->0, 2->1
        # - 0 与 1 双向：0->1, 1->0
        edges = [
            (2, 0), (2, 1),  # Catalyst -> Imine / Thiol（单向）
            (0, 1), (1, 0),  # Imine <-> Thiol（双向）
        ]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


class ReactionDataset(Dataset):
    """
    每条样本构造：
    - x: [3, input_dim], 节点顺序 = [Imine(0), Thiol(1), Catalyst(2)]
    - edge_index: 三角形双向
    - y: [1]
    """

    def __init__(self, cfg: Config, outman: Optional[OutputManager] = None):
        super().__init__()
        self.cfg = cfg
        self.outman = outman
        self.edge_index = GraphBuilder.build_edge_index()
        self._load_all()

    def _infer_feat_cols(self, df: pd.DataFrame, id_col: str, override_cols: Optional[List[str]]) -> List[str]:
        if override_cols is not None:
            return list(override_cols)
        return [c for c in df.columns if c != id_col]

    def _df_to_map(self, df: pd.DataFrame, id_col: str, feat_cols: List[str]) -> Dict[str, np.ndarray]:
        mapping = {}
        for _, row in df.iterrows():
            key = str(row[id_col]).strip()
            vec = row[feat_cols].values.astype(np.float32)
            mapping[key] = vec
        return mapping

    def _standardize(self, df: pd.DataFrame, feat_cols: List[str]):
        # 数值化 -> 均值填补 -> 标准化(zscore/minmax) -> 兜底
        df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        col_means = df[feat_cols].mean()
        df[feat_cols] = df[feat_cols].fillna(col_means)
        if self.cfg.standardize_features:
            if self.cfg.feature_scale_mode == "zscore":
                mu = df[feat_cols].mean()
                sigma = df[feat_cols].std(ddof=0).replace(0.0, 1.0)
                df[feat_cols] = (df[feat_cols] - mu) / sigma
            elif self.cfg.feature_scale_mode == "minmax":
                mn = df[feat_cols].min()
                mx = df[feat_cols].max()
                denom = (mx - mn).replace(0.0, 1.0)
                df[feat_cols] = (df[feat_cols] - mn) / denom
            else:
                raise ValueError("feature_scale_mode 仅支持 'zscore' 或 'minmax'")
        df[feat_cols] = df[feat_cols].replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)

    def _load_all(self):
        cfg = self.cfg
        # 路径解析
        path_rxn = resolve_in_data_dir(cfg.data_dir, cfg.reactions_csv)
        path_imine = resolve_in_data_dir(cfg.data_dir, cfg.imine_feat_csv)
        path_thiol = resolve_in_data_dir(cfg.data_dir, cfg.thiol_feat_csv)
        path_catal = resolve_in_data_dir(cfg.data_dir, cfg.catalyst_feat_csv)

        # 读取反应表，列名小写+去空格；空串->NaN
        rxn = pd.read_csv(path_rxn, encoding="utf-8-sig")
        rxn.columns = [str(c).strip().lower() for c in rxn.columns]
        rxn = rxn.replace(r"^\s*$", np.nan, regex=True)

        label_col = str(cfg.label_col).strip().lower()
        col_thiol = str(cfg.rxn_col_thiol).strip().lower()
        col_imine = str(cfg.rxn_col_imine).strip().lower()
        col_catal = str(cfg.rxn_col_catalyst).strip().lower()

        required = [col_thiol, col_imine, col_catal, label_col]
        missing = [c for c in required if c not in rxn.columns]
        assert len(missing) == 0, f"反应表缺列：{missing}；实际列：{list(rxn.columns)}"

        before = len(rxn)
        rxn = rxn.dropna(subset=required).copy()
        if len(rxn) < before:
            print(f"[数据清洗] 丢弃含 NaN 行：{before - len(rxn)} 条。")
        rxn[label_col] = pd.to_numeric(rxn[label_col], errors="coerce")
        before = len(rxn)
        rxn = rxn.dropna(subset=[label_col]).reset_index(drop=True)
        if len(rxn) < before:
            print(f"[数据清洗] 丢弃标签不可解析行：{before - len(rxn)} 条。")

        # 三份特征表
        df_imine = pd.read_csv(path_imine, encoding="utf-8-sig")
        df_thiol = pd.read_csv(path_thiol, encoding="utf-8-sig")
        df_catal = pd.read_csv(path_catal, encoding="utf-8-sig")

        cols_imine = self._infer_feat_cols(
            df_imine, cfg.id_col_imine, cfg.feat_cols_imine)
        cols_thiol = self._infer_feat_cols(
            df_thiol, cfg.id_col_thiol, cfg.feat_cols_thiol)
        cols_catal = self._infer_feat_cols(
            df_catal, cfg.id_col_catalyst, cfg.feat_cols_catalyst)
        assert len(cols_imine) == cfg.input_dim
        assert len(cols_thiol) == cfg.input_dim
        assert len(cols_catal) == cfg.input_dim

        # 标准化
        self._standardize(df_imine, cols_imine)
        self._standardize(df_thiol, cols_thiol)
        self._standardize(df_catal, cols_catal)

        # 映射
        map_imine = self._df_to_map(df_imine, cfg.id_col_imine, cols_imine)
        map_thiol = self._df_to_map(df_thiol, cfg.id_col_thiol, cols_thiol)
        map_catal = self._df_to_map(df_catal, cfg.id_col_catalyst, cols_catal)

        # 标签缩放（可选）
        y_raw = rxn[label_col].astype(np.float32).to_numpy()
        if cfg.normalize_label:
            y_min, y_max = float(np.nanmin(y_raw)), float(np.nanmax(y_raw))
            self._label_minmax = (y_min, y_max)
            if y_max > y_min:
                y_scaled = (y_raw - y_min) / (y_max - y_min)
            else:
                y_scaled = np.zeros_like(y_raw)
        else:
            self._label_minmax = None
            y_scaled = y_raw.copy()

        self.graphs: List[Data] = []
        kept_rows = []
        miss_count = 0

        for i in range(len(rxn)):
            r = rxn.iloc[i]
            k_im, k_th, k_ca = str(r[col_imine]).strip(), str(
                r[col_thiol]).strip(), str(r[col_catal]).strip()
            if k_im not in map_imine or k_th not in map_thiol or k_ca not in map_catal:
                miss_count += 1
                continue
            x0, x1, x2 = map_imine[k_im], map_thiol[k_th], map_catal[k_ca]
            x = np.stack([x0, x1, x2], axis=0).astype(np.float32)
            yv = float(y_scaled[i])
            if not np.isfinite(yv):
                miss_count += 1
                continue
            data = Data(
                x=torch.from_numpy(x),
                edge_index=self.edge_index.clone(),
                y=torch.tensor([yv], dtype=torch.float32)
            )
            self.graphs.append(data)
            kept_rows.append({
                "thiol": k_th, "imine": k_im, "catalyst": k_ca,
                "label_raw": float(y_raw[i]), "label_scaled": float(y_scaled[i])
            })

        if miss_count > 0:
            print(
                f"[数据清洗] 因缺失/非法被跳过：{miss_count} 条；最终样本：{len(self.graphs)} 条。")

        if self.outman and self.cfg.save_outputs and self.cfg.save_processed_dataset:
            self.outman.save_csv("processed_reactions.csv",
                                 pd.DataFrame(kept_rows))

        if self.outman and self.cfg.save_outputs:
            meta = {"label_normalized": self._label_minmax is not None}
            if self._label_minmax is not None:
                meta["label_min"], meta["label_max"] = float(
                    self._label_minmax[0]), float(self._label_minmax[1])
            self.outman.save_json("label_meta.json", meta)

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]

    def inverse_transform_label(self, y_scaled: np.ndarray) -> np.ndarray:
        if self._label_minmax is None:
            return y_scaled
        y_min, y_max = self._label_minmax
        return y_scaled * (y_max - y_min) + y_min
