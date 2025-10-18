# -*- coding: utf-8 -*-
# 说明：路径解析、输出落盘、指标计算等通用工具。

import os
import json
import math
from typing import Dict
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import hashlib
from dataclasses import asdict, is_dataclass


class OutputManager:
    def __init__(self, data_dir: str, processed_subdir: str, save_outputs: bool = True):
        self.root_dir = data_dir
        self.out_dir = os.path.join(data_dir, processed_subdir)
        self.save_outputs = save_outputs

    def ensure_dirs(self):
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    def path(self, filename: str) -> str:
        return os.path.join(self.out_dir, filename)

    def save_json(self, filename: str, obj: dict):
        if not self.save_outputs:
            return
        self.ensure_dirs()
        with open(self.path(filename), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def save_csv(self, filename: str, df: pd.DataFrame):
        if not self.save_outputs:
            return
        self.ensure_dirs()
        df.to_csv(self.path(filename), index=False, encoding="utf-8")

    def save_model(self, filename: str, model_state: dict):
        if not self.save_outputs:
            return
        self.ensure_dirs()
        torch.save(model_state, self.path(filename))


def resolve_in_data_dir(data_dir: str, relative_or_abs: str) -> str:
    p = str(relative_or_abs)
    if os.path.isabs(p):
        return p
    return os.path.join(data_dir, p)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # 计算 MSE/RMSE/R2/MAE，前置过滤 NaN/inf
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        print("[警告] 计算指标时有效样本为空。")
        return {"MSE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "MAE": float("nan")}
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "R2": r2, "MAE": mae}


def config_to_dict(cfg) -> dict:
    # 将 Config 对象转 dict（简单起见，直接取 __dict__）
    # 如使用 dataclass，可用 asdict
    d = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith(
        "_") and not callable(getattr(cfg, k))}
    # 过滤掉非内置类型（如设备对象），转字符串
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)  # 尝试序列化
            out[k] = v
        except Exception:
            out[k] = str(v)
    return out


def make_run_id(cfg_dict: dict) -> str:
    # 时间戳 + 配置哈希的短码
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    h = hashlib.md5(json.dumps(cfg_dict, sort_keys=True).encode(
        "utf-8")).hexdigest()[:8]
    return f"{ts}-{h}"


class RunManager:
    """
    负责：
    - 生成 run_id 和 run 目录
    - 保存 config.json / metrics.json / training_history.json / best_model.pt / val_predictions.csv 等
    - 维护 runs_index.csv（总表）
    """

    def __init__(self, outman: OutputManager, cfg):
        self.outman = outman
        self.cfg_dict = config_to_dict(cfg)
        # run_id：优先 run_name，否则时间戳+哈希
        self.run_id = cfg.run_name.strip() or make_run_id(self.cfg_dict)
        self.run_dir = self.outman.path(f"runs/{self.run_id}")
        if self.outman.save_outputs:
            self.outman.ensure_dirs()
            os.makedirs(self.run_dir, exist_ok=True)

    def path(self, filename: str) -> str:
        return os.path.join(self.run_dir, filename)

    def save_config(self):
        if not self.outman.save_outputs:
            return
        with open(self.path("config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg_dict, f, ensure_ascii=False, indent=2)

    def save_metrics(self, metrics: dict):
        if not self.outman.save_outputs:
            return
        with open(self.path("metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def save_history(self, history: dict):
        if not self.outman.save_outputs:
            return
        with open(self.path("training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def save_model(self, state_dict: dict):
        if not self.outman.save_outputs:
            return
        torch.save(state_dict, self.path("best_model.pt"))

    def save_csv_local(self, filename: str, df: pd.DataFrame):
        if not self.outman.save_outputs:
            return
        df.to_csv(self.path(filename), index=False, encoding="utf-8")

    def append_runs_index(self, metrics: dict):
        """
        追加一行到 data/processed_data/runs_index.csv
        包含 run_id、时间、关键配置与指标，便于纵览历史运行。
        """
        if not self.outman.save_outputs:
            return
        idx_path = self.outman.path("runs_index.csv")
        row = {
            "run_id": self.run_id,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "pooling": self.cfg_dict.get("pooling"),
            "gat_hidden": self.cfg_dict.get("gat_hidden"),
            "gat_heads": self.cfg_dict.get("gat_heads"),
            "dropout": self.cfg_dict.get("dropout"),
            "lr": self.cfg_dict.get("lr"),
            "weight_decay": self.cfg_dict.get("weight_decay"),
            "epochs": self.cfg_dict.get("epochs"),
            "batch_size": self.cfg_dict.get("batch_size"),
            "use_huber": self.cfg_dict.get("use_huber"),
            "normalize_label": self.cfg_dict.get("normalize_label"),
            "standardize_features": self.cfg_dict.get("standardize_features"),
            # 指标
            "R2": metrics.get("R2"),
            "RMSE": metrics.get("RMSE"),
            "MAE": metrics.get("MAE"),
            "MSE": metrics.get("MSE"),
        }
        df_row = pd.DataFrame([row])
        if os.path.exists(idx_path):
            df_old = pd.read_csv(idx_path, encoding="utf-8")
            df_new = pd.concat([df_old, df_row], axis=0, ignore_index=True)
        else:
            df_new = df_row
        df_new.to_csv(idx_path, index=False, encoding="utf-8")
