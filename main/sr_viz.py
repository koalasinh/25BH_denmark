# -*- coding: utf-8 -*-
# 说明：可视化工具，包含：
# 1) 三角形图结构绘制（节点坐标、标签）
# 2) 训练曲线绘制（train/val loss 与 val R2）
# 输出 PNG 到 data/processed_data 下

import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图像


class Visualizer:
    def __init__(self, out_path_func):
        """
        out_path_func: 一个函数，接受文件名，返回输出路径（如 OutputManager.path）
        """
        self._path = out_path_func

    def plot_triangle_graph(self, node_positions, node_labels, filename="triangle_graph.png"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4))
        # 画节点
        for idx, (x, y) in node_positions.items():
            plt.scatter([x], [y], s=300, c="#1f77b4")
            label = node_labels.get(idx, str(idx))
            plt.text(x, y+0.08, f"{idx}:{label}",
                     ha="center", va="bottom", fontsize=10)

        # 0 <-> 1（无向，画一条直线）
        x0, y0 = node_positions[0]
        x1, y1 = node_positions[1]
        plt.plot([x0, x1], [y0, y1], "k-", linewidth=2)

        # 2 -> 0 和 2 -> 1（单向，用箭头）
        x2, y2 = node_positions[2]
        # 画 2->0 箭头
        plt.annotate("", xy=(x0, y0), xytext=(x2, y2),
                     arrowprops=dict(arrowstyle="->", color="k", lw=2))
        # 画 2->1 箭头
        plt.annotate("", xy=(x1, y1), xytext=(x2, y2),
                     arrowprops=dict(arrowstyle="->", color="k", lw=2))

        plt.title("Directed Reaction Graph: 2->(0,1), 0<->1")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self._path(filename), dpi=150)
        plt.close()

    def plot_training_curves(self, history: Dict[str, List[float]], filename: str = "training_curves.png"):
        """
        history: 包含 keys: 'train_loss', 'val_loss', 'val_r2'
        """
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        val_r2 = history.get("val_r2", [])

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Loss 曲线
        axs[0].plot(train_loss, label="Train Loss")
        axs[0].plot(val_loss, label="Val Loss")
        axs[0].set_title("Loss Curves")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        # R2 曲线
        axs[1].plot(val_r2, label="Val R2", color="green")
        axs[1].set_title("Validation R2")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("R2")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(self._path(filename), dpi=150)
        plt.close()
