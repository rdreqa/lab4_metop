"""Графики для ноутбука и отчета."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import predict


def _save(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def plot_dataset(data, path):
    X = data["X_plot"]
    y = data["y"]

    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=24, alpha=0.85, edgecolor="k", linewidth=0.2)
    suffix = "" if data["plot_kind"] == "features" else " в PCA-проекции"
    plt.title(f"Датасет {data['name']}{suffix}")
    plt.xlabel("x1" if data["plot_kind"] == "features" else "PC1")
    plt.ylabel("x2" if data["plot_kind"] == "features" else "PC2")
    plt.grid(alpha=0.25)
    return _save(path)


def plot_history(history, title, path):
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="validation")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.legend()
    plt.grid(alpha=0.25)
    return _save(path)


def plot_accuracy(history, title, path):
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_accuracy"], label="train")
    plt.plot(history["epoch"], history["val_accuracy"], label="validation")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(alpha=0.25)
    return _save(path)


def plot_decision_boundary(model, data, title, path):
    """2D-граница. Для 5D датасета рисуется срез в PCA-пространстве."""
    X2 = data["X_plot"]
    y = data["y"]
    x_min, x_max = X2[:, 0].min() - 0.6, X2[:, 0].max() + 0.6
    y_min, y_max = X2[:, 1].min() - 0.6, X2[:, 1].max() + 0.6

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    if data["plot_kind"] == "features":
        grid_input = grid_2d
        suffix = ""
    else:
        grid_input = data["pca"].inverse_transform(grid_2d)
        suffix = " в PCA-проекции"

    zz = predict(model, grid_input).reshape(xx.shape)

    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, zz, alpha=0.25)
    plt.scatter(X2[:, 0], X2[:, 1], c=y, s=24, edgecolor="k", linewidth=0.2)
    plt.title(f"{title}{suffix}")
    plt.xlabel("x1" if data["plot_kind"] == "features" else "PC1")
    plt.ylabel("x2" if data["plot_kind"] == "features" else "PC2")
    plt.grid(alpha=0.2)
    return _save(path)


def plot_metric_bars(df, metric, title, path):
    labels = [f"{row.dataset}\n{row.model}" for row in df.itertuples()]
    values = df[metric].values

    plt.figure(figsize=(9, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(metric)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.25)
    return _save(path)
