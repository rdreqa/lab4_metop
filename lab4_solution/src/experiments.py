"""Запуск всех экспериментов лабораторной."""

from pathlib import Path

import pandas as pd

from .data import make_lab_datasets
from .models import score
from .plots import plot_accuracy, plot_dataset, plot_decision_boundary, plot_history, plot_metric_bars
from .training import train

MODEL_NAMES = {
    "brelu": "BReLU + Sigmoid",
    "sigmoid": "Перцептрон с 1 скрытым слоем",
}


def run_experiments(seed=467866, output_dir="outputs"):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    datasets = make_lab_datasets(seed)
    for key, data in datasets.items():
        plot_dataset(data, figures_dir / f"dataset_{key}.png")

    # Небольшая сетка: достаточно для подбора, но код и запуск остаются простыми.
    hidden_grid = [8, 16, 32]
    settings = [
        {"optimizer": "sgd", "lr": 0.05, "batch_size": 32},
        {"optimizer": "adam", "lr": 0.01, "batch_size": 32},
    ]

    all_rows = []
    best_rows = []

    for dataset_key, data in datasets.items():
        for activation in ["brelu", "sigmoid"]:
            candidates = []

            for hidden in hidden_grid:
                for setting in settings:
                    model, history = train(
                        data["X_train"],
                        data["y_train"],
                        data["X_val"],
                        data["y_val"],
                        activation=activation,
                        hidden=hidden,
                        optimizer=setting["optimizer"],
                        lr=setting["lr"],
                        batch_size=setting["batch_size"],
                        epochs=350,
                        patience=70,
                        seed=seed + hidden + (0 if activation == "brelu" else 1000),
                    )

                    val = score(model, data["X_val"], data["y_val"])
                    test = score(model, data["X_test"], data["y_test"])
                    row = {
                        "dataset_key": dataset_key,
                        "dataset": data["name"],
                        "model": MODEL_NAMES[activation],
                        "activation": activation,
                        "optimizer": setting["optimizer"],
                        "hidden": hidden,
                        "lr": setting["lr"],
                        "batch_size": setting["batch_size"],
                        "epochs_done": len(history["epoch"]),
                        "val_loss": val["loss"],
                        "val_accuracy": val["accuracy"],
                        "val_f1": val["f1"],
                        "test_loss": test["loss"],
                        "test_accuracy": test["accuracy"],
                        "test_precision": test["precision"],
                        "test_recall": test["recall"],
                        "test_f1": test["f1"],
                        "confusion_matrix": test["confusion_matrix"],
                        "model_obj": model,
                        "history": history,
                    }
                    candidates.append(row)
                    all_rows.append({k: v for k, v in row.items() if k not in ["model_obj", "history"]})

            best = max(candidates, key=lambda r: (r["val_f1"], r["val_accuracy"], -r["val_loss"]))
            best_rows.append({k: v for k, v in best.items() if k not in ["model_obj", "history"]})

            short = "brelu" if activation == "brelu" else "perceptron"
            title = f"{data['name']}: {MODEL_NAMES[activation]}"
            plot_history(best["history"], title, figures_dir / f"learning_loss_{dataset_key}_{short}.png")
            plot_accuracy(best["history"], title, figures_dir / f"learning_accuracy_{dataset_key}_{short}.png")
            plot_decision_boundary(best["model_obj"], data, title, figures_dir / f"decision_boundary_{dataset_key}_{short}.png")

    grid = pd.DataFrame(all_rows)
    best = pd.DataFrame(best_rows)
    grid.to_csv(tables_dir / "grid_results.csv", index=False)
    best.to_csv(tables_dir / "best_results.csv", index=False)

    plot_metric_bars(best, "test_accuracy", "Test accuracy лучших моделей", figures_dir / "metrics_accuracy_comparison.png")
    plot_metric_bars(best, "test_f1", "Test F1 лучших моделей", figures_dir / "metrics_f1_comparison.png")

    return {
        "datasets": datasets,
        "grid": grid,
        "best": best,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
    }
