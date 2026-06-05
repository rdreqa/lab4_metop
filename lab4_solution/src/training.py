"""Цикл обучения: mini-batches, validation, early stopping."""

import copy

import numpy as np

from .models import backward, forward, init_model, score
from .optimizers import init_optimizer_state, update


def train(
    X_train,
    y_train,
    X_val,
    y_val,
    activation="brelu",
    hidden=16,
    optimizer="adam",
    lr=0.01,
    batch_size=32,
    epochs=300,
    l2=1e-4,
    patience=50,
    seed=42,
):
    """Обучить сеть и вернуть лучшую модель по validation loss."""
    rng = np.random.default_rng(seed)
    model = init_model(X_train.shape[1], hidden=hidden, activation=activation, seed=seed)
    opt_state = init_optimizer_state(model)

    best_model = copy.deepcopy(model)
    best_val_loss = float("inf")
    no_improve = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_train))

        for start in range(0, len(X_train), batch_size):
            idx = order[start:start + batch_size]
            _, cache = forward(model, X_train[idx])
            grads = backward(model, cache, y_train[idx], l2=l2)
            update(model, grads, optimizer, opt_state, lr)

        train_metrics = score(model, X_train, y_train)
        val_metrics = score(model, X_val, y_val)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_model, history
