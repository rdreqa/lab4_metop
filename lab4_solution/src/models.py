import numpy as np

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def brelu(x):
    """Bounded ReLU: min(max(x, 0), 1)."""
    return np.clip(x, 0.0, 1.0)


def activate(z, name):
    if name == "brelu":
        return brelu(z)
    if name == "sigmoid":
        return sigmoid(z)


def activation_grad(z, a, name):
    if name == "brelu":
        return ((z > 0.0) & (z < 1.0)).astype(float)
    if name == "sigmoid":
        return a * (1.0 - a)


def init_model(input_dim, hidden=16, activation="brelu", seed=42):
    rng = np.random.default_rng(seed)

    return {
        "W1": rng.normal(0.0, 0.35, size=(input_dim, hidden)),
        "b1": np.zeros((1, hidden)),
        "W2": rng.normal(0.0, 0.35, size=(hidden, 1)),
        "b2": np.zeros((1, 1)),
        "activation": activation,
    }


def forward(model, X):
    z1 = X @ model["W1"] + model["b1"]
    a1 = activate(z1, model["activation"])
    z2 = a1 @ model["W2"] + model["b2"]
    p = sigmoid(z2)
    cache = {"X": X, "z1": z1, "a1": a1, "p": p}
    return p, cache


def predict_proba(model, X):
    p, _ = forward(model, X)
    return p.ravel()


def predict(model, X):
    return (predict_proba(model, X) >= 0.5).astype(int)


def binary_cross_entropy(y_true, y_prob, eps=1e-12):
    y_true = y_true.reshape(-1, 1)
    y_prob = np.clip(y_prob.reshape(-1, 1), eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss)


def backward(model, cache, y_true, l2=0.0):
    X = cache["X"]
    z1 = cache["z1"]
    a1 = cache["a1"]
    p = cache["p"]
    y_true = y_true.reshape(-1, 1)
    n = len(y_true)

    dz2 = (p - y_true) / n
    dW2 = a1.T @ dz2 + l2 * model["W2"]
    db2 = dz2.sum(axis=0, keepdims=True)

    da1 = dz2 @ model["W2"].T
    dz1 = da1 * activation_grad(z1, a1, model["activation"])
    dW1 = X.T @ dz1 + l2 * model["W1"]
    db1 = dz1.sum(axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def score(model, X, y):
    prob = predict_proba(model, X)
    pred = (prob >= 0.5).astype(int)
    y = y.astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    accuracy = (tp + tn) / len(y)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "loss": binary_cross_entropy(y, prob),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
