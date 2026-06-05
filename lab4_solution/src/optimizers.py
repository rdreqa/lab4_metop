import numpy as np
PARAMS = ["W1", "b1", "W2", "b2"]


def init_optimizer_state(model):
    return {
        "t": 0,
        "m": {name: np.zeros_like(model[name]) for name in PARAMS},
        "v": {name: np.zeros_like(model[name]) for name in PARAMS},
    }

def sgd_step(model, grads, lr):
    for name in PARAMS:
        model[name] -= lr * grads[name]


def adam_step(model, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]

    for name in PARAMS:
        grad = grads[name]
        state["m"][name] = beta1 * state["m"][name] + (1.0 - beta1) * grad
        state["v"][name] = beta2 * state["v"][name] + (1.0 - beta2) * (grad ** 2)

        m_hat = state["m"][name] / (1.0 - beta1 ** t)
        v_hat = state["v"][name] / (1.0 - beta2 ** t)
        model[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def update(model, grads, optimizer, state, lr):
    if optimizer == "sgd":
        sgd_step(model, grads, lr)
    elif optimizer == "adam":
        adam_step(model, grads, state, lr)
    else:
        raise ValueError("optimizer должен быть 'sgd' или 'adam'")
