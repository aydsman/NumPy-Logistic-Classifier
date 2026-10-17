import numpy as np
import matplotlib.pyplot as plt

SEED = 42
N_DOTS = 200
CENTER_0 = (0.0, 0.0)
CENTER_1 = (4.0, 4.0)
SPREAD = 1.5
TRAIN_FRACTION = 0.8
LEARNING_RATE = 0.1
EPOCHS = 500
PRINT_EVERY = 10


# ---------- Helper functions (the math) ----------

def make_data(rng):
    """Returns X (200x2 matrix of [x1, x2]) and y (200 labels of 0 or 1)."""
    half = N_DOTS // 2
    X0 = rng.normal(loc=CENTER_0, scale=SPREAD, size=(half, 2))
    X1 = rng.normal(loc=CENTER_1, scale=SPREAD, size=(half, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(half), np.ones(half)])
    return X, y


def linear(X, w, b):
    return X @ w + b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce_loss(y, p):
    # log(0) is -infinity, so p must stay strictly between 0 and 1
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradients(X, y, p):
    errors = p - y
    dw = X.T @ errors / len(y)
    db = np.mean(errors)
    return dw, db


# ---------- The four big functions ----------

def split(X, y, rng):
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    cut = int(len(y) * TRAIN_FRACTION)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def train_step(X, y, w, b, lr):
    """One epoch: guess, measure loss, compute gradients, update. Returns new w, b and the loss."""
    p = sigmoid(linear(X, w, b))
    loss = bce_loss(y, p)

    dw, db = gradients(X, y, p)
    w = w - lr * dw
    b = b - lr * db
    return w, b, loss


def train(X, y, lr, epochs):
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []

    for epoch in range(1, epochs + 1):
        w, b, loss = train_step(X, y, w, b, lr)
        losses.append(loss)

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:4d} | loss {loss:.4f}")

    return w, b, losses


def predict(X, w, b):
    p = sigmoid(linear(X, w, b))
    return (p >= 0.5).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ---------- Plots ----------

def plot_loss(losses):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (BCE)")
    plt.title("Loss curve")


def plot_boundary(X, y, w, b):
    plt.figure()
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="class 0", alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="class 1", alpha=0.7)

    # The boundary is where z = 0:  w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b) / w2
    x1_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_line = -(w[0] * x1_line + b) / w[1]
    plt.plot(x1_line, x2_line, "k--", label="decision boundary")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision boundary (all 200 dots)")
    plt.legend()


# ---------- Run everything ----------

def main():
    rng = np.random.default_rng(SEED)

    X, y = make_data(rng)
    X_train, y_train, X_test, y_test = split(X, y, rng)

    w, b, losses = train(X_train, y_train, LEARNING_RATE, EPOCHS)

    train_acc = accuracy(y_train, predict(X_train, w, b))
    test_acc = accuracy(y_test, predict(X_test, w, b))

    print()
    print(f"Learned w = [{w[0]:.4f}, {w[1]:.4f}], b = {b:.4f}")
    print(f"Train accuracy: {train_acc:.2%} ({len(y_train)} dots)")
    print(f"Test accuracy:  {test_acc:.2%} ({len(y_test)} dots)")

    plot_loss(losses)
    plot_boundary(X, y, w, b)
    plt.show()


if __name__ == "__main__":
    main()
