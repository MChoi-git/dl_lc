import jax
from jax import numpy as jnp, random

# Linear regression:
#   y = x @ W + b

def model(params, x):
    W = params[0]
    b = params[1]
    return x @ W + b


def loss_fn(params, x_batched, y_batched):
    preds_batched = model(params, x_batched)
    loss = jnp.mean((preds_batched - y_batched) ** 2)
    return loss


def update(params, x_batched, y_batched, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batched, y_batched)
    update_fn = lambda p, g: p - lr * g
    params = jax.tree_util.tree_map(update_fn, params, grads)
    return params, loss


def main():
    k = random.PRNGKey(0)
    k1, k2, k3, k4, k5 = random.split(k, 5)
    x_batched = random.normal(k1, (128, 8))
    W_true = random.normal(k2, (8, 4))
    b_true = random.normal(k3, (4,))
    y_batched = model((W_true, b_true), x_batched)

    W = random.normal(k4, (8, 4))
    b = random.normal(k5, (4,))
    params = (W, b)

    x_batched = x_batched.reshape(32, -1, 8)
    y_batched = y_batched.reshape(32, -1, 4) 

    for _ in range(10):
        epoch_loss = 0
        for i in range(32):
            batch = x_batched[i]
            targets = y_batched[i]

            params, loss = update(params, batch, targets, 0.1)
            epoch_loss += loss

        epoch_loss = epoch_loss / 32
        print(f"Loss: {epoch_loss}")


if __name__ == "__main__":
    main()
