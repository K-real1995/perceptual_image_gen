from collections.abc import Callable
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from perceptual_gen.image_io import ImageTensor, clip_0_1, save_image


def train_step(
    image: tf.Variable,
    loss_func: Callable[[ImageTensor], tf.Tensor],
    optimizer: tf.keras.optimizers.Optimizer,
) -> float:
    with tf.GradientTape() as tape:
        loss_value = loss_func(image)
    gradient = tape.gradient(loss_value, image)
    optimizer.apply_gradients([(gradient, image)])
    image.assign(clip_0_1(image))
    return float(loss_value.numpy().item())


def optimize_image(
    content_image: tf.Tensor,
    loss_func: Callable[[ImageTensor], tf.Tensor],
    steps: int,
    learning_rate: float,
    seed: int,
    save_every: int = 0,
    snapshot_dir: str | Path | None = None,
) -> tuple[tf.Variable, list[float]]:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    image = tf.Variable(
        np.random.rand(*content_image.shape).astype(np.float32),
        trainable=True,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.99,
        epsilon=1e-1,
    )

    loss_history: list[float] = []
    progress = tqdm(range(steps), desc="Optimizing", unit="step")

    for step in progress:
        loss_value = train_step(image, loss_func, optimizer)
        loss_history.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")

        if save_every > 0 and step > 0 and step % save_every == 0 and snapshot_dir:
            snapshot_path = Path(snapshot_dir) / f"step_{step:04d}.png"
            save_image(image, snapshot_path)

    return image, loss_history
