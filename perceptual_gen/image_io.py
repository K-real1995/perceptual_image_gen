from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ImageTensor: TypeAlias = tf.Tensor | tf.Variable


def clip_0_1(image: ImageTensor) -> tf.Tensor:
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def load_image(path: str | Path, max_dim: int = 256) -> tf.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)
    long_dim = tf.cast(tf.reduce_max(shape[:2]), tf.float32)
    scale = tf.cast(max_dim, tf.float32) / long_dim
    new_height = tf.cast(tf.round(tf.cast(shape[0], tf.float32) * scale), tf.int32)
    new_width = tf.cast(tf.round(tf.cast(shape[1], tf.float32) * scale), tf.int32)
    image = tf.image.resize(image, [new_height, new_width])
    image = image[tf.newaxis, :]
    return image


def tensor_to_numpy(image: ImageTensor) -> np.ndarray:
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    return np.asarray(image.numpy())


def save_image(image: ImageTensor | np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = tensor_to_numpy(image) if isinstance(image, (tf.Tensor, tf.Variable)) else image
    plt.imsave(path, np.clip(array, 0.0, 1.0))
