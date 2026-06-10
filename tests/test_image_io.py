from pathlib import Path
from typing import cast

import tensorflow as tf

from perceptual_gen.image_io import ImageTensor, clip_0_1, load_image


def test_load_image_shape_and_range(sample_image_path: Path):
    image = load_image(sample_image_path, max_dim=256)

    assert image.shape.rank == 4
    assert image.shape[0] == 1
    assert image.shape[-1] == 3
    assert tf.reduce_max(image) <= 1.0
    assert tf.reduce_min(image) >= 0.0


def test_clip_0_1_clamps_values():
    image = cast(ImageTensor, tf.constant([[[[-0.5, 0.5, 1.5]]]]))
    clipped = clip_0_1(image)

    assert float(tf.reduce_min(clipped)) == 0.0
    assert float(tf.reduce_max(clipped)) == 1.0
