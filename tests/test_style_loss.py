import numpy as np
import tensorflow as tf

from perceptual_gen.style_extractor import StyleExtractor
from perceptual_gen.style_loss import build_style_loss


def test_gram_matrix_shape(sample_image_tensor: tf.Tensor):
    extractor = StyleExtractor(["block1_conv1"])
    style_features = extractor(sample_image_tensor)
    gram = style_features["block1_conv1"]

    assert len(gram.shape) == 3
    assert gram.shape[0] == 1
    assert gram.shape[1] == gram.shape[2]


def test_style_loss_returns_scalar_and_gradient(sample_image_tensor: tf.Tensor):
    extractor = StyleExtractor(["block1_conv1", "block2_conv1"])
    style_targets = extractor(sample_image_tensor)
    loss_func = build_style_loss(extractor, style_targets, tv_weight=1e4)

    image = tf.Variable(
        np.random.rand(*sample_image_tensor.shape).astype(np.float32),
        trainable=True,
    )

    with tf.GradientTape() as tape:
        loss_value = loss_func(image)
    gradient = tape.gradient(loss_value, image)

    assert loss_value.shape.rank == 0
    assert gradient is not None
