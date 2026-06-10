import numpy as np
import tensorflow as tf

from perceptual_gen.style_content_extractor import (
    DEFAULT_CONTENT_LAYERS,
    DEFAULT_STYLE_LAYERS,
    StyleAndContentExtractor,
)
from perceptual_gen.style_content_loss import build_style_content_loss


def test_style_and_content_extractor_structure(sample_image_tensor: tf.Tensor):
    extractor = StyleAndContentExtractor()
    features = extractor(sample_image_tensor)

    assert sorted(features.keys()) == ["content", "style"]
    assert list(features["style"].keys()) == list(DEFAULT_STYLE_LAYERS)
    assert list(features["content"].keys()) == list(DEFAULT_CONTENT_LAYERS)

    for gram in features["style"].values():
        shape = gram.shape.as_list()
        assert len(shape) == 3
        assert shape[1] == shape[2]

    for content_map in features["content"].values():
        assert len(content_map.shape) == 4


def test_style_content_loss_returns_scalar_and_gradient(sample_image_tensor: tf.Tensor):
    extractor = StyleAndContentExtractor()
    style_image = sample_image_tensor
    content_image = sample_image_tensor

    style_features = extractor(style_image)
    content_features = extractor(content_image)
    loss_func = build_style_content_loss(
        extractor=extractor,
        style_targets=style_features["style"],
        content_targets=content_features["content"],
        style_weight=100.0,
        content_weight=5.0,
        tv_weight=0.1,
    )

    image = tf.Variable(
        np.random.rand(*sample_image_tensor.shape).astype(np.float32),
        trainable=True,
    )

    with tf.GradientTape() as tape:
        loss_value = loss_func(image)
    gradient = tape.gradient(loss_value, image)

    assert loss_value.shape.rank == 0
    assert gradient is not None
