import numpy as np
import tensorflow as tf

from perceptual_gen.loss import build_perceptual_loss
from perceptual_gen.vgg_extractor import FeatureExtractor


def test_loss_returns_scalar_and_gradient(sample_image_tensor: tf.Tensor):
    layer = "block2_conv2"
    extractor = FeatureExtractor([layer])
    target_features = extractor(sample_image_tensor)
    loss_func = build_perceptual_loss(extractor, target_features, tv_weight=0.01)

    image = tf.Variable(
        np.random.rand(*sample_image_tensor.shape).astype(np.float32),
        trainable=True,
    )

    with tf.GradientTape() as tape:
        loss_value = loss_func(image)
    gradient = tape.gradient(loss_value, image)

    assert loss_value.shape.rank == 0
    assert gradient is not None
