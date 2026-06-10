from collections.abc import Callable

import tensorflow as tf

from perceptual_gen.image_io import ImageTensor
from perceptual_gen.style_content_extractor import StyleAndContentExtractor


def build_style_content_loss(
    extractor: StyleAndContentExtractor,
    style_targets: dict[str, tf.Tensor],
    content_targets: dict[str, tf.Tensor],
    style_weight: float,
    content_weight: float,
    tv_weight: float,
) -> Callable[[ImageTensor], tf.Tensor]:
    mse = tf.keras.losses.MeanSquaredError()

    def loss(image: ImageTensor) -> tf.Tensor:
        features = extractor(image)
        style_loss = tf.add_n(
            [
                mse(features["style"][name], style_targets[name])
                for name in style_targets
            ]
        )
        style_loss *= 1.0 / len(style_targets)

        content_loss = tf.add_n(
            [
                mse(features["content"][name], content_targets[name])
                for name in content_targets
            ]
        )
        content_loss *= 1.0 / len(content_targets)

        tv_loss = tf.reduce_sum(tf.image.total_variation(image))
        return (
            style_weight * style_loss
            + content_weight * content_loss
            + tv_weight * tv_loss
        )

    return loss
