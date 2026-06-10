from collections.abc import Callable

import tensorflow as tf

from perceptual_gen.image_io import ImageTensor
from perceptual_gen.style_extractor import StyleExtractor


def build_style_loss(
    extractor: StyleExtractor,
    style_targets: dict[str, tf.Tensor],
    tv_weight: float,
) -> Callable[[ImageTensor], tf.Tensor]:
    mse = tf.keras.losses.MeanSquaredError()

    def loss(image: ImageTensor) -> tf.Tensor:
        current_features = extractor(image)
        style_loss = tf.add_n(
            [
                mse(current_features[name], style_targets[name])
                for name in style_targets
            ]
        )
        style_loss *= 1.0 / len(style_targets)
        tv_loss = tf.reduce_sum(tf.image.total_variation(image))
        return style_loss + tv_loss * tv_weight

    return loss
