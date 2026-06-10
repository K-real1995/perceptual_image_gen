from collections.abc import Callable

import tensorflow as tf

from perceptual_gen.image_io import ImageTensor
from perceptual_gen.vgg_extractor import FeatureExtractor


def build_perceptual_loss(
    extractor: FeatureExtractor,
    target_features: dict[str, tf.Tensor],
    tv_weight: float,
) -> Callable[[ImageTensor], tf.Tensor]:
    mse = tf.keras.losses.MeanSquaredError()

    def loss(image: ImageTensor) -> tf.Tensor:
        current_features = extractor(image)
        perceptual_loss = tf.add_n(
            [
                mse(current_features[name], target_features[name])
                for name in target_features
            ]
        )
        perceptual_loss *= 1.0 / len(target_features)
        tv_loss = tf.reduce_sum(tf.image.total_variation(image))
        return perceptual_loss + tv_loss * tv_weight

    return loss
