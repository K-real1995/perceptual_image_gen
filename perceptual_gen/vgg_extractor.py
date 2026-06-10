import tensorflow as tf

from perceptual_gen.image_io import ImageTensor
from perceptual_gen.vgg_model import get_vgg_layers_model


class FeatureExtractor:
    def __init__(self, layers: list[str]):
        self.content_layers = layers
        self.model = get_vgg_layers_model(layers)

    def __call__(self, inputs: ImageTensor) -> dict[str, tf.Tensor]:
        scaled_inputs = tf.cast(inputs, tf.float32) * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(scaled_inputs)
        outputs = self.model(preprocessed_input)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        return {name: value for name, value in zip(self.content_layers, outputs)}
