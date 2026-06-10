import tensorflow as tf

from perceptual_gen.image_io import ImageTensor
from perceptual_gen.vgg_model import get_vgg_layers_model

DEFAULT_STYLE_LAYERS = (
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
)


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleExtractor:
    def __init__(self, layers: list[str] | None = None):
        self.style_layers = list(layers or DEFAULT_STYLE_LAYERS)
        self.model = get_vgg_layers_model(self.style_layers)

    def __call__(self, inputs: ImageTensor) -> dict[str, tf.Tensor]:
        scaled_inputs = tf.cast(inputs, tf.float32) * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(scaled_inputs)
        outputs = self.model(preprocessed_input)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        style_outputs = [gram_matrix(style_output) for style_output in outputs]
        return {name: value for name, value in zip(self.style_layers, style_outputs)}
