import tensorflow as tf

from perceptual_gen.image_io import ImageTensor
from perceptual_gen.style_extractor import DEFAULT_STYLE_LAYERS, gram_matrix
from perceptual_gen.vgg_model import get_vgg_layers_model

DEFAULT_CONTENT_LAYERS = ("block4_conv2",)

StyleContentFeatures = dict[str, dict[str, tf.Tensor]]


class StyleAndContentExtractor:
    def __init__(
        self,
        style_layers: list[str] | None = None,
        content_layers: list[str] | None = None,
    ):
        self.style_layers = list(style_layers or DEFAULT_STYLE_LAYERS)
        self.content_layers = list(content_layers or DEFAULT_CONTENT_LAYERS)
        layer_names = self.style_layers + self.content_layers
        self.model = get_vgg_layers_model(layer_names)
        self.model.trainable = False

    def __call__(self, inputs: ImageTensor) -> StyleContentFeatures:
        scaled_inputs = tf.cast(inputs, tf.float32) * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(scaled_inputs)
        outputs = self.model(preprocessed_input)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        style_outputs = outputs[: len(self.style_layers)]
        content_outputs = outputs[len(self.style_layers) :]

        return {
            "style": {
                name: gram_matrix(output, normalize_channels=True)
                for name, output in zip(self.style_layers, style_outputs)
            },
            "content": {
                name: output
                for name, output in zip(self.content_layers, content_outputs)
            },
        }
