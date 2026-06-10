import tensorflow as tf

from perceptual_gen.image_io import ImageTensor


def get_vgg_layers_model(layer_names: list[str]) -> tf.keras.Model:
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)


class FeatureExtractor:
    def __init__(self, layers: list[str]):
        self.content_layers = layers
        self.model = get_vgg_layers_model(layers)

    def __call__(self, inputs: ImageTensor) -> dict[str, tf.Tensor]:
        scaled_inputs = tf.cast(inputs, tf.float32) * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(scaled_inputs)
        outputs = self.model(preprocessed_input)
        return {name: value for name, value in zip(self.content_layers, outputs)}
