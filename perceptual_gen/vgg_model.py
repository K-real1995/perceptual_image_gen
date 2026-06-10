import tensorflow as tf


def get_vgg_layers_model(layer_names: list[str]) -> tf.keras.Model:
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)
