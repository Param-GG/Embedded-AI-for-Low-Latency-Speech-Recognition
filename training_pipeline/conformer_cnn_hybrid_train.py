import tensorflow as tf
from tensorflow.keras import layers, models
import preprocess_data


# 1. Define the Custom Conformer Layer
class ConformerLayer(layers.Layer):
    def __init__(self, filters, kernel_size, dropout_rate=0.1, **kwargs):
        super(ConformerLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Depthwise Separable Convolution
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size, padding="same", activation="relu"
        )
        self.pointwise_conv = layers.Conv2D(
            filters, (1, 1), padding="same", activation="relu"
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = layers.LayerNormalization()

        # Feed-Forward Network
        self.ffn1 = layers.Dense(filters * 4, activation="relu")
        self.ffn2 = layers.Dense(filters, activation="relu")

    def call(self, inputs, training=None):
        # Depthwise Separable Convolution
        conv = self.depthwise_conv(inputs)
        conv = self.pointwise_conv(conv)
        conv = self.dropout(conv, training=training)
        conv = conv + inputs  # Residual connection
        conv = self.layer_norm(conv)

        # Feed-Forward Network
        ffn = self.ffn1(conv)
        ffn = self.ffn2(ffn)
        ffn = self.dropout(ffn, training=training)
        ffn = ffn + conv  # Residual connection
        ffn = self.layer_norm(ffn)

        return ffn


# 2. Build the Conformer Model
def build_conformer_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = inputs
    # Stack Conformer layers
    for _ in range(3):  # Stack 3 Conformer layers
        x = ConformerLayer(filters=64, kernel_size=(3, 3))(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


# 3. Prepare the Dataset
train_ds, val_ds, class_names = preprocess_data.prepare_speech_commands_dataset(
    "./datasets/speech_commands_v0.02"
)

train_ds = train_ds.map(
    preprocess_data.preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.map(
    preprocess_data.preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

input_shape = (99, 12, 1)  # Adjust based on MFCC output
num_classes = len(class_names)

# 4. Compile and Train the Model
model = build_conformer_model(input_shape, num_classes)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(train_ds, validation_data=val_ds, epochs=1)


# 5. Quantize the Model for Deployment
def quantize_and_export(model, output_path="model_conformer.tflite"):
    # Create a representative dataset generator for quantization
    def representative_dataset():
        for batch, label in val_ds.take(100):
            yield [batch.numpy()]

    # Convert to TensorFlow Lite model with integer-only quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Save the TFLite model
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    # Export as a C array for Arduino
    with open(output_path.replace(".tflite", ".h"), "w") as f:
        f.write("#include <stddef.h>\n\n")
        f.write("const unsigned char model_conformer[] = {")
        f.write(",".join(f"0x{b:02x}" for b in tflite_model))
        f.write("};\n")
        f.write(f"const unsigned int model_conformer_len = {len(tflite_model)};\n")


quantize_and_export(model, "edge_device_deployment/model_conformer.h")
