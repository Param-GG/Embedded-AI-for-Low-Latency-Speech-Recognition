import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
import preprocess_data

# Configuration
BATCH_SIZE = 32
EPOCHS = 150
NUM_CLASSES = None  # Will be set dynamically

# Load Dataset
train_ds, val_ds, class_names = preprocess_data.prepare_speech_commands_dataset(
    "./datasets/speech_commands_v0.02", batch_size=BATCH_SIZE
)
NUM_CLASSES = len(class_names)


# Relative Positional Embedding
class RelativePositionalEmbedding(layers.Layer):
    def __init__(self, units, max_distance=20, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.max_distance = max_distance

    def build(self, input_shape):
        self.relative_embedding = self.add_weight(
            shape=(2 * self.max_distance + 1, input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        indices = tf.range(-self.max_distance, self.max_distance + 1)
        relative_indices = tf.minimum(tf.maximum(indices, -seq_len + 1), seq_len - 1)
        return tf.gather(self.relative_embedding, relative_indices + self.max_distance)


# Multi-Head Self-Attention with Relative Positional Encoding
class ConformerAttention(layers.Layer):
    def __init__(self, units, num_heads=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.query_dense = layers.Dense(self.units)
        self.key_dense = layers.Dense(self.units)
        self.value_dense = layers.Dense(self.units)
        self.output_dense = layers.Dense(self.units)

        self.positional_encoding = RelativePositionalEmbedding(self.units)

        super().build(input_shape)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        depth_per_head = self.units // self.num_heads

        x = tf.reshape(x, (batch_size, length, self.num_heads, depth_per_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        scale = tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        attention_scores = tf.matmul(query, key, transpose_b=True) / scale

        # Add relative positional encoding
        positional_embedding = self.positional_encoding(inputs)
        attention_scores += positional_embedding

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, value)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(
            attention_output,
            (tf.shape(attention_output)[0], tf.shape(attention_output)[1], self.units),
        )

        return self.output_dense(attention_output)


def build_conformer_hybrid_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Initial CNN blocks
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for sequence modeling
    x = layers.Reshape((-1, x.shape[-1]))(x)

    # Conformer Blocks
    x = layers.Dense(128, activation="relu")(x)
    x = ConformerAttention(units=128, num_heads=4)(x)
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)

    # Classification Layer
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# Build and Compile Model
model = build_conformer_hybrid_model((99, 12, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Training
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


# Visualization Function
def print_acc_and_loss(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


print_acc_and_loss(history)


# TFLite Conversion and Export
def quantize_and_export(model, output_path="model.h"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)


quantize_and_export(model, "edge_device_deployment/conformer_model.tflite")
