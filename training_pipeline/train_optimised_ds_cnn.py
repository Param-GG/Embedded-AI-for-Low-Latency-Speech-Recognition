import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import preprocess_data
import matplotlib.pyplot as plt

# 1. Load and preprocess dataset
train_ds, val_ds, class_names = preprocess_data.prepare_speech_commands_dataset(
    "./datasets/speech_commands_v0.02",
)


# 2. Optimized DS-CNN Model
def build_optimized_ds_cnn(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                strides=(1, 1),
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.DepthwiseConv2D(
                (3, 3), activation="relu", kernel_regularizer=regularizers.l2(1e-4)
            ),
            layers.Conv2D(
                64, (1, 1), activation="relu", kernel_regularizer=regularizers.l2(1e-4)
            ),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


input_shape = (99, 12, 1)  # Example for MFCC features
model = build_optimized_ds_cnn(input_shape, len(class_names))

# 3. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# 4. Train Model
def train_model(model, train_ds, val_ds, epochs=100):
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Plot results
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title("Training vs Validation Accuracy")
    plt.show()

    return history


history = train_model(model, train_ds, val_ds)


# 5. Quantize and Export
def quantize_model(model, output_path="optimized_model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Optimized DS-CNN model exported to {output_path}")


quantize_model(model)
