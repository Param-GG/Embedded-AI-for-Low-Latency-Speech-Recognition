import os
import tensorflow as tf
from tensorflow.keras import layers, models
import preprocess_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Set constants
SAVED_MODEL_DIR = "rcnn_model_speech_commands"
TFLITE_MODEL_PATH = "rcnn_model_quantized.tflite"
SAVED_MODEL_PATH = "rcnn_model_speech_commands.h5"


# 1. Prepare Dataset
def prepare_datasets():
    # 1. Load and preprocess dataset
    train_ds, val_ds, test_ds, class_names = (
        preprocess_data.prepare_speech_commands_dataset(
            "./datasets/speech_commands_v0.02"
        )
    )

    return train_ds, val_ds, test_ds, class_names
    return train_ds, val_ds, test_ds, class_names


# 2. Build RCNN Model
def build_rcnn_model(input_shape, num_classes):
    """
    Builds an RCNN model.
    """
    inputs = layers.Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for recurrent layer
    x = layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)

    # Recurrent layer
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


# 3. Train Model
def train_model(model, train_ds, val_ds, epochs=10):
    """
    Trains the RCNN model.
    """
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )
    return model, history


# # 4. Quantize Model
# def quantize_model(model, representative_data_gen):
#     """
#     Converts the trained model to a TensorFlow Lite model with quantization.
#     """
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.representative_dataset = representative_data_gen
#     tflite_model = converter.convert()

#     with open(TFLITE_MODEL_PATH, "wb") as f:
#         f.write(tflite_model)
#     print(f"Quantized model saved to {TFLITE_MODEL_PATH}")


# def representative_data_gen():
#     """
#     Generator for representative dataset used for quantization.
#     """
#     for mfccs, _ in train_ds.take(100):
#         yield [mfccs]


# 4. Print accuracy and loss curves
def print_acc_and_loss(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 6. Quantize and export
def quantize_and_export(model, output_path="model.tflite"):
    import numpy as np

    # Create a representative dataset generator for quantization
    def representative_dataset():
        for _ in range(100):
            # Replace with actual representative MFCC input samples
            data = np.random.rand(1, 99, 12, 1).astype(np.float32)
            yield [data]

    # Convert to TensorFlow Lite model with integer-only quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # Quantize inputs to int8
    converter.inference_output_type = tf.int8  # Quantize outputs to int8

    tflite_model = converter.convert()

    # Save the TFLite model in binary format for Arduino deployment
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    # Export as a C array for Arduino
    with open(output_path.replace(".tflite", ".h"), "w") as f:
        # f.write("#include <stddef.h>\n\n")
        f.write("const unsigned char model_data[] = {\n")
        f.write(",".join(f"0x{b:02x}" for b in tflite_model) + "\n")
        f.write("};\n")
        f.write(f"const unsigned int model_data_len = {len(tflite_model)};\n")

    print(f"Quantized model exported to {output_path}")


# 5. Evaluate on test set
def evaluate_model(model, test_ds):
    y_true = []
    y_pred = []

    for feats, labels in test_ds:
        y_true.extend(labels)
        y_pred_prob = model.predict(feats)
        y_pred.extend(np.argmax(y_pred_prob, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))


# 5. Main Function
if __name__ == "__main__":
    # Dataset preparation
    train_ds, val_ds, test_ds, class_names = prepare_datasets()
    train_ds, val_ds, test_ds, class_names = prepare_datasets()

    # Build model
    # input_shape = [None, preprocess_data.preprocess_audio.NUM_MFCCS, 1]
    input_shape = [None, 12, 1]
    model = build_rcnn_model(input_shape, len(class_names))

    # Train the model
    model, history = train_model(model, train_ds, val_ds, epochs=10)

    # Save the trained model
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    model.save(os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_PATH))
    print(f"Trained model saved to {SAVED_MODEL_DIR}")

    print_acc_and_loss(history=history)

    # # Quantize and export the model
    # quantize_model(model, representative_data_gen)
    evaluate_model(model, test_ds)
    quantize_and_export(model, "edge_device_deployment\keyword_spotting\rcnn_model.h")

    print("Training, quantization, and export completed.")
