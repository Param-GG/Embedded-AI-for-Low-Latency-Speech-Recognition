import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import preprocess_data

# 1. Load and preprocess dataset
train_ds, val_ds, test_ds, class_names = (
    preprocess_data.prepare_speech_commands_dataset("./datasets/speech_commands_v0.02")
)


# 2. Build DS-CNN model
def build_model(input_shape):
    model = models.Sequential(
        [
            layers.InputLayer(shape=input_shape),
            layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1)),
            layers.DepthwiseConv2D((3, 3), activation="relu"),
            layers.Conv2D(64, (1, 1), activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(len(class_names), activation="softmax"),
        ]
    )
    return model


input_shape = (99, 12, 1)  # Adjust based on MFCC output
model = build_model(input_shape)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# 3. Compile and train
def train_model(model):
    history = model.fit(train_ds, validation_data=val_ds, epochs=2)

    # Extract training and validation accuracy
    train_accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    # Print the accuracies for each epoch
    for epoch, (train_acc, val_acc) in enumerate(
        zip(train_accuracy, val_accuracy), start=1
    ):
        print(
            f"Epoch {epoch}: Training Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}"
        )

    return history


history = train_model(model)


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


print_acc_and_loss(history=history)


# 5. Evaluate on test set
def evaluate_model(model, test_ds):
    y_true = []
    y_pred = []

    for feats, labels in test_ds:
        y_true.extend(labels)
        y_pred_prob = model.predict(feats)
        y_pred.extend(np.argmax(y_pred_prob, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))


evaluate_model(model, test_ds)


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


quantize_and_export(model, "edge_device_deployment\keyword_spotting\model.h")

# 7.
