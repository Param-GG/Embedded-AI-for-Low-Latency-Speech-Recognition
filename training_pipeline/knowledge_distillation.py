import tensorflow as tf
from tensorflow.keras import layers, models
import preprocess_data
import numpy as np

# 1. Load and preprocess dataset with AUTOTUNE
train_ds, val_ds, class_names = preprocess_data.prepare_speech_commands_dataset(
    "./datasets/speech_commands_v0.02"
)

# Use AUTOTUNE for better training performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# The rest of the Knowledge Distillation code remains unchanged


# 1. Teacher Model (Larger Model)
def build_teacher_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


input_shape = (99, 12, 1)
teacher_model = build_teacher_model(input_shape, len(class_names))

# Compile and Train Teacher Model
teacher_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
teacher_model.fit(train_ds, validation_data=val_ds, epochs=1)
teacher_model.save("teacher_model.h5")


# 2. Student Model (Compact Model)
def build_student_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.DepthwiseConv2D((3, 3), activation="relu"),
            layers.Conv2D(64, (1, 1), activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


student_model = build_student_model(input_shape, len(class_names))


# Knowledge Distillation Loss
def distillation_loss(y_true, y_pred, y_teacher, alpha=0.7, temperature=5):
    """
    Combines soft-label (teacher) and hard-label (true) losses.
    """
    y_teacher = tf.nn.softmax(y_teacher / temperature)
    y_pred_soft = tf.nn.softmax(y_pred / temperature)

    soft_loss = tf.keras.losses.KLDivergence()(y_teacher, y_pred_soft)
    hard_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)

    return alpha * hard_loss + (1 - alpha) * soft_loss * (temperature**2)


def create_lr_scheduler():
    return tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * 0.95**epoch  # Exponential decay
    )


# Ensure teacher model is built before accessing input/output
dummy_input = tf.random.normal([1, *input_shape])
teacher_model(dummy_input)


def train_student_model(student_model, teacher_model, train_ds, val_ds, epochs=50):
    teacher_logits = tf.keras.Model(
        teacher_model.input, teacher_model.layers[-2].output
    )  # Extract logits

    student_model.compile(
        optimizer="adam",
        loss=lambda y_true, y_pred: distillation_loss(
            y_true,
            y_pred,
            teacher_logits.predict(train_ds.map(lambda x, y: x)),
            alpha=0.7,  # Adjusted alpha
            temperature=5,  # Adjusted temperature
        ),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    lr_scheduler = create_lr_scheduler()

    history = student_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler],
    )

    return history


history = train_student_model(student_model, teacher_model, train_ds, val_ds)


def quantize_student_model_for_arduino(
    student_model, output_path="student_model_quantized.h"
):
    # Initialize TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)

    # Set optimizations for integer-only quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Specify input and output types
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Provide a representative dataset for quantization
    def representative_dataset_gen():
        for x, _ in train_ds.take(
            100
        ):  # Use a small subset for representative sampling
            yield [tf.cast(x, tf.float32)]

    converter.representative_dataset = representative_dataset_gen

    # Convert the model
    tflite_quant_model = converter.convert()

    # Save the quantized model as a C header for Arduino
    with open(output_path, "w") as f:
        f.write("const unsigned char model[] = {\n")
        f.write(",".join([str(x) for x in tflite_quant_model]) + "\n")
        f.write("};\n")
        f.write(f"unsigned int model_len = {len(tflite_quant_model)};")
    print(f"Quantized model saved to {output_path}")


quantize_student_model_for_arduino(student_model)
