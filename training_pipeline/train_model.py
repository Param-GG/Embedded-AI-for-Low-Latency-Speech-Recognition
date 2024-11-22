import tensorflow as tf
from tensorflow.keras import layers, models
import preprocess_data

# 1. Load and preprocess dataset
train_ds, val_ds, class_names = preprocess_data.prepare_speech_commands_dataset("./dataset/speech_commands_v0.02")

# 2. Build DS-CNN model
def build_model(input_shape):
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1)),
        layers.DepthwiseConv2D((3, 3), activation='relu'),
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(len(class_names), activation='softmax')
    ])
    return model

input_shape = (99, 12, 1)  # Adjust based on MFCC output
model = build_model(input_shape)

# 3. Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 4. Quantize and export
def quantize_and_export(model, output_path="model.h"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, "w") as f:
        f.write("const unsigned char model[] = {\n")
        f.write(",".join([str(x) for x in tflite_model]) + "\n")
        f.write("};\n")
        f.write(f"unsigned int model_len = {len(tflite_model)};")

quantize_and_export(model, "../edge_device_deployment/model.h")
