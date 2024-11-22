import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt 
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
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Compile and train
def train_model(model):
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    # Extract training and validation accuracy
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Print the accuracies for each epoch
    for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracy, val_accuracy), start=1):
        print(f"Epoch {epoch}: Training Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")

    return history

history = train_model(model)

# 4. Print accuracy and loss curves
def print_acc_and_loss(history):
    plt.subplot(111)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.show()

print_acc_and_loss(history=history)

# 5. Quantize and export
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
