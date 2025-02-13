import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot
from tensorflow.keras import mixed_precision
import numpy as np
import time
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_handling import prepare_speech_commands_dataset

def configure_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4608)]
                )
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("Mixed precision policy:", policy)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

configure_gpu()

def build_rcnn(input_shape, num_classes):
    """Build RCNN model for keyword spotting."""
    inputs = keras.Input(shape=input_shape)
    
    # Initial Conv layer
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Recurrent Conv Blocks
    for _ in range(2):
        recurrent = layers.Conv2D(64, (3, 3), padding='same')(x)
        recurrent = layers.BatchNormalization()(recurrent)
        recurrent = layers.Activation('relu')(recurrent)
        recurrent = layers.Conv2D(64, (3, 3), padding='same')(recurrent)
        recurrent = layers.BatchNormalization()(recurrent)
        x = layers.Add()([x, recurrent])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
    
    # Final layers
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

class ProgressBar(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        print(f"\nTraining for {self.epochs} epochs...")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{self.epochs}")
        self.train_progbar = tqdm(
            total=self.params['steps'],
            desc="Training",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [ETA {remaining}]'
        )

    def on_batch_end(self, batch, logs=None):
        self.train_progbar.update(1)
        self.train_progbar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'acc': f"{logs['accuracy']:.4f}"
        })

    def on_epoch_end(self, epoch, logs=None):
        self.train_progbar.close()
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        print(f" - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
        print(f" - Val Loss: {logs['val_loss']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")

def quantize_and_export(model, val_ds, output_path="rcnn_model.tflite"):
    def representative_dataset():
        for features, _ in val_ds.take(100):
            sample = tf.dtypes.cast(features, tf.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    c_output_path = output_path.replace('.tflite', '.h')
    with open(c_output_path, 'w') as f:
        f.write('#ifndef RCNN_MODEL_H\n#define RCNN_MODEL_H\n\n')
        f.write('const unsigned char rcnn_model_data[] = {\n')
        f.write(','.join(f'0x{b:02x}' for b in tflite_model))
        f.write('\n};\n')
        f.write(f'const unsigned int rcnn_model_len = {len(tflite_model)};\n')
        f.write('\n#endif // RCNN_MODEL_H')

    print(f"Quantized RCNN model size: {len(tflite_model) / 1024:.2f} KB")

def main():
    data_dir = 'datasets/speech_commands_v0_extracted'
    batch_size = 32
    train_ds, val_ds, test_ds, class_names = prepare_speech_commands_dataset(data_dir, batch_size=batch_size)
    
    print("\nDataset Check:")
    for features, labels in train_ds.take(1):
        print(f"Feature shape: {features.shape}")
        print(f"Feature min/max:", tf.reduce_min(features).numpy(), tf.reduce_max(features).numpy())
        print(f"Number of unique labels:", len(tf.unique(labels)[0]))
    print(f"Number of classes:", len(class_names))

    input_shape = (99, 12, 1)  # MFCC shape
    model = build_rcnn(input_shape, len(class_names))
    
    epochs = 30
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ProgressBar()],
        verbose=0
    )
    
    model.save("rcnn_model.h5")
    print("Original model saved as rcnn_model.h5")
    
    # Quantize and export
    quantize_and_export(model, val_ds, "arduino_model_rcnn.tflite")
    print("Quantized model exported as arduino_model_rcnn.tflite")

if __name__ == "__main__":
    main()