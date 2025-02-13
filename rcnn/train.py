import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_model_optimization as tfmot
import time
from tqdm import tqdm
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_handling import prepare_speech_commands_dataset

# Set GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_rcnn(input_shape, num_classes):
    """
    Build an RCNN model that extracts spatial features with Conv layers and
    captures temporal dependencies using a GRU.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Convolutional feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Reshape for recurrent processing:
    # Suppose the remaining height dimension is treated as "time" and width*channels as features.
    shape = tf.keras.backend.int_shape(x)  # (batch, height, width, channels)
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    
    # GRU to capture temporal dependencies
    x = layers.GRU(64, return_sequences=False)(x)
    
    # Final classification layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
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

def quantize_and_export(model, val_ds, output_path="model.tflite"):
    """
    Quantize model to int8 and export for Arduino.
    """
    def representative_dataset():
        # Use validation dataset for calibration
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

    # Save as .tflite file
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Save as C header file for Arduino
    c_output_path = output_path.replace('.tflite', '.h')
    with open(c_output_path, 'w') as f:
        f.write('#ifndef MODEL_H\n#define MODEL_H\n\n')
        f.write('const unsigned char model_data[] = {\n')
        f.write(','.join(f'0x{b:02x}' for b in tflite_model))
        f.write('\n};\n')
        f.write(f'const unsigned int model_len = {len(tflite_model)};\n')
        f.write('\n#endif // MODEL_H')

    print(f"Quantized model size: {len(tflite_model) / 1024:.2f} KB")

def main():
    # Prepare data
    data_dir = 'datasets/speech_commands_v0_extracted'
    batch_size = 16
    train_ds, val_ds, test_ds, class_names = prepare_speech_commands_dataset(data_dir, batch_size=batch_size)
    
    # Print dataset info
    print("\nDataset Check:")
    for features, labels in train_ds.take(1):
        print(f"Feature shape: {features.shape}")
        print(f"Feature min/max: {tf.reduce_min(features).numpy()}, {tf.reduce_max(features).numpy()}")
        print(f"Number of unique labels: {len(tf.unique(labels)[0])}")
    print(f"Number of classes: {len(class_names)}")

    # Build the original RCNN model
    input_shape = (99, 12, 1)  # MFCC shape
    base_model = build_rcnn(input_shape, len(class_names))
    
    # Train the original RCNN model
    epochs = 25
    base_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ProgressBar()],
        verbose=0
    )
    
    # Save the original, unpruned, unquantized RCNN model
    base_model.save("rcnn_model.h5")
    print("Original RCNN model saved as rcnn_model.h5")
    
    # ----- Now, Prune and Quantize -----
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    num_train_steps = np.ceil(sum(1 for _ in train_ds) * epochs).astype(np.int32)
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30,
            final_sparsity=0.70,
            begin_step=0,
            end_step=num_train_steps
        )
    }
    
    # Wrap the original RCNN model with pruning
    pruned_model = prune_low_magnitude(base_model, **pruning_params)
    
    # Recompile the pruned model
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Further train the pruned model for additional epochs to fine-tune pruning
    additional_epochs = 10
    pruned_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=additional_epochs,
        callbacks=[ProgressBar(), tfmot.sparsity.keras.UpdatePruningStep()],
        verbose=0
    )
    
    # Strip pruning wrappers from the pruned model
    final_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    final_pruned_model.save("rcnn_model_pruned.h5")
    print("Pruned RCNN model saved as rcnn_model_pruned.h5")
    
    # Quantize and export the pruned model for Arduino
    quantize_and_export(final_pruned_model, val_ds, "arduino_model_rcnn_pruned.tflite")
    print("Pruned and quantized RCNN model exported as arduino_model_rcnn_pruned.tflite")

if __name__ == "__main__":
    main()
