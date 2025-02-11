import tensorflow as tf
from tensorflow.keras import layers, models
from dataset import prepare_speech_commands_dataset  # Assuming you have a dataset prep function
import time
from tqdm import tqdm  # Progress bar
import numpy as np

# Set GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define DS-CNN Model
def build_ds_cnn(input_shape):
    model = models.Sequential()
    
    # Example DS-CNN architecture (simplified)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(12, activation='softmax'))  # Assuming 12 classes
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare data
data_dir = 'datasets/speech_commands_v0_extracted'
batch_size = 32
train_ds, val_ds, test_ds, class_names = prepare_speech_commands_dataset(data_dir, batch_size=batch_size)

# Build model
input_shape = (None, 99, 12, 1)  # Example input shape
model = build_ds_cnn(input_shape)

# Define a custom callback to show progress and metrics after each epoch
class ProgressBar(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        logs['epoch_time'] = epoch_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

    def on_batch_end(self, batch, logs=None):
        current_epoch = self.params['epoch']
        total_batches = self.params['steps']
        current_batch = batch + 1

        if current_batch == total_batches:
            eta = (time.time() - self.epoch_start_time) * (total_batches - current_batch) / current_batch
            eta_minutes = eta / 60
            print(f"ETA: {eta_minutes:.2f} minutes")
        else:
            time_elapsed = time.time() - self.epoch_start_time
            print(f"Time Elapsed for this batch: {time_elapsed:.2f}s")

# Fit model with progress bar and metrics
epochs = 10
model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0, callbacks=[ProgressBar()])

# Save model
model.save("ds_cnn_model.h5")
