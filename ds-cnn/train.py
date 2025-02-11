import tensorflow as tf
from tensorflow.keras import layers, models
import time
from tqdm import tqdm
from data_preprocessing.dataset_handling import prepare_speech_commands_dataset

# Set GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_ds_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1)),
        layers.DepthwiseConv2D((3, 3), activation="relu"),
        layers.Conv2D(64, (1, 1), activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
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
            desc=f"Training",
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

def main():
    # Prepare data
    data_dir = 'datasets/speech_commands_v0_extracted'
    batch_size = 16
    train_ds, val_ds, test_ds, class_names = prepare_speech_commands_dataset(data_dir, batch_size=batch_size)
    
    # Print dataset info
    print("\nDataset Check:")
    for features, labels in train_ds.take(1):
        print(f"Feature shape: {features.shape}")
        print(f"Feature min/max:", tf.reduce_min(features).numpy(), tf.reduce_max(features).numpy())
        print(f"Number of unique labels:", len(tf.unique(labels)[0]))
    print(f"Number of classes:", len(class_names))

    # Build model
    input_shape = (99, 12, 1)  # MFCC shape
    model = build_ds_cnn(input_shape, len(class_names))
    
    # Train model
    epochs = 25
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ProgressBar()],
        verbose=0
    )
    
    # Save model
    model.save("ds_cnn_model.h5")

if __name__ == "__main__":
    main()