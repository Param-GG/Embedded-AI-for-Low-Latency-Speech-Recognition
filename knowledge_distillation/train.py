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

def build_teacher_model(input_shape, num_classes):
    """Build a larger RCNN model as the teacher."""
    inputs = keras.Input(shape=input_shape)
    
    # Initial Conv layer
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # More complex recurrent conv blocks
    for filters in [128, 256, 256]:
        recurrent = layers.Conv2D(filters, (3, 3), padding='same')(x)
        recurrent = layers.BatchNormalization()(recurrent)
        recurrent = layers.Activation('relu')(recurrent)
        recurrent = layers.Conv2D(filters, (3, 3), padding='same')(recurrent)
        recurrent = layers.BatchNormalization()(recurrent)
        if filters != 128:
            x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.Add()([x, recurrent])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def build_student_model(input_shape, num_classes):
    """Build a smaller, efficient model as the student."""
    model = keras.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", strides=(1, 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (1, 1), activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

class DistillationModel(keras.Model):
    def __init__(self, student, teacher, temp=3.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temp = temp
        
    def compile(self, optimizer, metrics, distillation_loss_fn, student_loss_fn, alpha=0.1):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        self.student_loss_fn = student_loss_fn
        self.alpha = alpha
        
    def train_step(self, data):
        x, y = data
        
        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            
            # Compute losses
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temp, axis=1),
                tf.nn.softmax(student_predictions / self.temp, axis=1)
            )
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Combine losses
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        # Compute gradients and update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "distillation_loss": distillation_loss,
            "student_loss": student_loss,
            "total_loss": loss
        })
        return results

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
            'total_loss': f"{logs.get('total_loss', 0):.4f}",
            'accuracy': f"{logs.get('accuracy', 0):.4f}"
        })

    def on_epoch_end(self, epoch, logs=None):
        self.train_progbar.close()
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        for key, value in logs.items():
            print(f" - {key}: {value:.4f}")

def apply_pruning(model, train_ds, val_ds, epochs=10):
    """Apply pruning to the student model."""
    num_train_steps = np.ceil(sum(1 for _ in train_ds) * epochs).astype(np.int32)
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30,
            final_sparsity=0.70,
            begin_step=0,
            end_step=num_train_steps
        )
    }
    
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    pruned_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    pruned_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            ProgressBar(),
            tfmot.sparsity.keras.UpdatePruningStep()
        ],
        verbose=0
    )
    
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    return final_model

def quantize_and_export(model, val_ds, output_path="distilled_model.tflite"):
    """Quantize and export the model for Arduino."""
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
        f.write('#ifndef DISTILLED_MODEL_H\n#define DISTILLED_MODEL_H\n\n')
        f.write('const unsigned char distilled_model_data[] = {\n')
        f.write(','.join(f'0x{b:02x}' for b in tflite_model))
        f.write('\n};\n')
        f.write(f'const unsigned int distilled_model_len = {len(tflite_model)};\n')
        f.write('\n#endif // DISTILLED_MODEL_H')

    print(f"Quantized model size: {len(tflite_model) / 1024:.2f} KB")

def main():
    # Prepare dataset
    data_dir = 'datasets/speech_commands_v0_extracted'
    batch_size = 32
    train_ds, val_ds, test_ds, class_names = prepare_speech_commands_dataset(
        data_dir, batch_size=batch_size
    )
    
    input_shape = (99, 12, 1)  # MFCC shape
    num_classes = len(class_names)
    
    # Build and train teacher model
    print("\nTraining teacher model...")
    teacher_model = build_teacher_model(input_shape, num_classes)
    teacher_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    teacher_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[ProgressBar()],
        verbose=0
    )
    
    teacher_model.save("teacher_model.h5")
    print("Teacher model saved as teacher_model.h5")
    
    # Build student model
    print("\nInitializing knowledge distillation...")
    student_model = build_student_model(input_shape, num_classes)
    
    # Create distillation model
    distillation_model = DistillationModel(student_model, teacher_model, temp=3.0)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    distillation_model.compile(
        optimizer=optimizer,
        metrics=['accuracy'],
        distillation_loss_fn=keras.losses.KLDivergence(),
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
        alpha=0.1
    )
    
    # Train with knowledge distillation
    print("\nTraining student model with knowledge distillation...")
    distillation_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[ProgressBar()],
        verbose=0
    )
    
    # Save the original student model
    student_model = distillation_model.student
    student_model.save("student_model.h5")
    print("Student model saved as student_model.h5")
    
    # Apply pruning to the student model
    print("\nApplying pruning to student model...")
    pruned_student = apply_pruning(student_model, train_ds, val_ds, epochs=10)
    pruned_student.save("student_model_pruned.h5")
    print("Pruned student model saved as student_model_pruned.h5")
    
    # Quantize and export the pruned student model
    print("\nQuantizing pruned student model...")
    quantize_and_export(pruned_student, val_ds, "arduino_model_student_pruned.tflite")
    print("Quantized student model exported as arduino_model_student_pruned.tflite")

if __name__ == "__main__":
    main()