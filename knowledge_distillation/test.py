import tensorflow as tf
import numpy as np
import sys, os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import dataset preparation function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_handling import prepare_speech_commands_dataset

def load_test_data(batch_size=16):
    data_dir = 'datasets/speech_commands_v0_extracted'
    _, _, test_ds, class_names = prepare_speech_commands_dataset(data_dir, batch_size=batch_size)
    return test_ds, class_names

def evaluate_keras_model(model, test_ds, model_name):
    """Evaluate a Keras model and measure inference time."""
    y_true, y_pred = [], []
    inference_times = []
    
    print(f"\nEvaluating {model_name}...")
    for features, labels in test_ds:
        # Measure inference time
        start_time = time.time()
        predictions = model.predict(features, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
        
        preds = np.argmax(predictions, axis=1)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
    
    return np.array(y_true), np.array(y_pred), inference_times

def evaluate_tflite_model(interpreter, test_ds, model_name):
    """Evaluate a TFLite model and measure inference time."""
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    y_true, y_pred = [], []
    inference_times = []
    
    print(f"\nEvaluating {model_name}...")
    for features, labels in test_ds:
        features_np = features.numpy()
        input_batch = np.round(features_np / input_scale + input_zero_point).astype(np.int8)
        
        batch_size_actual = input_batch.shape[0]
        for i in range(batch_size_actual):
            sample = np.expand_dims(input_batch[i], axis=0)
            
            # Measure inference time
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output_int8 = interpreter.get_tensor(output_details[0]['index'])
            output_float = (output_int8.astype(np.float32) - output_zero_point) * output_scale
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            pred = np.argmax(output_float)
            y_pred.append(pred)
            
        y_true.extend(labels.numpy())
    
    return np.array(y_true), np.array(y_pred), inference_times

def plot_confusion_matrix(y_true, y_pred, class_names, title, cmap="Blues"):
    """Plot confusion matrix with given parameters."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap=cmap)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def calculate_metrics(y_true, y_pred, class_names, model_name, inference_times=None):
    """Calculate and display various metrics."""
    print(f"\nClassification Report ({model_name}):")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        print(f"Average Inference Time: {avg_inference_time:.2f} ms ± {std_inference_time:.2f} ms")
    
    # Per-class accuracy
    print("\nPer-class Accuracies:")
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_acc = np.sum(y_pred[class_mask] == i) / np.sum(class_mask)
        print(f"{class_name}: {class_acc:.4f}")

def plot_comparison_metrics(teacher_metrics, student_metrics, quantized_metrics, class_names):
    """Plot comparison of accuracies between models."""
    metrics_data = {
        'Teacher': teacher_metrics,
        'Student': student_metrics,
        'Quantized Student': quantized_metrics
    }
    
    # Prepare data for plotting
    models = list(metrics_data.keys())
    accuracies = [m['accuracy'] for m in metrics_data.values()]
    inf_times = [m['inference_time'] for m in metrics_data.values()]
    
    # Plot accuracies
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Plot inference times
    plt.subplot(1, 2, 2)
    plt.bar(models, inf_times)
    plt.title('Average Inference Time Comparison')
    plt.ylabel('Time (ms)')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load test data
    test_ds, class_names = load_test_data(batch_size=16)
    
    # 1. Evaluate Teacher Model
    teacher_model = tf.keras.models.load_model("teacher_model.h5")
    y_true_teacher, y_pred_teacher, teacher_times = evaluate_keras_model(
        teacher_model, test_ds, "Teacher Model"
    )
    calculate_metrics(y_true_teacher, y_pred_teacher, class_names, "Teacher Model", teacher_times)
    plot_confusion_matrix(
        y_true_teacher, y_pred_teacher, class_names,
        "Confusion Matrix (Teacher Model)", "Blues"
    )
    teacher_size = os.path.getsize("teacher_model.h5") / 1024  # KB
    print(f"\nTeacher Model Size: {teacher_size:.2f} KB")
    
    # 2. Evaluate Student Model (Original)
    student_model = tf.keras.models.load_model("student_model.h5")
    y_true_student, y_pred_student, student_times = evaluate_keras_model(
        student_model, test_ds, "Student Model"
    )
    calculate_metrics(y_true_student, y_pred_student, class_names, "Student Model", student_times)
    plot_confusion_matrix(
        y_true_student, y_pred_student, class_names,
        "Confusion Matrix (Student Model)", "Greens"
    )
    student_size = os.path.getsize("student_model.h5") / 1024  # KB
    print(f"\nStudent Model Size: {student_size:.2f} KB")
    
    # 3. Evaluate Quantized Student Model
    interpreter = tf.lite.Interpreter(model_path="arduino_model_student_pruned.tflite")
    y_true_quant, y_pred_quant, quant_times = evaluate_tflite_model(
        interpreter, test_ds, "Quantized Student Model"
    )
    calculate_metrics(y_true_quant, y_pred_quant, class_names, "Quantized Student Model", quant_times)
    plot_confusion_matrix(
        y_true_quant, y_pred_quant, class_names,
        "Confusion Matrix (Quantized Student Model)", "Oranges"
    )
    quantized_size = os.path.getsize("arduino_model_student_pruned.tflite") / 1024  # KB
    print(f"\nQuantized Student Model Size: {quantized_size:.2f} KB")
    
    # 4. Plot Comparison Metrics
    metrics = {
        'teacher': {
            'accuracy': np.mean(y_true_teacher == y_pred_teacher),
            'inference_time': np.mean(teacher_times)
        },
        'student': {
            'accuracy': np.mean(y_true_student == y_pred_student),
            'inference_time': np.mean(student_times)
        },
        'quantized': {
            'accuracy': np.mean(y_true_quant == y_pred_quant),
            'inference_time': np.mean(quant_times)
        }
    }
    
    plot_comparison_metrics(
        metrics['teacher'],
        metrics['student'],
        metrics['quantized'],
        class_names
    )
    
    # Print compression ratios
    print("\nModel Size Comparisons:")
    print(f"Teacher vs Student Compression Ratio: {teacher_size / student_size:.2f}x")
    print(f"Teacher vs Quantized Student Compression Ratio: {teacher_size / quantized_size:.2f}x")
    print(f"Student vs Quantized Student Compression Ratio: {student_size / quantized_size:.2f}x")

if __name__ == "__main__":
    main()