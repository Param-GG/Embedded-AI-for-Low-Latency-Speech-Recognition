import tensorflow as tf
import numpy as np
import sys, os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset preparation function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_handling import prepare_speech_commands_dataset

def load_test_data(batch_size=16):
    data_dir = 'datasets/speech_commands_v0_extracted'
    _, _, test_ds, class_names = prepare_speech_commands_dataset(data_dir, batch_size=batch_size)
    return test_ds, class_names

def main():
    # Load test data
    test_ds, class_names = load_test_data(batch_size=16)
    
    # Load quantized TFLite model
    interpreter = tf.lite.Interpreter(model_path="arduino_model_rcnn_pruned.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    y_true, y_pred = [], []
    total_samples = 0
    correct_predictions = 0
    
    print("\nEvaluating quantized RCNN model...")
    # Evaluate using the test dataset
    for features, labels in test_ds:
        # Convert float32 features to int8 using quantization parameters
        features_np = features.numpy()
        input_batch = np.round(features_np / input_scale + input_zero_point).astype(np.int8)
        
        batch_size_actual = input_batch.shape[0]
        for i in range(batch_size_actual):
            # Process one sample at a time
            sample = np.expand_dims(input_batch[i], axis=0)
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            
            # Get the output and dequantize it
            output_int8 = interpreter.get_tensor(output_details[0]['index'])
            output_float = (output_int8.astype(np.float32) - output_zero_point) * output_scale
            pred = np.argmax(output_float)
            y_pred.append(pred)
            
            # Track accuracy
            true_label = labels.numpy()[i]
            if pred == true_label:
                correct_predictions += 1
            total_samples += 1
            
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Display metrics
    print("\nClassification Report (Quantized RCNN):")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create and display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Quantized RCNN)")
    plt.tight_layout()
    plt.show()
    
    # Calculate and display additional metrics
    running_accuracy = correct_predictions / total_samples
    print(f"\nRunning Accuracy: {running_accuracy:.4f}")
    
    # Per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_acc = np.sum(y_pred[class_mask] == i) / np.sum(class_mask)
        class_accuracies[class_name] = class_acc
    
    print("\nPer-class Accuracies:")
    for class_name, acc in class_accuracies.items():
        print(f"{class_name}: {acc:.4f}")
    
    # Calculate memory usage
    interpreter_size = os.path.getsize("arduino_model_rcnn.tflite") / 1024  # KB
    print(f"\nModel Size: {interpreter_size:.2f} KB")

if __name__ == "__main__":
    main()