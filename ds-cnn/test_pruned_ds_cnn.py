# test_ds_cnn_metrics_quantized.py
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
    test_ds, class_names = load_test_data(batch_size=16)
    
    # Load quantized TFLite model
    interpreter = tf.lite.Interpreter(model_path="arduino_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    y_true, y_pred = [], []
    
    # Evaluate using the test dataset
    for features, labels in test_ds:
        # The quantized model expects int8 input
        input_batch = features.numpy().astype(np.int8)
        batch_size = input_batch.shape[0]
        for i in range(batch_size):
            sample = np.expand_dims(input_batch[i], axis=0)
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output)
            y_pred.append(pred)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Display metrics
    print("Classification Report (Quantized DS-CNN):")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Quantized DS-CNN)")
    plt.show()

if __name__ == "__main__":
    main()
