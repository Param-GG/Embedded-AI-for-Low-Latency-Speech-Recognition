# test_ds_cnn_metrics_original.py
import tensorflow as tf
import numpy as np
import sys, os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset preparation function from your data_preprocessing module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.dataset_handling import prepare_speech_commands_dataset

def load_test_data(batch_size=16):
    data_dir = 'datasets/speech_commands_v0_extracted'
    _, _, test_ds, class_names = prepare_speech_commands_dataset(data_dir, batch_size=batch_size)
    return test_ds, class_names

def main():
    test_ds, class_names = load_test_data(batch_size=16)
    model = tf.keras.models.load_model("ds_cnn_model.h5")
    
    y_true, y_pred = [], []
    # Evaluate on the entire test dataset
    for features, labels in test_ds:
        predictions = model.predict(features)
        preds = np.argmax(predictions, axis=1)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Display metrics
    print("Classification Report (Original DS-CNN):")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Original DS-CNN)")
    plt.show()

if __name__ == "__main__":
    main()
