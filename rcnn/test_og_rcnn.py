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
    # Load test data and model
    test_ds, class_names = load_test_data(batch_size=16)
    model = tf.keras.models.load_model("rcnn_model.h5")
    
    y_true, y_pred = [], []
    
    # Evaluate on the entire test dataset
    print("\nEvaluating original RCNN model...")
    for features, labels in test_ds:
        predictions = model.predict(features)
        preds = np.argmax(predictions, axis=1)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Display metrics
    print("\nClassification Report (Original RCNN):")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create and display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Original RCNN)")
    plt.tight_layout()
    plt.show()
    
    # Calculate and display additional metrics
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_acc = np.sum(y_pred[class_mask] == i) / np.sum(class_mask)
        class_accuracies[class_name] = class_acc
    
    print("\nPer-class Accuracies:")
    for class_name, acc in class_accuracies.items():
        print(f"{class_name}: {acc:.4f}")

if __name__ == "__main__":
    main()