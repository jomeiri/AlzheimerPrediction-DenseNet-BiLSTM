import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test, num_classes):
    """
    Evaluates the trained deep learning model.
    Implements Algorithm 7 from the thesis. [cite: 1014]

    Args:
        model (tf.keras.Model): The trained Keras model.
        X_test (np.array): Testing data (longitudinal MRI sequences).
                           Shape: (num_samples, time_points, H, W, D, Channels)
        y_test (np.array): True testing labels (integer-encoded).
        num_classes (int): Number of distinct classes (e.g., 2 for AD/CN, 3 for NC/MCI/AD).

    Returns:
        tuple: (accuracy, sensitivity, specificity, auc_score)
    """
    print("Making predictions on the test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # 1. Accuracy [cite: 1168]
    accuracy = accuracy_score(y_test, y_pred_classes)

    # 2. Sensitivity (Recall) - for multi-class, typically calculated per-class or macro/weighted average
    # Thesis does not specify multi-class sensitivity calculation. For binary, it's straightforward.
    # For multi-class, we'll use 'macro' average as a common practice.
    # If num_classes > 2, you might need to specify `average` (e.g., 'macro', 'weighted', 'binary')
    # For binary classification (AD/CN), it simplifies. Assume `pos_label=1` for 'AD' if 0=CN, 1=AD
    if num_classes == 2:
        sensitivity = recall_score(y_test, y_pred_classes, average='binary', pos_label=1) # Assuming 1 is the positive class (e.g., AD)
        specificity = recall_score(y_test, y_pred_classes, average='binary', pos_label=0) # Assuming 0 is the negative class (e.g., CN)
    else: # Multi-class scenario (e.g., NC/MCI/AD)
        # For multi-class, sensitivity and specificity calculation is more nuanced.
        # Often, macro or weighted averages are used, or calculated for each class individually.
        # A common way to get "overall" sensitivity/specificity in multi-class is via confusion matrix.
        # Simplified: Use `recall_score` with `average='macro'` for sensitivity.
        # Specificity for multi-class: (TN / (TN + FP)) for each class
        sensitivity = recall_score(y_test, y_pred_classes, average='macro')
        # Specificity is harder to get directly for multi-class with sklearn.
        # You'd need a confusion matrix:
        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred_classes)
        # TN = sum(np.delete(np.delete(cm, i, axis=0), i, axis=1)).sum() for class i
        # FP = cm[:, i].sum() - cm[i, i]
        # specificity = TN / (TN + FP)
        # For simplicity in this general script, we'll return None for multi-class specificity or implement per-class.
        specificity = None # Placeholder - implement per-class if needed

    # 3. AUC (Area Under the Curve) [cite: 1180]
    # For multi-class, AUC needs to be calculated using one-vs-rest (OvR) approach.
    if num_classes > 2:
        y_test_binarized = label_binarize(y_test, classes=range(num_classes))
        auc_score = 0.0
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
            auc_score += auc(fpr, tpr)
        auc_score /= num_classes # Average AUC for multi-class
    else: # Binary classification
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1]) # Assuming class 1 is positive
        auc_score = auc(fpr, tpr)

    return accuracy, sensitivity, specificity, auc_score
