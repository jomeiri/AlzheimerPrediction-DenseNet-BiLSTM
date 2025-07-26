import os
from src.preprocess_data import load_and_preprocess_adni_data
from src.model import build_densenet_bilstm_model
from src.train import train_model
from src.evaluate import evaluate_model
from sklearn.model_selection import StratifiedKFold
import numpy as np

# --- Configuration ---
DATA_DIR = 'data/ADNI_MRI_raw' # Path to your raw ADNI NIfTI files
PREPROCESSED_DIR = 'data/ADNI_preprocessed' # Output for preprocessed data
MODEL_SAVE_PATH = 'trained_model.h5'
NUM_CLASSES = 2 # AD/CN (adjust for other classifications like MCI/AD, etc.)
EPOCHS = 90 # As per thesis [cite: 1151]
BATCH_SIZE = 64 # As per thesis [cite: 1151]
LEARNING_RATE = 0.001 # As per thesis justification
DROPOUT_RATE = 0.10 # As per thesis [cite: 986, 1153]
L2_REG_LAMBDA = 0.01 # As per thesis [cite: 986, 1153]
K_FOLDS = 10 # For stratified 10-fold cross-validation [cite: 1097]
TIME_POINTS = 7 # Number of time points (e.g., M0, M6, ..., M48) [cite: 1118]
# Assuming a fixed input shape after preprocessing for DenseNet
INPUT_SHAPE_3D = (200, 168, 168, 1) # Example shape after removing zero-density pixels [cite: 1135]
# DenseNet-121 parameters (from Table 2.1 in thesis, simplified for illustration)
DENSENET_CONFIG = {
    'depth': 121,
    'growth_rate': 16, # From thesis Table 2.1 filter size 16
    'num_dense_blocks': 4, # As per thesis Figure 2.4
    'input_shape': INPUT_SHAPE_3D
}
# BiLSTM parameters
BILSTM_UNITS = 256 # Assuming this as hidden size based on common practice, adjust as needed

def main():
    print("Starting Alzheimer's Disease Prediction using Hybrid DenseNet-BiLSTM.")

    # 1. Load and Preprocess Data (as per Algorithm 2 in thesis)
    # This function needs to handle NIfTI files, AC-PC correction, skull stripping,
    # cerebellum removal, N3 correction, resampling to 256x256x256 (or 200x168x168 after zero-density removal),
    # tissue segmentation (GM, WM, CSF), spatial normalization (HAMMER),
    # and extraction of GM volumetric maps.
    # It must return longitudinal sequences for each patient.
    print(f"Loading and preprocessing data from {DATA_DIR}...")
    # This will be a complex step, requiring external libraries like SimpleITK, FSL, etc.
    # For a placeholder, let's assume `all_mri_sequences` is a list of tuples:
    # `[(patient_id, [mri_scan_t0, mri_scan_t6, ..., mri_scan_t48], label), ...]`
    # where each mri_scan is a numpy array (preprocessed GM volume)
    # and labels are 0 (NC), 1 (MCI), 2 (AD) as per ADNI classification.
    preprocessed_data, labels = load_and_preprocess_adni_data(DATA_DIR, PREPROCESSED_DIR, target_shape=INPUT_SHAPE_3D[:-1])
    print(f"Loaded {len(preprocessed_data)} longitudinal MRI sequences.")

    # Convert labels to numerical format if not already (e.g., AD: 0, CN: 1, MCI: 2)
    # Ensure `labels` is a numpy array of integers for stratified split
    # Example: If your labels are strings 'AD', 'CN', map them to 0, 1
    unique_labels = np.unique(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        label_map = {label: i for i, label in enumerate(unique_labels)}
        numerical_labels = np.array([label_map[l] for l in labels])
    else:
        numerical_labels = labels

    # Stratified K-Fold Cross-Validation [cite: 1097]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_sensitivities = []
    fold_specificities = []
    fold_aucs = []

    # Convert preprocessed_data to a numpy array for consistent indexing if it's a list
    # Ensure it's in a format suitable for direct indexing by skf.split
    # e.g., `np.array(preprocessed_data, dtype=object)` if elements are variable length sequences
    # Or, if `preprocessed_data` is just paths, handle loading within the loop
    X = np.array(preprocessed_data, dtype=object) # Assuming preprocessed_data is a list of sequences
    y = numerical_labels

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Training for Fold {fold + 1}/{K_FOLDS} ---")
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # 2. Build the Hybrid DenseNet-BiLSTM Model [cite: 820]
        # The model needs to be re-initialized for each fold to ensure independent training.
        model = build_densenet_bilstm_model(
            densenet_config=DENSENET_CONFIG,
            bilstm_units=BILSTM_UNITS,
            num_classes=NUM_CLASSES,
            time_points=TIME_POINTS,
            dropout_rate=DROPOUT_RATE,
            l2_reg_lambda=L2_REG_LAMBDA
        )
        print("Model built successfully.")
        model.summary() # Print model architecture

        # 3. Train the Model (as per Algorithm 6 in thesis)
        print("Training model...")
        history = train_model(
            model,
            X_train_fold,
            y_train_fold,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        print("Model training complete.")

        # 4. Evaluate the Model (as per Algorithm 7 in thesis)
        print("Evaluating model...")
        accuracy, sensitivity, specificity, auc_score = evaluate_model(
            model,
            X_test_fold,
            y_test_fold,
            num_classes=NUM_CLASSES
        )
        print(f"Fold {fold + 1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  AUC: {auc_score:.4f}")

        fold_accuracies.append(accuracy)
        fold_sensitivities.append(sensitivity)
        fold_specificities.append(specificity)
        fold_aucs.append(auc_score)

    print("\n--- Cross-Validation Summary ---")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Average Sensitivity: {np.mean(fold_sensitivities):.4f} ± {np.std(fold_sensitivities):.4f}")
    print(f"Average Specificity: {np.mean(fold_specificities):.4f} ± {np.std(fold_specificities):.4f}")
    print(f"Average AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # Optionally save the last trained model (or the best performing model from folds)
    # model.save(MODEL_SAVE_PATH)
    # print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
