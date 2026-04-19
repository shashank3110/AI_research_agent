
# evaluate_ssl.py
import torch
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

# Import necessary components from the provided codebase
# Assuming the other files are in the same directory
from ssl_self_distillation import SelfDistillationModel, create_masked_data
from ssl_contrastive import ContrastiveTimeSeriesModel, time_series_augmentations, contrastive_loss
from train_ssl import DummyTimeSeriesDataset, set_seed # Re-using dataset and seed function

# --- Configuration ---
# Model parameters (must match training script)
INPUT_DIM = 1
HIDDEN_DIM = 128
LATENT_DIM = 64
SEQ_LEN = 200
NUM_SAMPLES = 1000
BATCH_SIZE = 64
TEMPERATURE = 0.1
MASK_RATIO = 0.15
MOMENTUM = 0.999
SEED = 42

# Evaluation parameters
TEST_SIZE = 0.2
ANOMALY_RATIO = 0.1 # Ratio of anomalies to introduce in the test set

# --- Helper Functions ---
def extract_features(model, data_loader, model_type='sd', device='cpu'):
    """Extracts features from the given data using the specified SSL model."""
    model.eval()
    all_features = []
    all_labels = [] # To keep track if it's normal or anomalous (for evaluation)

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            batch_size = batch_data.shape[0]

            if model_type == 'sd':
                # For Self-Distillation, use the student encoder
                # Ensure model is in eval mode and teacher is frozen
                sd_model_instance = model # Assuming model is SelfDistillationModel
                student_encoder = sd_model_instance.student_encoder
                # Create a non-masked version for feature extraction
                features = student_encoder(batch_data) # Shape: (batch_size, seq_len, latent_dim)
                # We'll average features across time steps for simplicity in downstream task
                features_avg = torch.mean(features, dim=1) # Shape: (batch_size, latent_dim)
                all_features.append(features_avg.cpu().numpy())
                all_labels.append(np.zeros(batch_size)) # Assume normal for now

            elif model_type == 'cl':
                # For Contrastive Learning, use the encoder
                cl_model_instance = model # Assuming model is ContrastiveTimeSeriesModel
                # Apply one augmentation or no augmentation for consistent feature extraction
                # Using no augmentation here for simplicity
                view1 = batch_data # No augmentation
                z1, _ = cl_model_instance(view1, view1) # Pass same view twice
                features = z1 # Shape: (batch_size, seq_len, latent_dim)
                features_avg = torch.mean(features, dim=1) # Shape: (batch_size, latent_dim)
                all_features.append(features_avg.cpu().numpy())
                all_labels.append(np.zeros(batch_size)) # Assume normal for now
            else:
                raise ValueError("Unknown model_type. Choose 'sd' or 'cl'.")

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

def introduce_anomalies(data, labels, anomaly_ratio, input_dim, seq_len):
    """Introduces synthetic anomalies into the data."""
    num_samples = len(data)
    num_anomalies = int(num_samples * anomaly_ratio)
    
    # Ensure we only select from indices that are currently labeled as normal (0)
    normal_indices = np.where(labels == 0)[0]
    if len(normal_indices) < num_anomalies:
        print(f"Warning: Not enough normal samples ({len(normal_indices)}) to introduce {num_anomalies} anomalies. Using all available normal samples.")
        num_anomalies = len(normal_indices)
        
    anomaly_indices = np.random.choice(normal_indices, num_anomalies, replace=False)

    anomalous_data = data.copy()
    anomalous_labels = labels.copy()

    for idx in anomaly_indices:
        # Introduce a spike or a significant deviation
        spike_magnitude = np.random.uniform(3, 6)
        spike_pos = np.random.randint(0, seq_len)
        # Ensure the data is float before adding spike
        anomalous_data[idx, spike_pos, 0] = float(anomalous_data[idx, spike_pos, 0]) + spike_magnitude
        anomalous_labels[idx] = 1 # Mark as anomaly

    return anomalous_data, anomalous_labels

# --- Main Evaluation Script ---
if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Dataset ---
    # Use a subset for training the anomaly detector and a separate set for testing
    full_dataset = DummyTimeSeriesDataset(NUM_SAMPLES, SEQ_LEN, INPUT_DIM)
    train_data_indices, test_data_indices = train_test_split(
        np.arange(NUM_SAMPLES), test_size=TEST_SIZE, random_state=SEED
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_data_indices)
    test_dataset_orig = torch.utils.data.Subset(full_dataset, test_data_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # No need to shuffle for feature extraction
    test_loader = DataLoader(test_dataset_orig, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Initialize Models ---
    # Self-Distillation Model
    sd_model = SelfDistillationModel(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    # In a real scenario, load trained weights:
    # sd_model.load_state_dict(torch.load("self_distillation_model.pth"))

    # Contrastive Learning Model
    cl_model = ContrastiveTimeSeriesModel(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, temperature=TEMPERATURE).to(device)
    # In a real scenario, load trained weights:
    # cl_model.load_state_dict(torch.load("contrastive_model.pth"))

    print("Models initialized (using random weights for demonstration).")

    # --- Feature Extraction ---
    print("Extracting features for Self-Distillation model...")
    sd_train_features, sd_train_labels = extract_features(sd_model, train_loader, model_type='sd', device=device)
    sd_test_features, sd_test_labels = extract_features(sd_model, test_loader, model_type='sd', device=device)
    print(f"SD Train Features shape: {sd_train_features.shape}")
    print(f"SD Test Features shape: {sd_test_features.shape}")

    print("\nExtracting features for Contrastive Learning model...")
    cl_train_features, cl_train_labels = extract_features(cl_model, train_loader, model_type='cl', device=device)
    cl_test_features, cl_test_labels = extract_features(cl_model, test_loader, model_type='cl', device=device)
    print(f"CL Train Features shape: {cl_train_features.shape}")
    print(f"CL Test Features shape: {cl_test_features.shape}")

    # --- Prepare Data for Anomaly Detection ---
    # Introduce anomalies into the test set features
    # Note: We are introducing anomalies *after* feature extraction for simplicity.
    # A more rigorous approach would be to introduce anomalies in the raw data
    # and then extract features. However, this demonstrates the concept.
    
    # Ensure features are numpy arrays before introducing anomalies
    sd_test_features_np = sd_test_features.astype(np.float64)
    sd_test_labels_np = sd_test_labels.astype(np.int64)
    cl_test_features_np = cl_test_features.astype(np.float64)
    cl_test_labels_np = cl_test_labels.astype(np.int64)

    sd_test_anomalous_data, sd_test_anomalous_labels = introduce_anomalies(
        sd_test_features_np, sd_test_labels_np, ANOMALY_RATIO, sd_test_features.shape[1], 1 # input_dim for features is latent_dim, seq_len is 1 after averaging
    )
    cl_test_anomalous_data, cl_test_anomalous_labels = introduce_anomalies(
        cl_test_features_np, cl_test_labels_np, ANOMALY_RATIO, cl_test_features.shape[1], 1
    )

    # Combine normal test features with anomalous ones
    sd_final_test_features = np.vstack((sd_test_features[sd_test_labels == 0], sd_test_anomalous_data))
    sd_final_test_labels = np.concatenate((sd_test_labels[sd_test_labels == 0], sd_test_anomalous_labels))

    cl_final_test_features = np.vstack((cl_test_features[cl_test_labels == 0], cl_test_anomalous_data))
    cl_final_test_labels = np.concatenate((cl_test_labels[cl_test_labels == 0], cl_test_anomalous_labels))

    # Shuffle the final test set
    shuffle_idx_sd = np.random.permutation(len(sd_final_test_features))
    sd_final_test_features = sd_final_test_features[shuffle_idx_sd]
    sd_final_test_labels = sd_final_test_labels[shuffle_idx_sd]

    shuffle_idx_cl = np.random.permutation(len(cl_final_test_features))
    cl_final_test_features = cl_final_test_features[shuffle_idx_cl]
    cl_final_test_labels = cl_final_test_labels[shuffle_idx_cl]

    print(f"\nSD Test set with anomalies: {np.sum(sd_final_test_labels)} samples, {np.sum(sd_final_test_labels == 1)} anomalies.")
    print(f"CL Test set with anomalies: {np.sum(cl_final_test_labels)} samples, {np.sum(cl_final_test_labels == 1)} anomalies.")


    # --- Train Anomaly Detector (Isolation Forest) ---
    print("\nTraining Isolation Forest on SD features...")
    iso_forest_sd = IsolationForest(contamination='auto', random_state=SEED)
    iso_forest_sd.fit(sd_train_features) # Train on normal training data features

    print("Training Isolation Forest on CL features...")
    iso_forest_cl = IsolationForest(contamination='auto', random_state=SEED)
    iso_forest_cl.fit(cl_train_features) # Train on normal training data features

    # --- Evaluate Anomaly Detectors ---
    print("\nEvaluating SD model performance...")
    sd_predictions = iso_forest_sd.predict(sd_final_test_features)
    sd_pred_labels = np.where(sd_predictions == -1, 1, 0) # Convert -1 (anomaly) to 1, 1 (normal) to 0

    sd_precision, sd_recall, sd_f1, _ = precision_recall_fscore_support(sd_final_test_labels, sd_pred_labels, average='binary')
    print(f"SD Model - Precision: {sd_precision:.4f}, Recall: {sd_recall:.4f}, F1 Score: {sd_f1:.4f}")

    print("\nEvaluating CL model performance...")
    cl_predictions = iso_forest_cl.predict(cl_final_test_features)
    cl_pred_labels = np.where(cl_predictions == -1, 1, 0) # Convert -1 (anomaly) to 1, 1 (normal) to 0

    cl_precision, cl_recall, cl_f1, _ = precision_recall_fscore_support(cl_final_test_labels, cl_pred_labels, average='binary')
    print(f"CL Model - Precision: {cl_precision:.4f}, Recall: {cl_recall:.4f}, F1 Score: {cl_f1:.4f}")

    print("\nEvaluation complete.")

