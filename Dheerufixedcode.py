# ================================================================================
# PROJECT: MACHINE LEARNING BASED NETWORK INTRUSION DETECTION SYSTEM (IDS)
# BRANCH: ELECTRONICS AND COMMUNICATION ENGINEERING
# ================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ------------------------------------------------------------------------------
# Note: You can download the standard 'KDDTrain+.txt' dataset from the NSL-KDD archive.
# We explicitly define the core ECE networking feature columns below.

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_score'
]

print("[INFO] Loading NSL-KDD network traffic data...")
# Replace 'KDDTrain+.txt' with your local dataset path when running
try:
    df = pd.read_csv('KDDTrain+.txt', names=columns, header=None)
    df.drop('difficulty_score', axis=1, inplace=True) # Drop unnecessary score column
except FileNotFoundError:
    print("[ERROR] 'KDDTrain+.txt' file not found. Generating simulated data for demonstration...")
    # Creating simulated fallback data so the code executes perfectly anywhere
    np.random.seed(42)
    fake_data = []
    for _ in range(1000):
        fake_data.append([
            np.random.randint(0, 10), np.random.choice(['tcp', 'udp', 'icmp']),
            np.random.choice(['http', 'smtp', 'ftp_data', 'private']), np.random.choice(['SF', 'S0', 'REJ']),
            np.random.randint(100, 5000), np.random.randint(0, 5000),
            0, 0, 0, 0, 0, np.random.choice([0, 1]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            np.random.randint(1, 100), np.random.randint(1, 100), 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 255, np.random.randint(1, 255), 1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
            np.random.choice(['normal', 'anomaly'])
        ])
    df = pd.DataFrame(fake_data, columns=columns[:-1])

# ------------------------------------------------------------------------------
# STEP 2: DATA PREPROCESSING & ENCODING (Converting Protocols to Numbers)
# ------------------------------------------------------------------------------
print("[INFO] Preprocessing network protocols and flag features...")

# Transform nominal features (TCP/UDP, Services, Status Flags) into numerical values
le_protocol = LabelEncoder()
df['protocol_type'] = le_protocol.fit_transform(df['protocol_type'])

le_service = LabelEncoder()
df['service'] = le_service.fit_transform(df['service'])

le_flag = LabelEncoder()
df['flag'] = le_flag.fit_transform(df['flag'])

# Binary target classification: 'normal' data packets vs 'anomaly' (malicious attacks)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Split features (X) and ground truth targets (y)
X = df.drop('label', axis=1)
y = df['label']

# ------------------------------------------------------------------------------
# STEP 3: TRAIN-TEST SPLIT (80% Training Data, 20% Evaluation Data)
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"[INFO] Training sample size: {X_train.shape[0]} packets")
print(f"[INFO] Testing sample size: {X_test.shape[0]} packets")

# ------------------------------------------------------------------------------
# STEP 4: MODEL INITIALIZATION & TRAINING (Random Forest)
# ------------------------------------------------------------------------------
print("[INFO] Initializing Random Forest Classifier with ensemble decision trees...")
ids_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

print("[INFO] Model Training in progress... Mapping communication packet behaviors...")
ids_classifier.fit(X_train, y_train)
print("[SUCCESS] Model training complete.")

# ------------------------------------------------------------------------------
# STEP 5: PREDICTION & EVALUATION
# ------------------------------------------------------------------------------
print("[INFO] Testing model on unseen evaluation data traffic...")
y_pred = ids_classifier.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print(f"             FINAL METRICS REPORT             ")
print("="*50)
print(f"Overall System Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nDetailed Classification Performance Report:")
print(classification_report(y_test, y_pred, target_names=['Normal Traffic', 'Intrusion Alert']))
print("="*50)

# ------------------------------------------------------------------------------
# STEP 6: FEATURE IMPORTANCE PLOT (Showing the math for your presentation)
# ------------------------------------------------------------------------------
importances = ids_classifier.feature_importances_
indices = np.argsort(importances)[-5:] # Top 5 most critical network metrics

plt.figure(figsize=(10, 5))
plt.title("ECE Analysis: Top 5 Packet Metrics Determining An Intrusion")
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Relative Statistical Importance Weight")
plt.tight_layout()
print("[INFO] Displaying optimization graph. Save this for your final viva review.")
plt.show()