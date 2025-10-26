import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time, psutil, os
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# =======================
# Resource Monitoring
# =======================
def measure_resources(start_time, label=""):
    elapsed = time.time() - start_time
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"\n=== {label} Resource Report ===")
    print(f"Elapsed Time: {elapsed:.2f} sec")
    print(f"Memory Usage: {mem:.2f} MB")
    return elapsed, mem


# =======================
# 1. Load Data
# =======================
print("Loading Dataset...\n")
file_name = "5G_Dataset_Network_Slicing_CRAWDAD_Shared.xlsx"
df = pd.read_excel(file_name, sheet_name="Model_Inputs_Outputs")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("Finished Loading\n")

features_all = [
    'LTE/5G UE Category (Input 2)',
    'Technology Supported (Input 3)',
    'Day (Input4)',
    'Time (Input 5)',
    'QCI (Input 6)',
    'Packet Loss Rate (Reliability)',
    'Packet Delay Budget (Latency)',
    'Slice Type (Output)'
]


features_filtered = [
    'Technology Supported (Input 3)',
    'Packet Loss Rate (Reliability)',
    'Packet Delay Budget (Latency)',
]


# switch between feature sets
features = features_filtered

target = 'Use CaseType (Input 1)'

# =======================
# 2. Data Cleaning
# =======================
# Clean latency values

df['Packet Delay Budget (Latency)'] = (
    df['Packet Delay Budget (Latency)']
    .astype(str)
    .str.replace('<', '', regex=False)
    .str.replace('ms', '', regex=False)
    .astype(float)
)


group_map = {
    'AR/VR/Gaming': 'Consumer',
    'Smartphone': 'Consumer',
    'Industry 4.0': 'Critical IoT', # Critical
    'IoT Devices': 'User-related IoT',
    'Smart City & Home': 'User-related IoT',
    'Healthcare': 'Critical IoT',
    'Public Safety/E911': 'Critical IoT',
    'Smart Transportation': 'Critical IoT'
}


df['Use Case Group'] = df[target].map(group_map)

# Encode target
group_encoder = LabelEncoder()
y_encoded = group_encoder.fit_transform(df['Use Case Group'])

# One-hot encode features
X = pd.get_dummies(df[features], drop_first=True)



# 4. Train-Test Split
# Capture indices from the split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Extract corresponding rows from df for test set
df_test = df.loc[X_test.index].copy()


# 5. Random Forest Training
print("Start Training...\n")
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)

measure_resources(start, "Random Forest")
print("Training Finished\n")


# 6. Evaluation
print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=group_encoder.classes_))

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d',
            xticklabels=group_encoder.classes_,
            yticklabels=group_encoder.classes_,
            cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#print(df.groupby('Technology Supported (Input 3)')['Use Case Group'].value_counts(normalize=True))
#print(df.groupby('Slice Type (Output)')['Use Case Group'].value_counts(normalize=True))

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt


# --- Shuffle sanity test (fixed version) ---
print("5. Shuffle sanity test (should drop accuracy to ~0.33 for 3 classes):")

# Shuffle only the training labels (same size as X_train)
y_shuffled = np.random.permutation(y_train)

rf_model.fit(X_train, y_shuffled)
acc_shuffled = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Accuracy after shuffling target labels: {acc_shuffled:.3f}")

if acc_shuffled > 0.5:
    print("Possible data leakage or deterministic mapping detected.")
else:
    print("Model accuracy drops after shuffling → learning is likely genuine.\n")



# --- Cross-validation sanity check ---
print("6. 5-Fold Cross-Validation Accuracy:")
from sklearn.ensemble import RandomForestClassifier
rf_temp = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
scores = cross_val_score(rf_temp, X, y_encoded, cv=5)
print(f"Cross-val accuracy (mean ± std): {scores.mean():.3f} ± {scores.std():.3f}")

if scores.mean() < 0.9 * accuracy_score(y_test, y_pred_rf):
    print("Cross-validation accuracy significantly lower than test accuracy → potential overfitting or leakage.")
else:
    print("Cross-validation accuracy consistent with test results.\n")

print("\n=== End of Diagnostics ===")



# 7. Clean Feature Importance Plot (aggregated)

# Get feature importances from the model
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Create DataFrame with raw feature names and importance
feat_imp_df = pd.DataFrame({
    'Feature': X.columns[indices],
    'Importance': importances[indices]
})

# Clean up names: remove text in parentheses like (Input 3)
feat_imp_df['Feature'] = feat_imp_df['Feature'].str.replace(r'\(.*?\)', '', regex=True).str.strip()

# --- Group related dummy features ---
feat_imp_df['BaseFeature'] = (
    feat_imp_df['Feature']
    .str.replace(r'_.+', '', regex=True)  # Remove suffixes like _mMTC, _eMBB, etc.
    .str.strip()
)

# Aggregate importances by base feature
feat_imp_grouped = (
    feat_imp_df.groupby('BaseFeature', as_index=False)['Importance'].sum()
    .sort_values('Importance', ascending=False)
)

# Show only top N aggregated features
TOP_N = 8
feat_imp_grouped = feat_imp_grouped.head(TOP_N)

# --- Plot aggregated feature importance ---
plt.figure(figsize=(9, 6))
sns.barplot(
    data=feat_imp_grouped,
    y='BaseFeature',
    x='Importance',
    palette='viridis'
)
plt.title("Top Feature Importances – Random Forest", fontsize=14, weight='bold')
plt.xlabel("Relative Importance", fontsize=12)
plt.ylabel("")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

# Annotate bar values
for i, v in enumerate(feat_imp_grouped["Importance"]):
    plt.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=9)

plt.show()



