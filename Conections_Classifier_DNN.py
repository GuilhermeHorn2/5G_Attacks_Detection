
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time, psutil, os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping



# Resource Monitoring
def measure_resources(start_time, label=""):
    elapsed = time.time() - start_time
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"\n=== {label} Resource Report ===")
    print(f"Elapsed Time: {elapsed:.2f} sec")
    print(f"Memory Usage: {mem:.2f} MB")
    return elapsed, mem

# 1. Load Data
print("Loading Dataset...\n")
file_name = "5G_Dataset_Network_Slicing_CRAWDAD_Shared.xlsx"  # relative path
df = pd.read_excel(file_name, sheet_name="Model_Inputs_Outputs")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("Finished Loading\n")

features = [
    'LTE/5G UE Category (Input 2)',
    'Technology Supported (Input 3)',
    'Day (Input4)',
    'Time (Input 5)',
    'QCI (Input 6)',
    'Packet Loss Rate (Reliability)',
    'Packet Delay Budget (Latency)',
    'Slice Type (Output)'
]

target = 'Use CaseType (Input 1)'

# Clean latency
df['Packet Delay Budget (Latency)'] = (
    df['Packet Delay Budget (Latency)']
    .astype(str)
    .str.replace('<', '', regex=False)
    .str.replace('ms', '', regex=False)
    .astype(float)
)

# Group mapping
group_map = {
    'AR/VR/Gaming': 'Consumer',
    'Smartphone': 'Consumer',
    'Industry 4.0': 'Industrial',
    'IoT Devices': 'Industrial',
    'Smart City & Home': 'Industrial',
    'Healthcare': 'Critical',
    'Public Safety/E911': 'Critical',
    'Smart Transportation': 'Critical'
}
df['Use Case Group'] = df[target].map(group_map)

# Encode target
group_encoder = LabelEncoder()
y_encoded = group_encoder.fit_transform(df['Use Case Group'])

# One-hot encode features
X = pd.get_dummies(df[features], drop_first=True)



# 3. Deep Neural Network (DNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_cat = to_categorical(y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.3, random_state=42)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start = time.time()
print("\n=== Training DNN ===")
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=1)
measure_resources(start, "DNN")
process = psutil.Process(os.getpid())
cpu_times = process.cpu_times()
print(f"User CPU time: {cpu_times.user:.2f} sec")
print(f"System CPU time: {cpu_times.system:.2f} sec")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n=== DNN Results ===")
print(classification_report(y_true_classes, y_pred_classes, target_names=group_encoder.classes_))

cm_dnn = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm_dnn, annot=True, fmt='d',
            xticklabels=group_encoder.classes_,
            yticklabels=group_encoder.classes_,
            cmap="Oranges")
plt.title("DNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
