import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# =======================
# Load dataset
# =======================
print("Loading Dataset...")
df_og = pd.read_excel("5G_Dataset_Network_Slicing_CRAWDAD_Shared.xlsx",
                      sheet_name="Model_Inputs_Outputs")
print("Loading Finished")

# Preprocessing
df = df_og.copy()
df['Packet Delay Budget (Latency)'] = (
    df['Packet Delay Budget (Latency)']
    .astype(str)
    .str.replace('<', '', regex=False)
    .str.replace('ms', '', regex=False)
    .astype(float)
)

#
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


# source ~/venv/bin/activate


df['Use Case Group'] = df['Use CaseType (Input 1)'].map(group_map)


# Add attack column if not already present
if "attack" not in df.columns:
    df["attack"] = 0  # all normal by default


###################### Injection Functions ########################################
df_attack = df
def inject_flood_rand(min, max):
    #
    global df_attack
    df = df_attack
    #print("len : ", len(df_attack))
    #print("\n")
    #print("Starting Random Injection")
    random_day = np.random.choice(df["Day (Input4)"].unique())
    random_time = np.random.choice(df["Time (Input 5)"].unique())
    #
    subset = df[(df["Day (Input4)"]==random_day) & (df["Time (Input 5)"]==random_time)]
    chosen_row = subset.sample(1, random_state=np.random.randint(1000))
    #
    #
    n = np.random.randint(min, max)
    #
    attack_rows = pd.concat([chosen_row]*n, ignore_index=True)
    attack_rows["attack"] = 1
    #
    df_attack = pd.concat([df, attack_rows], ignore_index=True)
    #
    df = df.sort_values(by=["Day (Input4)", "Time (Input 5)"]).reset_index(drop=True)
    #print("Random Injection Finished")
    return df 


def check_injection():
    attacks_only = df_attack[df_attack["attack"] == 1]
    print(attacks_only.head())
    print("Total injected rows:", len(attacks_only))

def group_rows():

    global df_attack
    df = df_attack
    # Columns to group by
 
    group_cols = [
        "Day (Input4)",
        "Time (Input 5)",
        "Slice Type (Output)",
        "Packet Delay Budget (Latency)",
        "Technology Supported (Input 3)",
        'Packet Loss Rate (Reliability)',
        "Use Case Group",
        "attack"
    ]

    # Perform aggregation
    df_agglutinated = df_attack.groupby(group_cols).agg(
        row_count=("attack", "count"),   # how many rows collapsed into this group
        avg_loss=("Packet Loss Rate (Reliability)", "mean")  # keep numeric features
    ).reset_index()

    print("Original dataset length:", len(df_attack))
    print("Agglutinated dataset length:", len(df_agglutinated))
    return df_agglutinated

def do_n_rand_injections(n, min, max):
    for i in range(n):
        inject_flood_rand(min, max)
############################################################################


# Make injections:
do_n_rand_injections(10, 20, 50)


# Group original dataset
df_agglutinated = group_rows()

# =======================
# Train/Test split
# =======================
FEATURES = ["row_count", "avg_loss"]
X_flood = df_agglutinated[FEATURES]
y_flood = df_agglutinated["attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X_flood, y_flood, test_size=0.3, random_state=42, stratify=y_flood
)

# =======================
# Train Flooding Model
# =======================
flooding_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
flooding_model.fit(X_train, y_train)

# =======================
# Evaluate
# =======================
y_pred = flooding_model.predict(X_test)

print("=== Flooding Detection Model (test set) ===")
print(classification_report(y_test, y_pred, target_names=["Normal","Attack"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Attack"],
            yticklabels=["Normal","Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Flooding Detector (Binary)")
plt.show()
