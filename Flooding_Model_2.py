import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ======================================
print("Loading Dataset...")
df_og = pd.read_excel("5G_Dataset_Network_Slicing_CRAWDAD_Shared.xlsx",
                   sheet_name="Model_Inputs_Outputs")
print("Loading Finished")
#
df = df_og
#
features = [
    'Technology Supported (Input 3)',
    'Packet Loss Rate (Reliability)',
    'Packet Delay Budget (Latency)',
    'Slice Type (Output)'
]
target = 'Use CaseType (Input 1)'
#
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




df['Use Case Group'] = df[target].map(group_map)
#

df_attack = df_og

df_attack["attack"] = 0

#####################################################################################
# Dividing the dataset to train both models
df_oracle, df_flooding = train_test_split(df, test_size=0.8, random_state=42, shuffle=True)

X_oracle = df_oracle[features]
y_oracle = df_oracle[target]

# Encode categorical variables automatically with pandas
X_oracle = pd.get_dummies(X_oracle)

oracle_model = RandomForestClassifier(random_state=42)
oracle_model.fit(X_oracle, y_oracle)

print("Oracle Trained")

####################################################################################

df_attack = df_flooding



def spoof_behavior(row):
    # Consumer -> Critical IoT .The attacker says the device is Critical IoT
    # but his behavior(with the expeciton of the slice) is of a smartphone
    
    spoofed = row.copy()
    #print("------>:",row["Use Case Group"])
    if "Consumer" in row["Use Case Group"]:
        spoofed["Use Case Group"] = "Critical IoT"
        spoofed["Slice Type (Output)"] = "URLLC"


    return spoofed


# helper
def _get_device_id_column(df):
    candidates = ['device_id', 'Device ID', 'Device_Id', 'UE ID', 'ue_id', 'UE', 'ue']
    for c in candidates:
        if c in df.columns:
            return c
    return None

def inject_flood_rand_multi(num_attackers=10, min_dup=10, max_dup=50, spoofed=True, seed=None):
    """
    Inject a multi-device flood at a random (Day, Time).
    - For `num_attackers` distinct attacker rows chosen from that window,
      produce requests for ALL slice types in the dataset.
    - Each (attacker, slice) pair will be duplicated k times with k uniform in [min_dup, max_dup).
    - If spoofed=True, apply spoof_behavior to each created row.
    - Updates global df_attack and returns it.
    """
    global df_attack
    if seed is not None:
        np.random.seed(seed)

    df = df_attack  # uses global dataset
    if len(df) == 0:
        raise ValueError("df_attack is empty")

    # choose a random day/time window that has at least num_attackers rows
    possible_days = df["Day (Input4)"].unique()
    chosen_day = np.random.choice(possible_days)
    possible_times = df[df["Day (Input4)"] == chosen_day]["Time (Input 5)"].unique()
    chosen_time = np.random.choice(possible_times)

    window_subset = df[(df["Day (Input4)"] == chosen_day) & (df["Time (Input 5)"] == chosen_time)]
    if window_subset.empty:
        raise RuntimeError("Chosen day/time window has no rows (unexpected)")

    # limit num_attackers to available rows
    num_attackers = min(num_attackers, len(window_subset))

    # pick distinct attacker prototype rows (sample without replacement)
    attackers = window_subset.sample(num_attackers, replace=False, random_state=np.random.randint(1_000_000))

    # slice types to request (use all distinct slice types in df)
    all_slices = df["Slice Type (Output)"].dropna().unique().tolist()
    if len(all_slices) == 0:
        raise RuntimeError("No slice types found in dataset")

    # Build attack rows list
    attack_rows_list = []
    for _, attacker_row in attackers.iterrows():
        # attacker_row is a Series; we will create new rows from it
        for sl in all_slices:
            dup_count = int(np.random.randint(min_dup, max_dup))
            # create dup_count copies of attacker_row
            block = pd.DataFrame([attacker_row.to_dict()] * dup_count)
            # set the slice type to current slice
            block["Slice Type (Output)"] = sl
            # mark attack flag
            block["attack"] = 1
            # optionally spoof behavior (apply function row-wise or vectorized)
            if spoofed:
                # if spoof_behavior expects a single Series, we can apply it per-row:
                block = block.apply(lambda r: spoof_behavior(r) if isinstance(spoof_behavior(r), pd.Series) else r, axis=1)
            attack_rows_list.append(block)

    # concatenate all created attack blocks
    if attack_rows_list:
        new_attack_rows = pd.concat(attack_rows_list, ignore_index=True)
    else:
        new_attack_rows = pd.DataFrame(columns=df.columns.tolist() + ["attack"])

    # append to global df_attack and sort
    df_attack = pd.concat([df, new_attack_rows], ignore_index=True)
    df_attack = df_attack.sort_values(by=["Day (Input4)", "Time (Input 5)"]).reset_index(drop=True)

    # info to return
    info = {
        "chosen_day": chosen_day,
        "chosen_time": chosen_time,
        "num_attackers": num_attackers,
        "slices_count": len(all_slices),
        "injected_rows": len(new_attack_rows)
    }
    print("Injected:", info)
    return df_attack, info





def check_injection():
    attacks_only = df_attack[df_attack["attack"] == 1]
    print(attacks_only.head())
    print("Total injected rows:", len(attacks_only))

def calc_avg_behavior():
    global df_attack
    df = df_attack
    num_cols = ['Packet Loss Rate (Reliability)', 'Packet Delay Budget (Latency)']
    cat_cols = ['Technology Supported (Input 3)', 'Slice Type (Output)']


    # Average of numerical features
    avg_numeric = df.groupby("Use Case Group")[num_cols].mean()

    # Most common (mode) of categorical features
    mode_categorical = df.groupby("Use Case Group")[cat_cols].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )

    print("\n=== Average numerical values per Group ===")
    print(avg_numeric)

    print("\n=== Most common categorical values per Group ===")
    print(mode_categorical)


def label_with_oracle(row, model):
    #print("Labeling...")
    x_row = pd.get_dummies(pd.DataFrame([row[features]]))
    x_row = x_row.reindex(columns=oracle_model.feature_names_in_, fill_value=0)
    predicted_group = model.predict(x_row)[0]
    if predicted_group in row["Use Case Group"]:
        return 0
    return 1



# ======================================

def categorical_entropy(series):
    """Compute Shannon entropy for a categorical pandas Series."""
    probs = series.value_counts(normalize=True).values
    if len(probs) <= 1:
        return 0.0
    return float(scipy_entropy(probs, base=2))

def tuple_key(row, cols):
    return tuple(row[c] for c in cols)

def aggregate_window_features(df,
                              group_cols=("Day (Input4)", "Time (Input 5)"),
                              tuple_cols=("Slice Type (Output)", "Technology Supported (Input 3)", "Use Case Group")):
    """
    Aggregate dataset by (Day, Time) to detect collective anomalies like flooding.
    Computes counts, diversity, repetition, entropy, and spoofing ratio.
    """
    df = df.copy()
    df["_tuple_key"] = df.apply(lambda r: tuple_key(r, tuple_cols), axis=1)

    out_rows = []
    for (day, time), sub in df.groupby(list(group_cols)):
        n = len(sub)
        unique_slices = sub["Slice Type (Output)"].nunique(dropna=True)
        unique_tech = sub["Technology Supported (Input 3)"].nunique(dropna=True)

        # top repeated identical connection
        tup_counts = Counter(sub["_tuple_key"])
        top_repeat = tup_counts.most_common(1)[0][1] if len(tup_counts) > 0 else 0

        # longest consecutive identical connection sequence
        max_run = 1
        curr_run = 1
        prev = None
        for v in sub["_tuple_key"].tolist():
            if v == prev:
                curr_run += 1
                max_run = max(max_run, curr_run)
            else:
                curr_run = 1
                prev = v

        # entropy measures
        slice_entropy = categorical_entropy(sub["Slice Type (Output)"].fillna("NA"))
        tech_entropy = categorical_entropy(sub["Technology Supported (Input 3)"].fillna("NA"))

        # spoofed ratio (if oracle predictions exist)
        spoofed_ratio = sub.get("spoofed", pd.Series(dtype=int)).mean() if "spoofed" in sub.columns else 0.0

        out_rows.append({
            "Day (Input4)": day,
            "Time (Input 5)": time,
            "row_count": n,
            "unique_slices": unique_slices,
            "unique_tech": unique_tech,
            "top_repeat": top_repeat,
            "max_run": max_run,
            "slice_entropy": slice_entropy,
            "tech_entropy": tech_entropy,
            "spoofed_ratio": spoofed_ratio,
            # Label as attack if any row in this window has attack==1
            "attack": int(sub["attack"].max())
        })

    return pd.DataFrame(out_rows)


#assume df_attack and oracle_model are already defined (from your previous code) ---

print("Injecting flood traffic...")
#df_attack, info = inject_flood_rand_multi(10, 20, 50, False, seed=42)
# Create floods in 10 distinct random day/time windows
for i in range(10):
    df_attack, _ = inject_flood_rand_multi(
        num_attackers=10, min_dup=20, max_dup=50, spoofed=True, seed=42+i
    )


print("Computing oracle-based spoof predictions...")
x_row = pd.get_dummies(df_attack[features])
x_row = x_row.reindex(columns=oracle_model.feature_names_in_, fill_value=0)
oracle_preds = oracle_model.predict(x_row)
df_attack["spoofed"] = (oracle_preds != df_attack["Use Case Group"]).astype(int)


print("Aggregating dataset by (Day, Time)...")
df_agglutinated = aggregate_window_features(df_attack)

print("\n=== Aggregated Dataset Summary ===")
print("Shape:", df_agglutinated.shape)
print("Columns:", list(df_agglutinated.columns))
print(df_agglutinated.head(10))
print("\nAttack label distribution:")
print(df_agglutinated["attack"].value_counts())


# Train Flooding Model

FEATURES = [
    "row_count",
    "unique_slices",
    "unique_tech",
    "top_repeat",
    "max_run",
    "slice_entropy",
    "tech_entropy",
    "spoofed_ratio"
]
TARGET = "attack"

X_flood = df_agglutinated[FEATURES].fillna(0)
y_flood = df_agglutinated[TARGET]

# Show label distribution
print("\nLabel counts after aggregation:")
print(y_flood.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_flood, y_flood, test_size=0.2, random_state=42, stratify=y_flood
)

print("\nTraining flooding detection model...")
flooding_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
flooding_model.fit(X_train, y_train)

# 
# Evaluate

y_pred = flooding_model.predict(X_test)
print("\n=== Flooding Detection Model (test set) ===")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Flooding"],
            yticklabels=["Normal","Flooding"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Flooding Detector")
plt.show()
