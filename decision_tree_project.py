import pandas as pd
import numpy as np
from collections import Counter
import math

# Load raw dataset
df = pd.read_csv("Airplane.csv")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "id"], errors='ignore')

# Drop rows with missing target
df = df.dropna(subset=["Arrival Delay in Minutes"])

# Define continuous features to discretize
continuous_features = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

# Convert to numeric
for col in continuous_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Discretize continuous features using quartile binning
for col in continuous_features:
    try:
        _, bins = pd.qcut(df[col], q=4, retbins=True, duplicates="drop")
        num_bins = len(bins) - 1
        if num_bins < 2:
            df.drop(columns=[col], inplace=True)
        else:
            labels = [f"Q{i+1}" for i in range(num_bins)]
            df[col] = pd.qcut(df[col], q=num_bins, labels=labels, duplicates="drop")
    except:
        df.drop(columns=[col], inplace=True)

# Drop any remaining missing data
df = df.dropna()

# Convert all to string
df = df.astype(str)

# Split into train/test
def train_test_split(data, test_size=0.2):
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(data) * (1 - test_size))
    return data.iloc[:split], data.iloc[split:]

# Entropy
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

# Gini index
def gini_index(labels):
    counts = Counter(labels)
    total = len(labels)
    return 1 - sum((count / total) ** 2 for count in counts.values())

# Information Gain
def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0
    for val in values:
        subset = data[data[feature] == val]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target])
    return total_entropy - weighted_entropy

# Tree node class
class Node:
    def __init__(self, feature=None, branches=None, label=None):
        self.feature = feature
        self.branches = branches if branches else {}
        self.label = label

    def is_leaf(self):
        return self.label is not None

# Recursive tree building
def build_tree(data, features, target, method="info_gain"):
    labels = data[target]
    if len(set(labels)) == 1:
        return Node(label=labels.iloc[0])
    if not features:
        majority = Counter(labels).most_common(1)[0][0]
        return Node(label=majority)

    if method == "gini":
        best_feature = min(features, key=lambda f: sum(
            gini_index(data[data[f] == val][target]) for val in data[f].unique()
        ))
    else:
        best_feature = max(features, key=lambda f: info_gain(data, f, target))

    node = Node(feature=best_feature)
    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        if subset.empty:
            majority = Counter(labels).most_common(1)[0][0]
            node.branches[val] = Node(label=majority)
        else:
            remaining = [f for f in features if f != best_feature]
            node.branches[val] = build_tree(subset, remaining, target, method)
    return node

# Tree printing
def print_tree(node, indent=""):
    if node.is_leaf():
        print(indent + f"Predict: {node.label}")
    else:
        for val, branch in node.branches.items():
            print(indent + f"If {node.feature} == {val}:")
            print_tree(branch, indent + "  ")

# Prediction
def predict_one(node, sample):
    while not node.is_leaf():
        value = sample.get(node.feature)
        if value in node.branches:
            node = node.branches[value]
        else:
            return None
    return node.label

def predict(node, data):
    return data.apply(lambda row: predict_one(node, row), axis=1)

# =========================
# Post-pruning functions
# =========================
def accuracy(node, data, target):
    preds = predict(node, data)
    return np.mean(preds == data[target])

def post_prune(node, validation_data, target):
    if node.is_leaf():
        return node

    for branch_value, child in node.branches.items():
        subset = validation_data[validation_data[node.feature] == branch_value]
        node.branches[branch_value] = post_prune(child, subset, target)

    # دقت قبل از هرس
    before_acc = accuracy(node, validation_data, target)

    # ساخت برگ بر اساس اکثریت
    labels = validation_data[target]
    if labels.empty:
        return node
    majority_label = Counter(labels).most_common(1)[0][0]
    leaf_node = Node(label=majority_label)

    # دقت بعد از هرس
    after_acc = accuracy(leaf_node, validation_data, target)

    if after_acc >= before_acc:
        return leaf_node
    else:
        return node

# =========================
# Main
# =========================
if __name__ == "__main__":
    # تقسیم به train / validation / test
    train, test = train_test_split(df, test_size=0.2)
    train, validation = train_test_split(train, test_size=0.25)  # 60% train, 20% validation, 20% test

    target = "satisfaction"
    features = list(train.columns)
    features.remove(target)

    print("Building decision tree...")
    tree = build_tree(train, features, target)

    print("\nDecision Tree Structure (Before Pruning):")
    print_tree(tree)

    # پس‌هرس
    tree = post_prune(tree, validation, target)

    print("\nDecision Tree Structure (After Pruning):")
    print_tree(tree)

    # دقت روی تست
    acc = np.mean(predict(tree, test) == test[target])
    print(f"\nAccuracy on test set after pruning: {acc:.2f}")
