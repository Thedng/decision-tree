
import pandas as pd
import numpy as np
from collections import Counter
import math
import random

# Load preprocessed data
df = pd.read_csv("airplane_processed.csv")

# Split into train/test
def train_test_split(data, test_size=0.2):
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int((1 - test_size) * len(data))
    return data.iloc[:split_idx], data.iloc[split_idx:]

# Entropy calculation
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

# Gini Index calculation
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
        most_common = Counter(labels).most_common(1)[0][0]
        return Node(label=most_common)

    if method == "gini":
        scores = [(f, gini_index(data[data[f] == val][target])) for f in features for val in data[f].unique()]
        best_feature = min(set(f for f, _ in scores), key=lambda f: sum(s for ff, s in scores if ff == f))
    else:
        best_feature = max(features, key=lambda f: info_gain(data, f, target))

    node = Node(feature=best_feature)
    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        if subset.empty:
            most_common = Counter(labels).most_common(1)[0][0]
            node.branches[val] = Node(label=most_common)
        else:
            remaining_features = [f for f in features if f != best_feature]
            node.branches[val] = build_tree(subset, remaining_features, target, method)
    return node

# Print tree textually
def print_tree(node, indent=""):
    if node.is_leaf():
        print(indent + "Predict:", node.label)
    else:
        for val, subtree in node.branches.items():
            print(f"{indent}If {node.feature} == {val}:")
            print_tree(subtree, indent + "  ")

# Prediction
def predict_one(node, sample):
    while not node.is_leaf():
        value = sample.get(node.feature)
        if value in node.branches:
            node = node.branches[value]
        else:
            return None
    return node.label

def predict(node, df):
    return df.apply(lambda row: predict_one(node, row), axis=1)

# Run
if __name__ == "__main__":
    train_data, test_data = train_test_split(df)
    target = "satisfaction"
    features = list(train_data.columns)
    features.remove(target)

    print("Building tree...")
    tree = build_tree(train_data, features, target, method="info_gain")

    print("\nDecision Tree:")
    print_tree(tree)

    preds = predict(tree, test_data)
    acc = np.mean(preds == test_data[target])
    print(f"\nAccuracy on test set: {acc:.2f}")
