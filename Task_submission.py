import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. This is for uncertainty Sampling Function

def uncertainty_sampling(model, X_pool):
    probs = model.predict_proba(X_pool)
    uncertainty = 1 - np.max(probs, axis=1)
    return np.argmax(uncertainty)


# 2. This is for dataset creation

X, y = make_classification(
    n_samples=3000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    class_sep=1.5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# 3. This is for initial labeled data 

INITIAL_SAMPLES = 20
rng = np.random.default_rng(42)
initial_idx = rng.choice(len(X_train), size=INITIAL_SAMPLES, replace=False)
X_labeled = X_train[initial_idx]
y_labeled = y_train[initial_idx]
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)


# 4. This is for baseline random sampling

baseline_model = LogisticRegression(max_iter=2000)
X_labeled_base = X_labeled.copy()
y_labeled_base = y_labeled.copy()
X_pool_base = X_pool.copy()
y_pool_base = y_pool.copy()
baseline_acc = []
baseline_labeled = []

for _ in range(100):
    rand_idx = rng.integers(0, len(X_pool_base))

    X_labeled_base = np.vstack([X_labeled_base, X_pool_base[rand_idx]])
    y_labeled_base = np.hstack([y_labeled_base, y_pool_base[rand_idx]])
    X_pool_base = np.delete(X_pool_base, rand_idx, axis=0)
    y_pool_base = np.delete(y_pool_base, rand_idx, axis=0)
    baseline_model.fit(X_labeled_base, y_labeled_base)
    preds = baseline_model.predict(X_test)
    baseline_acc.append(accuracy_score(y_test, preds))
    baseline_labeled.append(len(y_labeled_base))


# 5. This is for active learning loop

model = LogisticRegression(max_iter=2000)
model.fit(X_labeled, y_labeled)
al_acc = []
al_labeled = []
for _ in range(100):
    query_idx = uncertainty_sampling(model, X_pool)


# 6. This is for simulated human labeling with a scripted oracle

    X_labeled = np.vstack([X_labeled, X_pool[query_idx]])
    y_labeled = np.hstack([y_labeled, y_pool[query_idx]])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    model.fit(X_labeled, y_labeled)
    preds = model.predict(X_test)
    al_acc.append(accuracy_score(y_test, preds))
    al_labeled.append(len(y_labeled))


# 7. This is for plot performance vs labeled samples

plt.figure(figsize=(8, 5))
plt.plot(al_labeled, al_acc, label="Active Learning", linewidth=2)
plt.plot(baseline_labeled, baseline_acc, linestyle="--", label="Random Sampling")
plt.xlabel("Number of Labeled Samples")
plt.ylabel("Accuracy")
plt.title("Active Learning vs Random Sampling")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("active_learning_vs_random.png", dpi=300)
plt.close()


# 8. This is the final report

print("===== REPORT =====")
print(f"Initial labeled samples       : {INITIAL_SAMPLES}")
print(f"Final labeled samples         : {al_labeled[-1]}")
print(f"Final Active Learning Accuracy: {al_acc[-1]:.4f}")
print(f"Final Random Sampling Accuracy: {baseline_acc[-1]:.4f}")
print(f"Accuracy Gain                 : {al_acc[-1] - baseline_acc[-1]:.4f}")

