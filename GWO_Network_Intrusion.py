import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==================================================
# CONFIGURATION
# ==================================================

DATASET = "your_dataset.csv"
TARGET = "label"

POPULATION = 8
ITERATIONS = 10
TOP_K = 20

np.random.seed(42)

# ==================================================
# 1. LOAD DATASET
# ==================================================

df = pd.read_csv(DATASET)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = LabelEncoder().fit_transform(df[TARGET].astype(str))

# Encode categorical input columns
X = pd.get_dummies(X, dummy_na=True)
X = X.replace([np.inf, -np.inf], np.nan)

# ==================================================
# 2. SPLIT DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Impute missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Normalize
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================================
# 3. INFORMATION GAIN FEATURE SELECTION
# ==================================================

ig = mutual_info_classif(X_train, y_train, random_state=42)

k = min(TOP_K, X_train.shape[1])
indices = np.argsort(ig)[::-1][:k]

X_train = X_train[:, indices]
X_test = X_test[:, indices]

print("Selected features:", k)

# ==================================================
# 4. PCA
# ==================================================

pca = PCA(n_components=0.95, random_state=42)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("PCA components:", X_train.shape[1])

# ==================================================
# 5. GREY WOLF OPTIMIZER
# ==================================================

class GWO:

    def __init__(self, objective, lb, ub):
        self.objective = objective
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(lb)

    def optimize(self):

        wolves = np.random.uniform(
            self.lb, self.ub, (POPULATION, self.dim)
        )

        alpha = np.zeros(self.dim)
        beta = np.zeros(self.dim)
        delta = np.zeros(self.dim)

        alpha_score = float("inf")
        beta_score = float("inf")
        delta_score = float("inf")

        history = []

        for t in range(ITERATIONS):

            for i in range(POPULATION):

                wolves[i] = np.clip(
                    wolves[i], self.lb, self.ub
                )

                score = self.objective(wolves[i])

                if score < alpha_score:

                    delta_score = beta_score
                    delta = beta.copy()

                    beta_score = alpha_score
                    beta = alpha.copy()

                    alpha_score = score
                    alpha = wolves[i].copy()

                elif score < beta_score:

                    delta_score = beta_score
                    delta = beta.copy()

                    beta_score = score
                    beta = wolves[i].copy()

                elif score < delta_score:

                    delta_score = score
                    delta = wolves[i].copy()

            a = 2 - 2 * t / ITERATIONS

            for i in range(POPULATION):

                new_positions = []

                for leader in [alpha, beta, delta]:

                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)

                    A = 2 * a * r1 - a
                    C = 2 * r2

                    D = abs(C * leader - wolves[i])

                    new_positions.append(leader - A * D)

                wolves[i] = np.mean(new_positions, axis=0)

                wolves[i] = np.clip(
                    wolves[i], self.lb, self.ub
                )

            history.append(alpha_score)

            print(
                f"Iteration {t+1}: Best Fitness = "
                f"{alpha_score:.5f}"
            )

        return alpha, alpha_score, history


# ==================================================
# 6. CLASSIFIERS AND SEARCH SPACES
# ==================================================

def create_model(name, p):

    if name == "RF":
        return RandomForestClassifier(
            n_estimators=int(p[0]),
            max_depth=int(p[1]),
            min_samples_split=int(p[2]),
            random_state=42,
            n_jobs=-1
        )

    if name == "DT":
        return DecisionTreeClassifier(
            max_depth=int(p[0]),
            min_samples_split=int(p[1]),
            min_samples_leaf=int(p[2]),
            random_state=42
        )

    if name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=int(p[0]),
            weights="uniform" if p[1] < 0.5 else "distance",
            p=int(p[2])
        )

    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(int(p[0]),),
            alpha=10**p[1],
            learning_rate_init=10**p[2],
            max_iter=300,
            random_state=42,
            early_stopping=True
        )


spaces = {
    "RF": ([50, 5, 2], [200, 30, 10]),
    "DT": ([2, 2, 1], [30, 20, 10]),
    "KNN": ([3, 0, 1], [25, 1, 2]),
    "MLP": ([20, -5, -4], [150, -1, -2])
}

# ==================================================
# 7. GWO HYPERPARAMETER OPTIMIZATION
# ==================================================

models = {}
histories = {}

min_count = np.bincount(y_train).min()

if min_count < 2:
    raise ValueError("SMOTE needs at least 2 samples per class.")

smote_neighbors = min(5, min_count - 1)


def optimize(name):

    lb, ub = spaces[name]

    def fitness(p):

        model = create_model(name, p)

        pipeline = Pipeline([
            ("smote", SMOTE(
                random_state=42,
                k_neighbors=smote_neighbors
            )),
            ("classifier", model)
        ])

        try:
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=3,
                scoring="f1_weighted"
            )
            return 1 - scores.mean()

        except Exception:
            return 1.0

    gwo = GWO(fitness, lb, ub)

    best, score, history = gwo.optimize()

    final_model = create_model(name, best)

    final_pipeline = Pipeline([
        ("smote", SMOTE(
            random_state=42,
            k_neighbors=smote_neighbors
        )),
        ("classifier", final_model)
    ])

    final_pipeline.fit(X_train, y_train)

    models[name] = final_pipeline
    histories[name] = history

    print(name, "Best parameters:", best)
    print(name, "Best fitness:", score)


for name in spaces:
    print("\nOptimizing:", name)
    optimize(name)

# ==================================================
# 8. SOFT VOTING
# ==================================================

class_order = np.unique(y_train)

probabilities = []

for name, model in models.items():

    raw = model.predict_proba(X_test)

    aligned = np.zeros((len(X_test), len(class_order)))

    for j, cls in enumerate(model.classes_):
        aligned[:, np.where(class_order == cls)[0][0]] = raw[:, j]

    probabilities.append(aligned)

ensemble_prob = np.mean(probabilities, axis=0)

ensemble_pred = class_order[np.argmax(ensemble_prob, axis=1)]

# ==================================================
# 9. PERFORMANCE EVALUATION
# ==================================================

def evaluate(name, actual, predicted, prob):

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual, predicted,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual, predicted,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(actual, predicted, labels=class_order)

    far_values = []

    for i in range(len(class_order)):

        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        far_values.append(
            fp / (fp + tn) if fp + tn else 0
        )

    far = np.mean(far_values)

    try:
        if len(class_order) == 2:
            auc = roc_auc_score(actual, prob[:, 1])
        else:
            auc = roc_auc_score(
                actual, prob,
                multi_class="ovr",
                average="weighted",
                labels=class_order
            )
    except ValueError:
        auc = np.nan

    print("\nMODEL:", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Detection Rate:", recall)
    print("F1 Score:", f1)
    print("False Alarm Rate:", far)
    print("ROC-AUC:", auc)

    print(classification_report(
        actual, predicted,
        labels=class_order,
        zero_division=0
    ))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "AUC": auc
    }


results = []

for name, model in models.items():

    pred = model.predict(X_test)
    raw = model.predict_proba(X_test)

    prob = np.zeros((len(X_test), len(class_order)))

    for j, cls in enumerate(model.classes_):
        prob[:, np.where(class_order == cls)[0][0]] = raw[:, j]

    results.append(
        evaluate(name, y_test, pred, prob)
    )

results.append(
    evaluate(
        "GWO Soft Voting",
        y_test,
        ensemble_pred,
        ensemble_prob
    )
)

# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

print("\nFINAL RESULTS")
print(results_df)

results_df.to_csv(
    "GWO_Results.csv",
    index=False
)

# ==================================================
# 11. CONFUSION MATRIX
# ==================================================

sns.heatmap(
    confusion_matrix(y_test, ensemble_pred,
                     labels=class_order),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("GWO Intrusion Detection Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("GWO_Confusion_Matrix.png")
plt.show()

# ==================================================
# 12. CONVERGENCE CURVE
# ==================================================

plt.figure(figsize=(9, 5))

for name, history in histories.items():
    plt.plot(history, label=name)

plt.title("Grey Wolf Optimizer Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.legend()
plt.grid()
plt.savefig("GWO_Convergence.png")
plt.show()

# ==================================================
# 13. PERFORMANCE COMPARISON
# ==================================================

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1"]
].plot(kind="bar", figsize=(10, 5))

plt.title("Classifier Performance Comparison")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("GWO_Performance.png")
plt.show()

print("\nGWO INTRUSION DETECTION COMPLETED")