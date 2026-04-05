import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv("data/student_mental_health_burnout.csv")

print("Shape          :", df.shape)
print("Columns        :", list(df.columns))
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive stats:\n", df.describe().round(2))

# ── Create binary target label ─────────────────────────────────────
# stress_level >= 7  →  1 (High Burnout Risk)
# stress_level <  7  →  0 (Low Burnout Risk)
# Remove the stress_level line entirely, replace with this:
df["BurnoutRisk"] = df["burnout_level"].str.strip().str.lower().map({
    "low"   : 0,
    "medium": 0,
    "high"  : 1
})

print("\nClass distribution:\n", df["BurnoutRisk"].value_counts())
print("Class balance (%):\n",
      df["BurnoutRisk"].value_counts(normalize=True).round(3) * 100)

features = [
    "daily_sleep_hours",
    "daily_study_hours",
    "screen_time_hours",
    "attendance_percentage",
    "academic_pressure_score",
    "physical_activity_hours",
    "social_support_score"
]
X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["BurnoutRisk"]


# ══════════════════════════════════════════════════════════════════
# STEP 2 — EDA FIGURES  (saved to outputs/)
# ══════════════════════════════════════════════════════════════════
import os
os.makedirs("outputs", exist_ok=True)

# Figure 1 — Class Distribution
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["BurnoutRisk"].value_counts().sort_index()
bars = ax.bar(["Low Risk (0)", "High Risk (1)"], counts,
              color=["#4CAF50", "#F44336"], edgecolor="white", width=0.45)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1, str(c),
            ha="center", va="bottom", fontweight="bold")
ax.set_title("Figure 1: Burnout Risk Distribution", fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/fig1_class_distribution.png")
plt.show()

# Figure 2 — Feature Histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Figure 2: Feature Distributions", fontweight="bold")
for ax, col in zip(axes.flatten(), features):
    ax.hist(X[col].dropna(), bins=20, color="#5C85D6", edgecolor="white", alpha=0.85)
    ax.set_title(col.replace("_", " ").title(), fontsize=10)
axes.flatten()[-1].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/fig2_feature_histograms.png")
plt.show()

# Figure 3 — Boxplots by Burnout Risk
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Figure 3: Feature Values by Burnout Risk Level", fontweight="bold")
plot_df = X.copy()
plot_df["BurnoutRisk"] = y.values.astype(str)  # convert to string "0" and "1"
palette = {"0": "#4CAF50", "1": "#F44336"}      # keys must match as strings

for ax, col in zip(axes.flatten(), features):
    sns.boxplot(data=plot_df, x="BurnoutRisk", y=col,
                hue="BurnoutRisk", palette=palette,
                ax=ax, width=0.45, linewidth=1.2,
                legend=False)
    ax.set_title(col.replace("_", " ").title(), fontsize=10)
    ax.set_xlabel("Burnout Risk")
axes.flatten()[-1].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/fig3_boxplots.png")
plt.show()

# Figure 4 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = X.corr().round(2)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Figure 4: Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/fig4_correlation_heatmap.png")
plt.show()

# Figure 5 — Sleep vs Screen Time scatter
fig, ax = plt.subplots(figsize=(6, 5))
colors = plot_df["BurnoutRisk"].map({0: "#4CAF50", 1: "#F44336"})
ax.scatter(plot_df["sleep_hours"], plot_df["screen_time"],
           c=colors, alpha=0.6, s=35, edgecolors="none")
import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(color="#4CAF50", label="Low Risk"),
    mpatches.Patch(color="#F44336", label="High Risk")
], title="Burnout Risk")
ax.set_title("Figure 5: Sleep Hours vs Screen Time", fontweight="bold")
ax.set_xlabel("Sleep Hours")
ax.set_ylabel("Screen Time (hrs/day)")
plt.tight_layout()
plt.savefig("outputs/fig5_sleep_vs_screen.png")
plt.show()

print("\n✓ EDA figures saved to outputs/")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING  (fixed: impute + scale instead of drop)
# ══════════════════════════════════════════════════════════════════
# Split FIRST, then impute/scale to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values with column median
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=features)

# Scale features to same range (critical for LR, KNN, SVM)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTraining set : {X_train_sc.shape}")
print(f"Test set     : {X_test_sc.shape}")


# ══════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN 5 MODELS & COMPARE
# ══════════════════════════════════════════════════════════════════
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN"                : KNeighborsClassifier(n_neighbors=5),
    "SVM"                : SVC(probability=True, random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    # Tree-based models don't need scaling; others do
    if name in ["Decision Tree", "Random Forest"]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]

    trained_models[name] = (model, y_pred, y_prob)

    results.append({
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1 Score" : round(f1_score(y_test, y_pred,        zero_division=0), 4),
    })

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
print("\n" + "=" * 62)
print("MODEL COMPARISON TABLE")
print("=" * 62)
print(results_df.to_string(index=False))
results_df.to_csv("outputs/model_comparison.csv", index=False)
print("\n✓ Table saved: outputs/model_comparison.csv")

# Best model
best_name = results_df.iloc[0]["Model"]
print(f"\nBest model: {best_name}")
print("\nDetailed report for best model:")
print(classification_report(y_test, trained_models[best_name][1]))


# ══════════════════════════════════════════════════════════════════
# STEP 5 — RESULT VISUALISATIONS
# ══════════════════════════════════════════════════════════════════

# Figure 6 — Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(results_df))
width = 0.2
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
colors  = ["#5C85D6", "#4CAF50", "#FF9800", "#F44336"]
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i*width, results_df[metric], width,
           label=metric, color=color, alpha=0.85, edgecolor="white")
ax.set_title("Figure 6: Model Comparison", fontweight="bold")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/fig6_model_comparison.png")
plt.show()

# Figure 7 — Confusion Matrix for Best Model
_, y_pred_best, _ = trained_models[best_name]
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"],
            linewidths=0.5, ax=ax)
ax.set_title(f"Figure 7: Confusion Matrix — {best_name}", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/fig7_confusion_matrix.png")
plt.show()

# Figure 8 — ROC Curves (all models)
fig, ax = plt.subplots(figsize=(7, 5))
line_colors = ["#5C85D6","#4CAF50","#FF9800","#F44336","#9C27B0"]
for (name, (model, _, y_prob)), color in zip(trained_models.items(), line_colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})", color=color, linewidth=1.8)
ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.4)
ax.set_title("Figure 8: ROC Curves — All Models", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/fig8_roc_curves.png")
plt.show()

# Figure 9 — Feature Importance (Random Forest)
rf_model = trained_models["Random Forest"][0]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
importances.plot(kind="barh", ax=ax, color="#5C85D6", edgecolor="white")
ax.set_title("Figure 9: Feature Importance — Random Forest", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/fig9_feature_importance.png")
plt.show()


# ══════════════════════════════════════════════════════════════════
# STEP 6 — PREDICT FOR A NEW STUDENT
# ══════════════════════════════════════════════════════════════════
new_student = pd.DataFrame([{
    "sleep_hours"         : 4,
    "study_hours"         : 9,
    "screen_time"         : 8,
    "attendance"          : 65,
    "assignments_per_week": 5,
    "physical_activity"   : 1,
    "social_activity"     : 1
}])

# Use best model for prediction
best_model = trained_models[best_name][0]
if best_name in ["Decision Tree", "Random Forest"]:
    pred  = best_model.predict(new_student)[0]
    prob  = best_model.predict_proba(new_student)[0][1]
else:
    ns_sc = scaler.transform(imputer.transform(new_student))
    pred  = best_model.predict(ns_sc)[0]
    prob  = best_model.predict_proba(ns_sc)[0][1]

print("\n" + "=" * 40)
print("PREDICTION FOR NEW STUDENT")
print("=" * 40)
print(f"Model used        : {best_name}")
print(f"Burnout Risk Label: {'HIGH RISK' if pred == 1 else 'LOW RISK'}")
print(f"Burnout Probability: {prob:.2%}")

print("\n✓ All outputs saved to outputs/")
print("  Figures 1–9 are ready for your report's List of Figures.")