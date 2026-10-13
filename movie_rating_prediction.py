"""
CODSOFT Data Science Internship - Task 2: Movie Rating Prediction
Regression model that predicts an IMDb rating from genre, director, actors,
duration, year (and optionally votes).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RS = 42
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------ 1. Load & clean
df = pd.read_csv("IMDb Movies India.csv", encoding="latin-1")
print("Raw shape:", df.shape)

df["Year"] = pd.to_numeric(df["Year"].str.extract(r"(\d{4})")[0], errors="coerce")
df["Duration"] = pd.to_numeric(df["Duration"].str.extract(r"(\d+)")[0], errors="coerce")
df["Votes"] = pd.to_numeric(df["Votes"].str.replace(",", ""), errors="coerce")

df = df.dropna(subset=["Rating"]).reset_index(drop=True)   # target must exist
print("Rows with a rating:", len(df))

# ------------------------------------------------------------ 2. Feature engineering
# Genres are comma-separated -> multi-hot columns (top genres only)
genre_lists = df["Genre"].fillna("Unknown").str.split(r",\s*")
top_genres = pd.Series([g for gl in genre_lists for g in gl]).value_counts().head(15).index
for g in top_genres:
    df["G_" + g] = genre_lists.apply(lambda gl: int(g in gl))
df["NumGenres"] = genre_lists.apply(len)

for c in ["Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[c] = df[c].fillna("Unknown")

# ------------------------------------------------------------ 3. EDA plots
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
sns.histplot(df["Rating"], bins=30, kde=True, ax=ax[0, 0]); ax[0, 0].set_title("Rating distribution")
df[[c for c in df if c.startswith("G_")]].sum().sort_values().plot(kind="barh", ax=ax[0, 1], title="Movies per genre")
sns.scatterplot(x="Year", y="Rating", data=df, alpha=0.3, ax=ax[1, 0]); ax[1, 0].set_title("Rating vs year")
sns.scatterplot(x="Votes", y="Rating", data=df, alpha=0.3, ax=ax[1, 1]); ax[1, 1].set_xscale("log"); ax[1, 1].set_title("Rating vs votes (log)")
plt.tight_layout(); plt.savefig("eda.png", dpi=120); plt.close()

# ------------------------------------------------------------ 4. Build features
genre_cols = [c for c in df if c.startswith("G_")]
cat_cols = ["Director", "Actor 1", "Actor 2", "Actor 3"]
base_num = ["Year", "Duration", "NumGenres"] + genre_cols

def make_pipeline(model, use_votes):
    nums = base_num + (["Votes"] if use_votes else [])
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
        # TargetEncoder cross-fits internally, so no target leakage during training
        ("cat", TargetEncoder(target_type="continuous", random_state=RS), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])

models = {
    "Ridge Regression": Ridge(alpha=10),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RS),
    "Gradient Boosting": HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=300, random_state=RS),
}

X = df[base_num + ["Votes"] + cat_cols]
y = df["Rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS)
cv = KFold(5, shuffle=True, random_state=RS)

# ------------------------------------------------------------ 5. Compare models (CV on train)
print("\n5-fold CV R^2 on training data")
print(f"{'Model':20s} {'without Votes':>14s} {'with Votes':>12s}")
scores = {}
for name, m in models.items():
    r = []
    for uv in (False, True):
        s = cross_val_score(make_pipeline(m, uv), X_train, y_train, cv=cv, scoring="r2")
        r.append(s.mean())
    scores[name] = r
    print(f"{name:20s} {r[0]:14.3f} {r[1]:12.3f}")

# ------------------------------------------------------------ 6. Final test evaluation
best_name = max(scores, key=lambda k: scores[k][0])
print("\nBest model (without Votes):", best_name)
rows = []
for uv, label in ((False, "Pre-release features (no Votes)"), (True, "With Votes")):
    pipe = make_pipeline(models[best_name], uv).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    rows.append((label, mae, rmse, r2))
    if not uv:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(y_test, pred, alpha=0.3); ax[0].plot([1, 10], [1, 10], "r--")
        ax[0].set(xlabel="Actual rating", ylabel="Predicted rating", title="Predicted vs actual (test)")
        sns.histplot(y_test - pred, bins=30, kde=True, ax=ax[1]); ax[1].set_title("Residuals")
        plt.tight_layout(); plt.savefig("predictions.png", dpi=120); plt.close()

print("\n=== Test set results (%s) ===" % best_name)
print(pd.DataFrame(rows, columns=["Setup", "MAE", "RMSE", "R2"]).round(3).to_string(index=False))
print("Baseline MAE (always predict mean rating): %.3f" % mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())))
