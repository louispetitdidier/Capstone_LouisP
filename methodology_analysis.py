"""
==============================================================================
Methodology Analysis Script
==============================================================================
Capstone: Discovering Hidden Patterns in E-Commerce Using Clustering
          Techniques - A Case Study on the Olist Brazilian Marketplace

Research Question: How can clustering techniques uncover meaningful customer
                   segments in a real e-commerce marketplace?

Author:  Louis Petitdidier
Date:    February 2026
Seed:    42 (reproducibility)
==============================================================================
"""

import os, sys, warnings, platform
import numpy as np
import pandas as pd

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
from math import pi

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

DIR = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(DIR, "figures")
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "figure.figsize": (10, 6)})

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}")

# ============================================================================
# 1. DATA LOADING & MERGING
# ============================================================================
print("\n=== 1. DATA LOADING & MERGING ===")

orders    = pd.read_csv(os.path.join(DIR, "olist_orders_dataset.csv"))
customers = pd.read_csv(os.path.join(DIR, "olist_customers_dataset.csv"))
items     = pd.read_csv(os.path.join(DIR, "olist_order_items_dataset.csv"))
payments  = pd.read_csv(os.path.join(DIR, "olist_order_payments_dataset.csv"))
reviews   = pd.read_csv(os.path.join(DIR, "olist_order_reviews_dataset.csv"))

for c in ["order_purchase_timestamp", "order_delivered_customer_date",
           "order_estimated_delivery_date"]:
    orders[c] = pd.to_datetime(orders[c])

orders = orders[orders["order_status"] == "delivered"].copy()
print(f"  Delivered orders: {len(orders)}")

# Merge
df = orders.merge(customers, on="customer_id", how="left")

pay_agg = payments.groupby("order_id").agg(
    payment_value=("payment_value", "sum"),
    payment_installments=("payment_installments", "mean")
).reset_index()
df = df.merge(pay_agg, on="order_id", how="left")
df = df.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")

df["delivery_time_days"] = (
    df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
).dt.total_seconds() / 86400

print(f"  Merged DataFrame: {df.shape}")

# ============================================================================
# 2. FEATURE ENGINEERING (RFM + Behavioral)
# ============================================================================
print("\n=== 2. FEATURE ENGINEERING ===")

ref_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

customer_df = df.groupby("customer_unique_id").agg(
    recency=("order_purchase_timestamp", lambda x: (ref_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum"),
    avg_review_score=("review_score", "mean"),
    avg_delivery_time=("delivery_time_days", "mean"),
    avg_installments=("payment_installments", "mean"),
).reset_index()

print(f"  Unique customers: {len(customer_df)}")
print(customer_df.describe().round(2).to_string())

# ============================================================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================================================
print("\n=== 3. DATA CLEANING & PREPROCESSING ===")

# Drop NaN
print(f"  Missing:\n{customer_df.isnull().sum().to_string()}")
customer_df = customer_df.dropna().copy()
print(f"  After dropna: {len(customer_df)}")

feat = ["recency", "frequency", "monetary",
        "avg_review_score", "avg_delivery_time", "avg_installments"]

# Outlier capping (IQR x 1.5) - skip features with zero IQR
for col in feat:
    Q1, Q3 = customer_df[col].quantile(0.25), customer_df[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        print(f"  {col}: IQR=0, skipping capping")
        continue
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((customer_df[col] < lo) | (customer_df[col] > hi)).sum()
    customer_df[col] = customer_df[col].clip(lo, hi)
    print(f"  {col}: {n_out} outliers capped [{lo:.2f}, {hi:.2f}]")

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(customer_df[feat])
print(f"  Standardized shape: {X.shape}")

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n=== 4. EDA ===")

# 4a. Distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col in zip(axes.flat, feat):
    customer_df[col].hist(bins=50, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title(col.replace("_", " ").title())
fig.suptitle("Feature Distributions (After Outlier Capping)", fontsize=14, y=1.01)
fig.tight_layout()
save(fig, "01_feature_distributions.png")

# 4b. Boxplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col in zip(axes.flat, feat):
    sns.boxplot(y=customer_df[col], ax=ax, color="#55A868", width=0.4)
    ax.set_title(col.replace("_", " ").title())
fig.suptitle("Box Plots of Customer Features", fontsize=14, y=1.01)
fig.tight_layout()
save(fig, "02_feature_boxplots.png")

# 4c. Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = customer_df[feat].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix")
fig.tight_layout()
save(fig, "03_correlation_heatmap.png")
print(f"  Correlations:\n{corr.round(3).to_string()}")

# 4d. Pairplot (small sample for speed)
sample = customer_df[feat].sample(min(1500, len(customer_df)), random_state=SEED)
pp = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 8})
pp.figure.suptitle("Pairplot (Sampled)", y=1.01)
pp.savefig(os.path.join(FIG, "04_pairplot.png"), bbox_inches="tight")
plt.close(pp.figure)
print("  [saved] 04_pairplot.png")

# ============================================================================
# 5. PCA
# ============================================================================
print("\n=== 5. PCA ===")

pca_full = PCA(random_state=SEED)
pca_full.fit(X)
ev = pca_full.explained_variance_ratio_
cum = np.cumsum(ev)
for i in range(len(ev)):
    print(f"  PC{i+1}: {ev[i]:.4f}  cumulative: {cum[i]:.4f}")

n_comp = int(np.argmax(cum >= 0.85)) + 1
print(f"  -> Keeping {n_comp} components (>= 85% variance)")

# Scree plot
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(range(1, len(ev)+1), ev, color="#4C72B0", alpha=0.8, label="Individual")
ax2 = ax1.twinx()
ax2.plot(range(1, len(cum)+1), cum, "o-", color="#C44E52", label="Cumulative")
ax2.axhline(0.85, ls="--", color="grey", alpha=0.6)
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Individual Variance")
ax2.set_ylabel("Cumulative Variance")
ax1.set_title("PCA Scree Plot")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="center right")
fig.tight_layout()
save(fig, "05_pca_scree.png")

pca = PCA(n_components=n_comp, random_state=SEED)
Xp = pca.fit_transform(X)

# Scatter PC1 vs PC2
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Xp[:, 0], Xp[:, 1], alpha=0.2, s=3, c="#4C72B0")
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax.set_title("Customers in PCA Space")
fig.tight_layout()
save(fig, "06_pca_scatter.png")

# Loadings
loadings = pd.DataFrame(pca_full.components_[:n_comp].T,
                         columns=[f"PC{i+1}" for i in range(n_comp)],
                         index=feat)
print(f"  Loadings:\n{loadings.round(3).to_string()}")

# ============================================================================
# 6. OPTIMAL NUMBER OF CLUSTERS
# ============================================================================
print("\n=== 6. OPTIMAL K ===")

K_range = range(2, 11)
inertias, sils = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    lab = km.fit_predict(Xp)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xp, lab))
    print(f"  k={k}  inertia={km.inertia_:.0f}  silhouette={sils[-1]:.4f}")

# Elbow
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(K_range), inertias, "o-", color="#4C72B0", lw=2)
ax.set_xlabel("k"); ax.set_ylabel("Inertia (WCSS)")
ax.set_title("Elbow Method")
fig.tight_layout()
save(fig, "07_elbow_method.png")

# Silhouette
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(K_range), sils, "o-", color="#55A868", lw=2)
best_k = list(K_range)[int(np.argmax(sils))]
ax.axvline(best_k, ls="--", color="#C44E52", label=f"Best k={best_k}")
ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score vs k")
ax.legend()
fig.tight_layout()
save(fig, "08_silhouette_vs_k.png")

K = best_k
print(f"  ** Optimal k = {K} **")

# ============================================================================
# 7. CLUSTERING MODELS
# ============================================================================
print("\n=== 7. CLUSTERING ===")
results = {}

# --- 7a. K-Means (Baseline) ---
print("  7a. K-Means...")
kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=10)
results["K-Means"] = kmeans.fit_predict(Xp)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(Xp[:, 0], Xp[:, 1], c=results["K-Means"], cmap="viridis", alpha=0.3, s=3)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c="red", marker="X", s=200, edgecolors="black", label="Centroids")
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax.set_title(f"K-Means (k={K})"); ax.legend()
fig.colorbar(sc, ax=ax, label="Cluster")
fig.tight_layout()
save(fig, "09_kmeans_clusters.png")

# --- 7b. Hierarchical (subsample + KNN assign) ---
print("  7b. Hierarchical Clustering...")
HSAMP = 5000
hidx = np.random.choice(len(Xp), size=min(HSAMP, len(Xp)), replace=False)
Xh = Xp[hidx]

# Dendrogram on small subsample
Z = linkage(Xh[:2000], method="ward")
fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=30, ax=ax,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.7 * max(Z[:, 2]))
ax.set_title("Dendrogram (Ward, Sample 2000)")
ax.set_xlabel("Cluster Size"); ax.set_ylabel("Distance")
fig.tight_layout()
save(fig, "10_dendrogram.png")

# Fit on subsample, assign rest via KNN
agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
sub_labels = agg.fit_predict(Xh)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xh, sub_labels)
results["Hierarchical"] = knn.predict(Xp)
print(f"    Fitted on {HSAMP} points, assigned full data via KNN")

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(Xp[:, 0], Xp[:, 1], c=results["Hierarchical"], cmap="viridis", alpha=0.3, s=3)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax.set_title(f"Hierarchical Clustering (k={K}, Ward)")
fig.colorbar(sc, ax=ax, label="Cluster")
fig.tight_layout()
save(fig, "11_hierarchical_clusters.png")

# --- 7c. DBSCAN ---
print("  7c. DBSCAN...")
k_nn = min(2 * Xp.shape[1], 10)
nn = NearestNeighbors(n_neighbors=k_nn)
nn.fit(Xp)
dists, _ = nn.kneighbors(Xp)
k_dist = np.sort(dists[:, -1])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_dist, color="#4C72B0", lw=1)
ax.set_xlabel("Points (sorted)")
ax.set_ylabel(f"{k_nn}-NN Distance")
ax.set_title("k-Distance Plot (DBSCAN eps)")
fig.tight_layout()
save(fig, "12_kdistance_plot.png")

eps_val = float(np.percentile(k_dist, 90))
db = DBSCAN(eps=eps_val, min_samples=k_nn)
results["DBSCAN"] = db.fit_predict(Xp)
n_cl = len(set(results["DBSCAN"]) - {-1})
n_noise = int((results["DBSCAN"] == -1).sum())
print(f"    eps={eps_val:.3f}  clusters={n_cl}  noise={n_noise}")

fig, ax = plt.subplots(figsize=(8, 6))
for lbl in sorted(set(results["DBSCAN"])):
    m = results["DBSCAN"] == lbl
    name = "Noise" if lbl == -1 else f"Cluster {lbl}"
    ax.scatter(Xp[m, 0], Xp[m, 1], alpha=0.3, s=3, label=name)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax.set_title(f"DBSCAN (eps={eps_val:.2f})")
ax.legend(markerscale=3, fontsize=8)
fig.tight_layout()
save(fig, "13_dbscan_clusters.png")

# --- 7d. GMM ---
print("  7d. GMM...")
bics, aics = [], []
for k in range(2, 11):
    g = GaussianMixture(n_components=k, covariance_type="full",
                        random_state=SEED, n_init=3)
    g.fit(Xp)
    bics.append(g.bic(Xp)); aics.append(g.aic(Xp))
    print(f"    k={k}  BIC={bics[-1]:.0f}  AIC={aics[-1]:.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(2,11), bics, "o-", color="#4C72B0", label="BIC")
ax.plot(range(2,11), aics, "s--", color="#C44E52", label="AIC")
ax.set_xlabel("k"); ax.set_ylabel("Information Criterion")
ax.set_title("GMM Model Selection (BIC & AIC)")
ax.legend()
fig.tight_layout()
save(fig, "14_gmm_bic_aic.png")

gmm = GaussianMixture(n_components=K, covariance_type="full",
                       random_state=SEED, n_init=5)
results["GMM"] = gmm.fit_predict(Xp)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(Xp[:, 0], Xp[:, 1], c=results["GMM"], cmap="viridis", alpha=0.3, s=3)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax.set_title(f"GMM Clustering (k={K})")
fig.colorbar(sc, ax=ax, label="Cluster")
fig.tight_layout()
save(fig, "15_gmm_clusters.png")

# --- 7e. Deep Clustering (Autoencoder + K-Means) ---
print("  7e. Deep Clustering (Autoencoder + K-Means)...")
enc_dim = 2
ae = MLPRegressor(
    hidden_layer_sizes=(32, 16, enc_dim, 16, 32),
    activation="relu", solver="adam", max_iter=300,
    random_state=SEED, early_stopping=True,
    validation_fraction=0.1, learning_rate_init=0.001,
    batch_size=256, verbose=False)
ae.fit(X, X)
mse = float(np.mean((ae.predict(X) - X) ** 2))
print(f"    Reconstruction MSE: {mse:.4f}")

# Extract bottleneck encoding (forward through first 3 layers)
def encode(model, data):
    h = data.copy()
    n_enc = len(model.hidden_layer_sizes_) // 2 + 1
    for i in range(n_enc):
        h = h @ model.coefs_[i] + model.intercepts_[i]
        if i < n_enc - 1:
            h = np.maximum(h, 0)
    return h

Xe = encode(ae, X)
print(f"    Encoded shape: {Xe.shape}")

km_deep = KMeans(n_clusters=K, random_state=SEED, n_init=10)
results["Deep Clustering"] = km_deep.fit_predict(Xe)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(Xe[:, 0], Xe[:, 1], c=results["Deep Clustering"],
                cmap="viridis", alpha=0.3, s=3)
ax.set_xlabel("Encoding Dim 1"); ax.set_ylabel("Encoding Dim 2")
ax.set_title(f"Deep Clustering (Autoencoder + K-Means, k={K})")
fig.colorbar(sc, ax=ax, label="Cluster")
fig.tight_layout()
save(fig, "16_deep_clustering.png")

# ============================================================================
# 8. EVALUATION & COMPARISON
# ============================================================================
print("\n=== 8. EVALUATION ===")

def evaluate(data, labels, name):
    mask = labels >= 0
    n_cl = len(set(labels[mask]))
    if n_cl < 2:
        return {"Method": name, "Clusters": n_cl,
                "Silhouette": np.nan, "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan}
    return {
        "Method": name, "Clusters": n_cl,
        "Silhouette": silhouette_score(data[mask], labels[mask]),
        "Davies-Bouldin": davies_bouldin_score(data[mask], labels[mask]),
        "Calinski-Harabasz": calinski_harabasz_score(data[mask], labels[mask]),
    }

rows = []
for method, lab in results.items():
    data = Xe if method == "Deep Clustering" else Xp
    row = evaluate(data, lab, method)
    rows.append(row)
    sil = row["Silhouette"]
    if np.isnan(sil):
        print(f"  {method:20s}  [insufficient clusters]")
    else:
        print(f"  {method:20s}  SIL={sil:.4f}  DBI={row['Davies-Bouldin']:.4f}  CHI={row['Calinski-Harabasz']:.0f}")

eval_df = pd.DataFrame(rows)
print(f"\n{eval_df.to_string(index=False)}")
eval_df.to_csv(os.path.join(DIR, "clustering_comparison.csv"), index=False)
print("  [saved] clustering_comparison.csv")

# Comparison chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, color in zip(axes, ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
                              ["#4C72B0", "#DD8452", "#55A868"]):
    valid = eval_df.dropna(subset=[metric])
    ax.barh(valid["Method"], valid[metric], color=color, edgecolor="white")
    ax.set_xlabel(metric); ax.set_title(metric)
fig.suptitle("Clustering Method Comparison", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "17_method_comparison.png")

# ============================================================================
# 9. CLUSTER PROFILING
# ============================================================================
print("\n=== 9. CLUSTER PROFILING (K-Means) ===")

from scipy.stats import f_oneway, kruskal

# Switch to 1-indexed clusters for the report
customer_df["cluster"] = results["K-Means"] + 1

profile = customer_df.groupby("cluster")[feat].mean()
print(profile.round(3).to_string())
profile.to_csv(os.path.join(DIR, "cluster_profiles.csv"))
print("  [saved] cluster_profiles.csv")

# 9b. Statistical Tests for feature differences
print("  Statistical tests (ANOVA/Kruskal-Wallis) per feature:")
test_results = []
for col in feat:
    groups = [customer_df[customer_df["cluster"] == c][col].dropna().values for c in sorted(customer_df["cluster"].unique())]
    # Using Kruskal as variables like frequency and recency are heavily skewed
    stat, p_val = kruskal(*groups)
    test_results.append({"Feature": col, "Test": "Kruskal-Wallis", "Statistic": stat, "p-value": p_val})
    print(f"    {col:20s}: p={p_val:.2e}")
pd.DataFrame(test_results).to_csv(os.path.join(DIR, "cluster_statistical_tests.csv"), index=False)

# 9c. Bar Plots per feature
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, col in zip(axes.flat, feat):
    sns.barplot(x="cluster", y=col, data=customer_df, ax=ax, palette="viridis", ci=None, edgecolor="black")
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean Value")
fig.suptitle("Mean Feature Values by Cluster", fontsize=16, y=1.02)
fig.tight_layout()
save(fig, "20_cluster_feature_bars.png")

# Radar chart
cats = [c.replace("_", "\n") for c in feat]
N = len(cats)
angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
pnorm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_r = plt.cm.viridis(np.linspace(0.2, 0.8, K))
for idx, (cid, row) in enumerate(pnorm.iterrows()):
    vals = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, vals, "o-", lw=2, label=f"Cluster {cid}", color=colors_r[idx])
    ax.fill(angles, vals, alpha=0.1, color=colors_r[idx])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, fontsize=9)
ax.set_title("Cluster Profiles (Normalised)", fontsize=14, y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
fig.tight_layout()
save(fig, "18_radar_profiles.png")

# Cluster sizes
fig, ax = plt.subplots(figsize=(7, 5))
counts = customer_df["cluster"].value_counts().sort_index()
ax.bar(np.arange(len(counts)), counts.values, color="#4C72B0", edgecolor="white")
ax.set_xticks(np.arange(len(counts)))
ax.set_xticklabels([str(c) for c in counts.index])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, str(v), ha="center")
ax.set_xlabel("Cluster"); ax.set_ylabel("Customers")
ax.set_title("Cluster Sizes (K-Means)")
fig.tight_layout()
save(fig, "19_cluster_sizes.png")

# ============================================================================
# 10. SEPARATE YEAR ANALYSIS (2018)
# ============================================================================
print("\n=== 10. SEPARATE YEAR ANALYSIS (2018) ===")
df_2018 = df[df["order_purchase_timestamp"].dt.year == 2018].copy()
ref_date_2018 = df_2018["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

cust_2018 = df_2018.groupby("customer_unique_id").agg(
    recency=("order_purchase_timestamp", lambda x: (ref_date_2018 - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum"),
    avg_review_score=("review_score", "mean"),
    avg_delivery_time=("delivery_time_days", "mean"),
    avg_installments=("payment_installments", "mean"),
).dropna().reset_index()

print(f"  2018 cohort size: {len(cust_2018)}")

# Cap outliers for 2018
for col in feat:
    Q1, Q3 = cust_2018[col].quantile(0.25), cust_2018[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR > 0:
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        cust_2018[col] = cust_2018[col].clip(lo, hi)

sc_18 = StandardScaler()
X_18 = sc_18.fit_transform(cust_2018[feat])
pca_18 = PCA(n_components=n_comp, random_state=SEED)
Xp_18 = pca_18.fit_transform(X_18)

km_18 = KMeans(n_clusters=K, random_state=SEED, n_init=10)
cust_2018["cluster"] = km_18.fit_predict(Xp_18) + 1

prof_18 = cust_2018.groupby("cluster")[feat].mean()
print(prof_18.round(3).to_string())
prof_18.to_csv(os.path.join(DIR, "cluster_profiles_2018.csv"))
print("  [saved] cluster_profiles_2018.csv")

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(Xp_18[:, 0], Xp_18[:, 1], c=cust_2018["cluster"], cmap="viridis", alpha=0.3, s=3)
ax.set_title(f"K-Means (k={K}) for 2018 cohort")
ax.set_xlabel(f"PC1"); ax.set_ylabel(f"PC2")
fig.colorbar(sc, ax=ax, label="Cluster")
fig.tight_layout()
save(fig, "21_kmeans_2018.png")

# ============================================================================
# 11. REPRODUCIBILITY
# ============================================================================
print("\n=== 10. REPRODUCIBILITY ===")
import sklearn, scipy
print(f"  Python:       {platform.python_version()}")
print(f"  NumPy:        {np.__version__}")
print(f"  pandas:       {pd.__version__}")
print(f"  matplotlib:   {matplotlib.__version__}")
print(f"  seaborn:      {sns.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  SciPy:        {scipy.__version__}")
print(f"  Seed:         {SEED}")
print(f"  OS:           {platform.system()} {platform.release()}")

print(f"\n{'='*60}")
print(f"DONE - All figures in: {FIG}")
print(f"{'='*60}")
