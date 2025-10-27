import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv(
    "cache_hit_data.csv",
    names=["compute_multiplier", "input_tokens", "output_tokens", "ctps"],
    header=0,
)
df = df[
    df["ctps"].notna()
    & (df["input_tokens"] > 0)
    & (df["output_tokens"] > 10)
    & (df["output_tokens"].notna())
]

# Features for clustering
X = df[["compute_multiplier", "input_tokens", "output_tokens", "ctps"]].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster into 2 groups (cached or not)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

cluster_stats = []
for i in range(2):
    cluster_df = df[labels == i]
    stats = {
        "cluster": i,
        "avg_input": cluster_df["input_tokens"].mean(),
        "avg_output": cluster_df["output_tokens"].mean(),
        "avg_ctps": cluster_df["ctps"].mean(),
        "median_ctps": cluster_df["ctps"].median(),
        "count": len(cluster_df),
    }
    cluster_stats.append(stats)
    print(f"\nCluster {i}:")
    print(f"  Count: {stats['count']}")
    print(f"  Avg input: {stats['avg_input']:.0f}")
    print(f"  Avg output: {stats['avg_output']:.0f}")
    print(f"  Avg CTPS: {stats['avg_ctps']:.0f}")
    print(f"  Median CTPS: {stats['median_ctps']:.0f}")

if (
    cluster_stats[0]["avg_input"] < cluster_stats[1]["avg_input"]
    and cluster_stats[0]["avg_ctps"] > cluster_stats[1]["avg_ctps"]
):
    cached_cluster = 0
elif (
    cluster_stats[1]["avg_input"] < cluster_stats[0]["avg_input"]
    and cluster_stats[1]["avg_ctps"] > cluster_stats[0]["avg_ctps"]
):
    cached_cluster = 1
else:
    cached_cluster = 0 if cluster_stats[0]["median_ctps"] > cluster_stats[1]["median_ctps"] else 1

print(f"\n==> Cached cluster identified as: {cached_cluster}")

df["is_cached"] = labels == cached_cluster

print(f"\nCache hit rate: {df['is_cached'].mean():.1%}")
print(f"Cached cluster - avg input tokens: {df[df['is_cached']]['input_tokens'].mean():.0f}")
print(f"Cached cluster - avg output tokens: {df[df['is_cached']]['output_tokens'].mean():.0f}")
print(f"Uncached cluster - avg input tokens: {df[~df['is_cached']]['input_tokens'].mean():.0f}")
print(f"Uncached cluster - avg output tokens: {df[~df['is_cached']]['output_tokens'].mean():.0f}")


def make_cache_detector(scaler, kmeans, cached_cluster):
    def is_kvcache_hit(metrics, compute_multiplier):
        if not all(k in metrics for k in ["ctps", "it", "ot"]) or metrics["ctps"] is None:
            return False
        if metrics["ot"] < 10:
            return False
        X = np.array([[compute_multiplier, metrics["it"], metrics["ot"], metrics["ctps"]]])
        X_scaled = scaler.transform(X)
        cluster = kmeans.predict(X_scaled)[0]
        return cluster == cached_cluster

    return is_kvcache_hit


is_kvcache_hit = make_cache_detector(scaler, kmeans, cached_cluster)

model_data = {"scaler": scaler, "kmeans": kmeans, "cached_cluster": cached_cluster}
with open("cache_hit_classifier.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Visualize...
fig = plt.figure(figsize=(15, 10))

# Plot 1: Input tokens vs CTPS
plt.subplot(2, 2, 1)
plt.scatter(
    df[df["is_cached"]]["input_tokens"], df[df["is_cached"]]["ctps"], alpha=0.3, label="Cached", s=1
)
plt.scatter(
    df[~df["is_cached"]]["input_tokens"],
    df[~df["is_cached"]]["ctps"],
    alpha=0.3,
    label="Uncached",
    s=1,
)
plt.xlabel("Input Tokens")
plt.ylabel("CTPS")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.title("Input Tokens vs CTPS")

# Plot 2: Distribution of input tokens
plt.subplot(2, 2, 2)
plt.hist(
    [df[df["is_cached"]]["input_tokens"], df[~df["is_cached"]]["input_tokens"]],
    bins=50,
    label=["Cached", "Uncached"],
    alpha=0.7,
)
plt.xlabel("Input Tokens")
plt.ylabel("Count")
plt.xscale("log")
plt.legend()
plt.title("Input Token Distribution")

# Plot 3: CTPS distribution
plt.subplot(2, 2, 3)
plt.hist(
    [df[df["is_cached"]]["ctps"], df[~df["is_cached"]]["ctps"]],
    bins=50,
    label=["Cached", "Uncached"],
    alpha=0.7,
    range=(0, 2000),
)
plt.xlabel("CTPS")
plt.ylabel("Count")
plt.legend()
plt.title("CTPS Distribution")

# Plot 4: Output tokens distribution
plt.subplot(2, 2, 4)
plt.hist(
    [df[df["is_cached"]]["output_tokens"], df[~df["is_cached"]]["output_tokens"]],
    bins=50,
    label=["Cached", "Uncached"],
    alpha=0.7,
    range=(0, 2000),
)
plt.xlabel("Output Tokens")
plt.ylabel("Count")
plt.legend()
plt.title("Output Token Distribution")

plt.tight_layout()
plt.show()
