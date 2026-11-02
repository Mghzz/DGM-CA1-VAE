import numpy as np
 
rng = np.random.default_rng(2026)
n_clusters = 8
n_per_cluster = 1200
angles = 2 * np.pi * np.arange(n_clusters) / n_clusters
centers = 4 * np.column_stack([np.cos(angles), np.sin(angles)])
 
s = np.vstack([
    centers[k] + 0.35 * rng.standard_normal((n_per_cluster, 2))
    for k in range(n_clusters)
]).astype(np.float32)
labels = np.repeat(np.arange(n_clusters), n_per_cluster)
 
W = rng.normal(0, 0.7, size=(2, 12)).astype(np.float32)
b = rng.uniform(0, 2 * np.pi, size=12).astype(np.float32)
s1, s2 = s[:, 0], s[:, 1]
basic = np.column_stack([
    s1 / 4, s2 / 4,
    np.sin(s1), np.cos(s1),
    np.sin(s2), np.cos(s2),
    s1 * s2 / 8,
    (s1**2 - s2**2) / 16
])
 
x = np.column_stack([basic, np.sin(s @ W + b)])
x = (x + 0.07 * rng.standard_normal(x.shape)).astype(np.float32)
 
order = rng.permutation(len(x))
x, labels = x[order], labels[order]
 
n_train = int(0.70 * len(x))
n_val = int(0.15 * len(x))
x_train = x[:n_train]
x_val = x[n_train:n_train + n_val]
x_test = x[n_train + n_val:]
y_train = labels[:n_train]
y_val = labels[n_train:n_train + n_val]
y_test = labels[n_train + n_val:]
mean = x_train.mean(axis=0)
std = x_train.std(axis=0).clip(min=1e-6)
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std
print(x_train.shape, x_val.shape, x_test.shape)
