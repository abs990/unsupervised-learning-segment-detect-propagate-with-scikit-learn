"""
Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - blobs_data
from sklearn.datasets import make_blobs

def blobs_data(random_state=42):
    # make_blobs with the five book centers and cluster_std [0.4, 0.3, 0.1, 0.1, 0.1]; return (X, y).
    return make_blobs(
        n_samples=2000, 
        centers=[[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]], 
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1], 
        random_state=random_state)

# Step 2 - fit_kmeans
from sklearn.cluster import KMeans

def fit_kmeans(X, k, random_state=42):
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)

# Step 3 - inertia_curve
def inertia_curve(X, ks):
    return {
        key : fit_kmeans(X, key).inertia_ for key in ks
    }

# Step 4 - silhouette_curve
from sklearn.metrics import silhouette_score

def silhouette_curve(X, ks):
    ks_filtered = [x for x in ks if x >= 2]
    return {
        k: float(silhouette_score(X, fit_kmeans(X, k).labels_)) for k in ks_filtered
    }

def best_k_by_silhouette(curve):
    # the k with the highest silhouette (smallest k on ties).
    return max(curve, key=lambda k: (curve[k], -k))

# Step 5 - fit_dbscan
from sklearn.cluster import DBSCAN

def fit_dbscan(X, eps=0.2, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)

def dbscan_summary(dbscan):
    unique_labels = list(dict.fromkeys(x for x in dbscan.labels_))
    return {
        'n_clusters': len([x for x in unique_labels if x != -1]),
        'n_noise': len([x for x in unique_labels if x == -1]),
        'n_core': len(dbscan.core_sample_indices_)}

# Step 6 - dbscan_predict
from sklearn.neighbors import KNeighborsClassifier

def dbscan_predict(dbscan, X_new, n_neighbors=50):
    # KNN on the core samples and their labels; predict labels for X_new.
    return KNeighborsClassifier(n_neighbors=n_neighbors).fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_]).predict(X_new)

# Step 7 - fit_gmm
from sklearn.mixture import GaussianMixture

def fit_gmm(X, n_components, random_state=42):
    return GaussianMixture(n_components, n_init=10, random_state=random_state).fit(X)

def bic_curve(X, ks):
    return {
        k:fit_gmm(X, k).bic(X) for k in ks
    }

# Step 8 - flag_anomalies
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples, percentile threshold, boolean mask of low-density points.
    densities = gmm.score_samples(X)
    return densities < np.percentile(densities, 100 * contamination)

# Step 9 - digits_data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def digits_data(test_size=0.25, random_state=42):
    # load_digits + stratified train_test_split; return (X_train, X_test, y_train, y_test).
    digits = load_digits()
    return train_test_split(
        digits.data, digits.target, 
        test_size=test_size, random_state=random_state, stratify=digits.target)

# Step 10 - baseline_50_random
from sklearn.linear_model import LogisticRegression

def baseline_50_random(X_train, y_train, X_test, y_test, n_labeled=50, random_state=42):
    # LogisticRegression(max_iter=10000) on the first n_labeled rows; return test accuracy.
    return LogisticRegression(max_iter=10000, random_state=random_state).fit(X_train[:n_labeled], y_train[:n_labeled]).score(X_test, y_test)

# Step 11 - representative_digits
def representative_digits(X_train, k=50, random_state=42):
    # fit k-means; rep_idx[j] = index of the row nearest centroid j; return (kmeans, rep_idx).
    model = fit_kmeans(X_train, k, random_state)
    return model, np.argmin(model.transform(X_train), axis=0)

# Step 12 - train_on_representatives
def train_on_representatives(X_train, y_train, rep_idx, X_test, y_test):
    # logistic regression on the representative rows only; return test accuracy.
    return LogisticRegression(max_iter=10000).fit(X_train[rep_idx], y_train[rep_idx]).score(X_test, y_test)

# Step 13 - propagate_and_train
def propagate_and_train(X_train, y_train, kmeans, rep_idx, X_test, y_test, percentile=20):
    # propagate each cluster's representative label to its closest percentile of points; train; report.
    n_samples, n_features = X_train.shape
    
    distances = kmeans.transform(X_train) # (n_samples, n_clusters)
    _, n_clusters = distances.shape

    clusters = kmeans.labels_

    own = distances[np.arange(n_samples), clusters]     # distance to "its" cluster
    cutoffs = np.array([np.percentile(own[clusters == c], percentile) for c in range(n_clusters)])

    mask = own <= cutoffs[clusters]
    prop_labels = y_train[rep_idx[clusters]]

    model = LogisticRegression(max_iter=10000).fit(X_train[mask, :], prop_labels[mask])
    y_pred = model.predict(X_test)

    return {
        "n_propagated": int(mask.sum()),
        "label_accuracy": float(np.sum(prop_labels[mask] == y_train[mask]) / len(y_train[mask])),
        "test_accuracy": model.score(X_test, y_test)
    }

# Step 14 - synthetic_image
def synthetic_image(size=48):
    # (size, size, 3) float image: red->yellow left half, blue->cyan right half, white top-left square.
    img = np.zeros((size, size, 3))
    half_w = size // 2

    # left half
    img[:, :half_w, 0] = 1
    img[:, :half_w, 1] = np.linspace(0, 1, half_w)
    img[:, :half_w, 2] = 0

    # right half
    img[:, half_w:, 0] = 0
    img[:, half_w:, 1] = np.linspace(0, 1, half_w)
    img[:, half_w:, 2] = 1

    # top left white square
    q_size = size // 4
    img[:q_size, :q_size, :] = [1, 1, 1]

    return img

# Step 15 - segment_colors (not yet solved)
# TODO: implement

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement

