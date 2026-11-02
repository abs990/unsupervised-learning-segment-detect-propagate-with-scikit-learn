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

# Step 6 - dbscan_predict (not yet solved)
# TODO: implement

# Step 7 - fit_gmm (not yet solved)
# TODO: implement

# Step 8 - flag_anomalies (not yet solved)
# TODO: implement

# Step 9 - digits_data (not yet solved)
# TODO: implement

# Step 10 - baseline_50_random (not yet solved)
# TODO: implement

# Step 11 - representative_digits (not yet solved)
# TODO: implement

# Step 12 - train_on_representatives (not yet solved)
# TODO: implement

# Step 13 - propagate_and_train (not yet solved)
# TODO: implement

# Step 14 - synthetic_image (not yet solved)
# TODO: implement

# Step 15 - segment_colors (not yet solved)
# TODO: implement

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement

