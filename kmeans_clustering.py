import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load preprocessed data
X_scaled = np.load('X_scaled.npy')

# Set optimal number of clusters (k)
k = 3  # Change this value based on elbow method result
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

print('Inertia:', kmeans.inertia_)

# Visualize clusters (using first two principal components for 2D plot)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
for cluster in range(k):
    plt.scatter(X_pca[labels == cluster, 0], X_pca[labels == cluster, 1], label=f'Cluster {cluster+1}')
plt.title('K-means Clusters (PCA-reduced)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_clusters.png')
plt.show()
print('Cluster plot saved as kmeans_clusters.png')