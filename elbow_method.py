import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load preprocessed data
X_scaled = np.load('X_scaled.npy')

# Elbow method to find optimal k
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('elbow_method.png')
plt.show()
print('Elbow method graph saved as elbow_method.png')