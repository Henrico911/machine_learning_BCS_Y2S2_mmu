import numpy as np

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.linkage_matrix = []

    def fit(self, X):
        n_samples = X.shape[0]
        distances = self.compute_pairwise_distances(X)
        clusters = [{'id': i, 'size': 1, 'points': {i}} for i in range(n_samples)]
        next_id = n_samples

        while len(clusters) > 1:
            min_dist = np.inf
            merge_indices = (-1, -1)

            # Find the closest clusters
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    dist = self.compute_cluster_distance(
                        clusters[i]['points'], clusters[j]['points'], distances, self.linkage
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_indices = (i, j)

            i, j = merge_indices
            if i > j:
                i, j = j, i
            c1, c2 = clusters[i], clusters[j]

            # Record the merge
            new_id = next_id
            new_points = c1['points'].union(c2['points'])
            new_size = c1['size'] + c2['size']
            self.linkage_matrix.append([c1['id'], c2['id'], min_dist, new_size])

            # Update clusters
            clusters.pop(j)
            clusters.pop(i)
            clusters.append({'id': new_id, 'size': new_size, 'points': new_points})
            next_id += 1

            # Early stopping if reached desired clusters
            if len(clusters) == self.n_clusters:
                break

        # Assign labels
        self.labels_ = self.get_labels_from_linkage(n_samples)
        return self

    def compute_pairwise_distances(self, X):
        n = X.shape[0]
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i][j] = np.sqrt(np.sum((X[i] - X[j])**2))
        return dist

    def compute_cluster_distance(self, points1, points2, dist_matrix, linkage):
        if linkage == 'single':
            min_d = np.inf
            for i in points1:
                for j in points2:
                    if dist_matrix[i][j] < min_d:
                        min_d = dist_matrix[i][j]
            return min_d
        elif linkage == 'complete':
            max_d = -np.inf
            for i in points1:
                for j in points2:
                    if dist_matrix[i][j] > max_d:
                        max_d = dist_matrix[i][j]
            return max_d
        elif linkage == 'average':
            total, count = 0.0, 0
            for i in points1:
                for j in points2:
                    total += dist_matrix[i][j]
                    count += 1
            return total / count if count > 0 else 0.0
        else:
            raise ValueError(f"Unsupported linkage: {linkage}")

    def get_labels_from_linkage(self, n_samples):
        if self.n_clusters < 1 or self.n_clusters > n_samples:
            raise ValueError("Invalid number of clusters")

        # Initialize parent array for union-find
        parent = list(range(n_samples + len(self.linkage_matrix)))
        for i, row in enumerate(self.linkage_matrix):
            new_id = n_samples + i
            a, b = int(row[0]), int(row[1])
            parent[a] = new_id
            parent[b] = new_id

        # Determine merges needed for desired clusters
        n_merges = n_samples - self.n_clusters
        if n_merges > len(self.linkage_matrix):
            n_merges = len(self.linkage_matrix)

        # Track active roots
        active = set()
        for idx in range(n_samples + n_merges):
            root = idx
            while parent[root] != root:
                root = parent[root]
            if idx < n_samples:
                active.add(root)

        # Assign labels
        label_map = {}
        label = 0
        labels = np.zeros(n_samples, dtype=int)
        for point in range(n_samples):
            root = point
            while parent[root] != root:
                root = parent[root]
            if root not in label_map:
                label_map[root] = label
                label += 1
            labels[point] = label_map[root]
        return labels

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data = load_iris()
    X = data.data

    model = AgglomerativeClustering(n_clusters=3, linkage='average')
    model.fit(X)
    print("Cluster labels:", model.labels_)