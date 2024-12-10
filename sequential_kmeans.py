import argparse
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential K-Means with Fixed Random Seed"
    )
    parser.add_argument(
        "--n_points", type=int, default=1000000, help="Number of data points"
    )
    parser.add_argument("--k", type=int, default=100, help="Number of clusters")
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimensionality of data points"
    )
    parser.add_argument(
        "--max_iter", type=int, default=1000, help="Maximum number of iterations"
    )
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def initialize_centroids(data, k):
    """Randomly initialize centroids from the data points."""
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids


def assign_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    # Compute distances using broadcasting
    diff = data[:, None, :] - centroids[None, :, :]  # Shape: (n_points, k, dim)
    dists = np.linalg.norm(diff, axis=2)  # Shape: (n_points, k)
    cluster_assignments = np.argmin(dists, axis=1)
    return cluster_assignments


def update_centroids(data, assignments, k):
    """Update centroids based on current cluster assignments."""
    dim = data.shape[1]
    new_centroids = np.zeros((k, dim), dtype=np.float32)
    for c in range(k):
        points = data[assignments == c]
        if points.size > 0:
            new_centroids[c] = points.mean(axis=0)
        else:
            # Reinitialize centroid if no points are assigned
            new_centroids[c] = data[np.random.choice(data.shape[0])]
    return new_centroids


def kmeans(data, initial_centroids, max_iter, tol):
    """Run the K-Means algorithm until convergence or max_iter is reached."""
    centroids = initial_centroids
    for i in range(max_iter):
        # Assignment step
        assignments = assign_clusters(data, centroids)

        # Update step
        new_centroids = update_centroids(data, assignments, centroids.shape[0])

        # Check for convergence
        diff = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if diff < tol:
            print(f"Converged after {i+1} iterations with diff={diff:.6f}")
            break
    return centroids, assignments


def main():
    args = parse_args()

    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    # Generate synthetic data
    data = np.random.randn(args.n_points, args.dim).astype(np.float32)

    # Initialize centroids
    centroids = initialize_centroids(data, args.k)

    # Start timing
    start_time = time.time()

    # Run K-Means
    final_centroids, assignments = kmeans(data, centroids, args.max_iter, args.tol)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Calculate clustering metrics
    if args.k > 1 and args.n_points >= args.k:
        silhouette = silhouette_score(data, assignments)
        davies_bouldin = davies_bouldin_score(data, assignments)
    else:
        silhouette = -1  # Undefined
        davies_bouldin = -1  # Undefined

    # Print results
    print(f"Sequential Execution Time: {execution_time:.4f} seconds")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")


if __name__ == "__main__":
    main()
