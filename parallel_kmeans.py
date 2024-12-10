import argparse
import numpy as np
from mpi4py import MPI
import time
from sklearn.metrics import silhouette_score, davies_bouldin_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel K-Means with MPI and Fixed Random Seed"
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


def distribute_data(n_points, dim, comm, seed):
    """Generate and distribute the dataset among MPI ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        np.random.seed(seed)  # Set seed on root for reproducibility
        data = np.random.randn(n_points, dim).astype(np.float32)
        chunks = np.array_split(data, size, axis=0)
    else:
        chunks = None

    local_data = comm.scatter(chunks, root=0)
    return local_data


def initialize_centroids(local_data, k, comm):
    """Initialize and broadcast centroids from rank 0 to all processes."""
    rank = comm.Get_rank()
    # Gather all local_data at rank 0 to choose initial centroids
    all_data = comm.gather(local_data, root=0)

    if rank == 0:
        all_data = np.vstack(all_data)
        centroids = all_data[np.random.choice(all_data.shape[0], k, replace=False)]
    else:
        centroids = None

    centroids = comm.bcast(centroids, root=0)
    return centroids


def compute_local_sums_counts(local_data, cluster_assignments, k):
    """Compute local partial sums and counts of each cluster."""
    dim = local_data.shape[1]
    new_sum = np.zeros((k, dim), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for c in range(k):
        points_c = local_data[cluster_assignments == c]
        if points_c.size > 0:
            new_sum[c] = np.sum(points_c, axis=0)
            counts[c] = points_c.shape[0]
    return new_sum, counts


def reduce_sums_counts(new_sum, counts, comm):
    """Reduce local sums and counts across all MPI ranks."""
    global_sum = np.zeros_like(new_sum)
    global_count = np.zeros_like(counts)
    comm.Allreduce(new_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(counts, global_count, op=MPI.SUM)
    return global_sum, global_count


def update_global_centroids(global_sum, global_count, old_centroids):
    """Compute updated global centroids from reduced sums and counts."""
    k, dim = old_centroids.shape
    new_centroids = old_centroids.copy()
    valid_clusters = global_count > 0
    new_centroids[valid_clusters] = (
        global_sum[valid_clusters] / global_count[valid_clusters, None]
    ).astype(np.float32)
    return new_centroids


def assign_clusters(local_data, centroids):
    """
    Assign each data point in local_data to the nearest centroid.
    This is a CPU-only operation using NumPy.
    """
    # Compute distances using broadcasting
    diff = local_data[:, None, :] - centroids[None, :, :]  # Shape: (local_n, k, dim)
    dists = np.linalg.norm(diff, axis=2)  # Shape: (local_n, k)
    cluster_assignments = np.argmin(dists, axis=1)
    return cluster_assignments


def kmeans(local_data, initial_centroids, max_iter, tol, comm):
    """Run the K-Means loop until convergence or max_iter is reached."""
    rank = comm.Get_rank()
    centroids = initial_centroids

    for i in range(max_iter):
        # Assignment step
        cluster_assignments = assign_clusters(local_data, centroids)

        # Update step
        new_sum, counts = compute_local_sums_counts(
            local_data, cluster_assignments, centroids.shape[0]
        )
        global_sum, global_count = reduce_sums_counts(new_sum, counts, comm)

        new_centroids = update_global_centroids(global_sum, global_count, centroids)

        # Check for convergence
        diff = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        centroids = comm.bcast(centroids, root=0)

        if diff < tol:
            if rank == 0:
                print(f"Converged after {i+1} iterations with diff={diff:.6f}")
            break
    return centroids, cluster_assignments


def gather_assignments(local_assignments, comm):
    """Gather all cluster assignments to the root process."""
    all_assignments = comm.gather(local_assignments, root=0)
    if comm.Get_rank() == 0:
        all_assignments = np.hstack(all_assignments)
    return all_assignments


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Start total timing
    if rank == 0:
        total_start_time = time.time()

    # Distribute data among ranks with fixed seed
    local_data = distribute_data(args.n_points, args.dim, comm, args.seed)

    # Initialize centroids
    centroids = initialize_centroids(local_data, args.k, comm)

    # Run K-Means
    final_centroids, local_assignments = kmeans(
        local_data, centroids, args.max_iter, args.tol, comm
    )

    # Gather all assignments at root for metric computation
    all_assignments = gather_assignments(local_assignments, comm)

    # End total timing
    if rank == 0:
        total_end_time = time.time()
        execution_time = total_end_time - total_start_time

        # Gather all data at root
        all_data = comm.gather(local_data, root=0)
        all_data = np.vstack(all_data)

        # Compute clustering metrics
        if args.k > 1 and args.n_points >= args.k:
            silhouette = silhouette_score(all_data, all_assignments)
            davies_bouldin = davies_bouldin_score(all_data, all_assignments)
        else:
            silhouette = -1  # Undefined
            davies_bouldin = -1  # Undefined

        # Print results
        print(f"Parallel Execution Time: {execution_time:.4f} seconds")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

        # Optionally, print final centroids
        # print("Final centroids:")
        # print(final_centroids)


if __name__ == "__main__":
    main()
