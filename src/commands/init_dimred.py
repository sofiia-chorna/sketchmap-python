import torch

from sklearn.decomposition import PCA

from src.utils.tensor import to_tensor


def run_pca(data_points: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    data_points_np = data_points.cpu().numpy()

    pca = PCA(n_components=n_components)
    pca_embedded = pca.fit_transform(data_points_np)

    return to_tensor(pca_embedded)


def run_mds(distance_matrix: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """
    Classical multidimensional scaling (MDS) on a distance matrix.
    Returns low-dimensional embedding (shape [n, low_dim])
    """
    distance_matrix = distance_matrix.cpu()

    num_points = distance_matrix.shape[0]

    # centering matrix: subtracts the mean from each row/column
    identity = torch.eye(num_points)
    ones = torch.ones((num_points, num_points)) / num_points
    centering_matrix = identity - ones

    # double-centering the squared distance matrix
    squared_distances = distance_matrix**2
    gram_matrix = -0.5 * centering_matrix @ squared_distances @ centering_matrix

    eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)

    sorted_ids = torch.argsort(eigenvalues, descending=True)
    top_ids = sorted_ids[:n_components]

    # principal components
    top_eigenvalues = eigenvalues[top_ids]
    top_eigenvectors = eigenvectors[:, top_ids]

    coordinates = top_eigenvectors * torch.sqrt(top_eigenvalues.clamp(min=0))

    return coordinates
