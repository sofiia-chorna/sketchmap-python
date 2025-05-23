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
    distance_matrix = distance_matrix.double()

    # compute gramm matrix
    M = -0.5 * distance_matrix**2
    M -= M.mean(dim=0, keepdim=True)  # substract column means
    M -= M.mean(dim=1, keepdim=True)  # substract row means

    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_indices = sorted_indices[:n_components]

    # principal components
    top_eigenvalues = eigenvalues[top_indices]
    top_eigenvectors = eigenvectors[:, top_indices]

    coordinates = top_eigenvectors * torch.sqrt(top_eigenvalues.clamp(min=0))

    # align signs to match c++ convention : make max abs value positive
    for i in range(n_components):
        col = coordinates[:, i]
        max_id = torch.argmax(torch.abs(col))
        if col[max_id] < 0:
            coordinates[:, i] *= -1

    return coordinates
