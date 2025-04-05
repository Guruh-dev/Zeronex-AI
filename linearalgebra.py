import numpy as np
from typing import Union, Tuple, List

class LinearAlgebra:
    def __init__(self):
        """Initialize the LinearAlgebra class for Zeronex AI"""
        pass

    @staticmethod
    def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication"""
        return np.dot(A, B)

    @staticmethod
    def transpose(matrix: np.ndarray) -> np.ndarray:
        """Compute matrix transpose"""
        return np.transpose(matrix)

    @staticmethod
    def inverse(matrix: np.ndarray) -> Union[np.ndarray, None]:
        """Compute matrix inverse if possible"""
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def eigen_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors

    @staticmethod
    def singular_value_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute SVD decomposition"""
        U, S, V = np.linalg.svd(matrix)
        return U, S, V

    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b"""
        return np.linalg.solve(A, b)

    @staticmethod
    def matrix_rank(matrix: np.ndarray) -> int:
        """Compute matrix rank"""
        return np.linalg.matrix_rank(matrix)

    @staticmethod
    def determinant(matrix: np.ndarray) -> float:
        """Compute matrix determinant"""
        return np.linalg.det(matrix)

    @staticmethod
    def orthogonalize(vectors: np.ndarray) -> np.ndarray:
        """Perform Gram-Schmidt orthogonalization"""
        Q, R = np.linalg.qr(vectors)
        return Q

    @staticmethod
    def project_vector(v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Project vector v onto vector u"""
        return (np.dot(v, u) / np.dot(u, u)) * u

    @staticmethod
    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix by its Frobenius norm"""
        return matrix / np.linalg.norm(matrix)

    @staticmethod
    def is_positive_definite(matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite"""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

# Example usage
if __name__ == "__main__":
    la = LinearAlgebra()
    
    # Create sample matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # Test matrix operations
    print("Matrix multiplication:")
    print(la.matrix_multiplication(A, B))
    
    print("\nMatrix transpose:")
    print(la.transpose(A))
    
    print("\nMatrix inverse:")
    print(la.inverse(A))
    
    print("\nEigenvalues and eigenvectors:")
    eigenvals, eigenvecs = la.eigen_decomposition(A)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    if __name__ == "__main__":
        'main'()
