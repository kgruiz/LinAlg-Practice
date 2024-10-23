import numpy as np

from matrix import FloatMatrix, Matrix
from matrixAdd import MatrixAdd
from matrixSubtract import MatrixSubtract
from scalarMultiply import ScalarMultiply
from unitVector import UnitVector
from vectorDot import VectorDot


def GramSchmidt(basis: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix:
    """
    Computes an orthonormal basis for the subspace represented by the input basis, using the Gram-Schmidt process.

    Args:
        basis (Matrix | FloatMatrix | np.ndarray): The input basis vectors, represented as a Matrix, FloatMatrix, or numpy ndarray.

    Returns:
        FloatMatrix: An orthonormal basis for the input subspace, represented as a FloatMatrix.

    Raises:
        ValueError: If the input type is not Matrix, FloatMatrix, or numpy ndarray.

    The function converts the input basis into a FloatMatrix if necessary and then applies the Gram-Schmidt orthogonalization
    procedure to generate an orthonormal basis. Each vector in the basis is iteratively projected onto the subspace formed
    by the previously processed vectors, and the resulting orthogonal component is normalized to create the orthonormal basis.
    """

    if isinstance(basis, Matrix):

        basis_ = FloatMatrix(basis)

    if isinstance(basis, np.ndarray):

        basis_ = FloatMatrix(basis)

    else:

        basis_ = FloatMatrix(basis)

    basisVectors = np.array([col for col in basis_.T])

    orthogonalVectors = np.empty(shape=(basis_.numCols, basis_.numRows))

    orthogonalVectors[0] = UnitVector(vector=basisVectors[0]).flatten()

    for basis_VectorNum in range(1, basis_.numCols):

        componentAlreadyAccounted = FloatMatrix(basis_.numRows, 1, 0, 0)

        basis_Vector = FloatMatrix(basisVectors[basis_VectorNum])

        for orthogonalVectorNum in range(0, basis_VectorNum):

            orthogonalVector = FloatMatrix(orthogonalVectors[orthogonalVectorNum])

            basis_ProjectedAmount = VectorDot(
                matrixA=basis_Vector, matrixB=orthogonalVector
            )

            basis_ProjectedInOrthogongalDireciton = ScalarMultiply(
                scalar=basis_ProjectedAmount, matrix=orthogonalVector
            )

            componentAlreadyAccounted = MatrixAdd(
                matrixA=componentAlreadyAccounted,
                matrixB=basis_ProjectedInOrthogongalDireciton,
            )

        orthogonalbasis_Component = MatrixSubtract(
            matrixA=basis_Vector, matrixB=componentAlreadyAccounted
        )

        unitOrthogonalbasis_Component = UnitVector(vector=orthogonalbasis_Component)

        orthogonalVectors[basis_VectorNum] = unitOrthogonalbasis_Component.flatten()

    orthonormalbasis_ = FloatMatrix(orthogonalVectors.T)

    return orthonormalbasis_
