import numpy as np

from Basis import Basis
from Matrix import FloatMatrix, Matrix
from MatrixAdd import MatrixAdd
from MatrixSubtract import MatrixSubtract
from ScalarMultiply import ScalarMultiply
from UnitVector import UnitVector
from VectorDot import VectorDot


def GramSchmidt(span: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix:
    """
    Computes an orthonormal Basis for the subspace represented by the input span, using the Gram-Schmidt process.

    Args:
        span (Matrix | FloatMatrix | np.ndarray): The input span vectors, represented as a Matrix, FloatMatrix, or numpy ndarray.

    Returns:
        FloatMatrix: An orthonormal Basis for the input subspace, represented as a FloatMatrix.

    Raises:
        ValueError: If the input type is not Matrix, FloatMatrix, or numpy ndarray.

    The function converts the input span into a FloatMatrix if necessary and then applies the Gram-Schmidt orthogonalization
    procedure to generate an orthonormal Basis. Each vector in the span is iteratively projected onto the subspace formed
    by the previously processed vectors, and the resulting orthogonal component is normalized to create the orthonormal Basis.
    """

    if isinstance(span, Matrix):

        span_ = FloatMatrix(span)

    if isinstance(span, np.ndarray):

        span_ = FloatMatrix(span)

    else:

        span_ = FloatMatrix(span)

    basis = Basis(span=span_)

    basisVectors = np.array([col for col in basis.T])

    orthogonalVectors = np.empty(shape=(basis.numCols, basis.numRows))

    orthogonalVectors[0] = UnitVector(vector=basisVectors[0]).flatten()

    for basisVectorNum in range(1, basis.numCols):

        componentAlreadyAccounted = FloatMatrix(basis.numRows, 1, 0, 0)

        basisVector = FloatMatrix(basisVectors[basisVectorNum])

        for orthogonalVectorNum in range(0, basisVectorNum):

            orthogonalVector = FloatMatrix(orthogonalVectors[orthogonalVectorNum])

            basisProjectedAmount = VectorDot(
                matrixA=basisVector, matrixB=orthogonalVector
            )

            basisProjectedInOrthogongalDirection = ScalarMultiply(
                scalar=basisProjectedAmount, matrix=orthogonalVector
            )

            componentAlreadyAccounted = MatrixAdd(
                matrixA=componentAlreadyAccounted,
                matrixB=basisProjectedInOrthogongalDirection,
            )

        orthogonalBasisComponent = MatrixSubtract(
            matrixA=basisVector, matrixB=componentAlreadyAccounted
        )

        unitOrthogonalBasisComponent = UnitVector(vector=orthogonalBasisComponent)

        orthogonalVectors[basisVectorNum] = unitOrthogonalBasisComponent.flatten()

    orthonormalBasis_ = FloatMatrix(orthogonalVectors.T)

    return orthonormalBasis_
