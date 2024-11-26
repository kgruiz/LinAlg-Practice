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

    Basis = Basis(span=span_)

    BasisVectors = np.array([col for col in Basis.T])

    orthogonalVectors = np.empty(shape=(Basis.numCols, Basis.numRows))

    orthogonalVectors[0] = UnitVector(vector=BasisVectors[0]).flatten()

    for Basis_VectorNum in range(1, Basis.numCols):

        componentAlreadyAccounted = FloatMatrix(Basis.numRows, 1, 0, 0)

        Basis_Vector = FloatMatrix(BasisVectors[Basis_VectorNum])

        for orthogonalVectorNum in range(0, Basis_VectorNum):

            orthogonalVector = FloatMatrix(orthogonalVectors[orthogonalVectorNum])

            Basis_ProjectedAmount = VectorDot(
                matrixA=Basis_Vector, matrixB=orthogonalVector
            )

            Basis_ProjectedInOrthogongalDireciton = ScalarMultiply(
                scalar=Basis_ProjectedAmount, matrix=orthogonalVector
            )

            componentAlreadyAccounted = MatrixAdd(
                matrixA=componentAlreadyAccounted,
                matrixB=Basis_ProjectedInOrthogongalDireciton,
            )

        orthogonalBasis_Component = MatrixSubtract(
            matrixA=Basis_Vector, matrixB=componentAlreadyAccounted
        )

        unitOrthogonalBasis_Component = UnitVector(vector=orthogonalBasis_Component)

        orthogonalVectors[Basis_VectorNum] = unitOrthogonalBasis_Component.flatten()

    orthonormalBasis_ = FloatMatrix(orthogonalVectors.T)

    return orthonormalBasis_
