import numpy as np

from matrix import FloatMatrix, Matrix
from matrixAdd import MatrixAdd
from matrixSubtract import MatrixSubtract
from scalarMultiply import ScalarMultiply
from unitVector import UnitVector
from vectorDot import VectorDot


def Determinat(matrix: Matrix | FloatMatrix | np.ndarray, iter=0) -> int | float:
    """
    Computes the determinat of a matrix
    """

    if isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)

    if isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)

    else:

        matrix_ = FloatMatrix(matrix)

    if matrix_.numRows != matrix_.numCols:

        raise Exception(
            f"Cannot compute the determinat of a non-square matrix.\n\
            Input matrix has shape: {matrix_.numRows} x {matrix_.numCols}"
        )

    if matrix_.numRows == 0:

        return 1

    if matrix_.numRows == 1:

        return matrix_[0][0]

    elif matrix_.numRows == 2:

        return (matrix_[0][0] * matrix_[1][1]) - (matrix_[1][0] * matrix_[0][1])

    else:

        determinant = 0

        for excludeCol in range(matrix_.numCols):

            if excludeCol % 2 == 0:

                alternatingSign = 1

            else:

                alternatingSign = -1

            subMatrix = np.delete(matrix_[1:], excludeCol, axis=1)

            determinant += (
                alternatingSign
                * matrix_[0][excludeCol]
                * (Determinat(matrix=subMatrix, iter=(iter + 1)))
            )

        return determinant
