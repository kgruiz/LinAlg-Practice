from typing import Union

import numpy as np

from Matrix import Base2Matrix, FloatMatrix, Matrix
from MatrixAdd import MatrixAdd
from MatrixSubtract import MatrixSubtract
from ScalarMultiply import ScalarMultiply
from UnitVector import UnitVector
from VectorDot import VectorDot


def Determinat(matrix: Union[Base2Matrix, np.ndarray], iteration: int = 0) -> float:
    """
    Calculate the determinant of a square matrix.

    Parameters
    ----------
    matrix : Union[Base2Matrix, np.ndarray]
        The input square matrix.

    Returns
    -------
    float
        The determinant of the matrix.

    Raises
    ------
    Exception
        If the matrix is not square.
    """

    if isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)

    if isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)

    else:

        matrix_ = FloatMatrix(matrix)

    if matrix_.numRows != matrix_.numCols:

        raise Exception(
            f"Cannot compute the Determinat of a non-square matrix.\n\
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
                * (Determinat(matrix=subMatrix, iteration=(iteration + 1)))
            )

        return determinant
