import numpy as np
from sympy import Basic, Symbol

from Matrix import FloatMatrix, Matrix


def ScalarMultiply(
    scalar: int | float | Basic, matrix: Matrix | FloatMatrix | np.ndarray
) -> FloatMatrix:

    if isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix.astype(object))

    elif isinstance(matrix, (Matrix, FloatMatrix)):

        matrix_ = FloatMatrix(matrix.matrix.astype(object))

    else:

        matrix_ = FloatMatrix(matrix)

    if (
        not isinstance(scalar, int)
        and not isinstance(scalar, float)
        and not isinstance(scalar, Basic)
    ):

        raise Exception(f"Invalid scalar type: {type(scalar)}")

    for row in range(matrix_.numRows):

        for col in range(matrix_.numCols):

            matrix_[row][col] = matrix_[row][col] * scalar

    return matrix_
