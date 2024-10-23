import numpy as np

from matrix import FloatMatrix, Matrix


def ScalarMultiply(
    scalar: int | float, matrix: Matrix | FloatMatrix | np.ndarray
) -> FloatMatrix:

    if isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)

    elif isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)

    else:

        matrix_ = FloatMatrix(matrix)

    if not isinstance(scalar, int) and not isinstance(scalar, float):

        raise Exception(f"Invalid scalar type: {type(scalar)}")

    for row in range(matrix_.numRows):

        for col in range(matrix_.numCols):

            matrix_[row][col] = matrix_[row][col] * scalar

    return matrix_
