import numpy as np
from sympy import Basic, Symbol

from Matrix import FloatMatrix, Matrix


def ScalarMultiply(
    scalar: int | float | Basic, matrix: Matrix | FloatMatrix | np.ndarray
) -> FloatMatrix:
    """
    Multiply a matrix by a scalar.

    Parameters
    ----------
    scalar : int | float | Basic
        The scalar value to multiply with the matrix.
    matrix : Matrix | FloatMatrix | np.ndarray
        The matrix to be multiplied by the scalar.

    Returns
    -------
    FloatMatrix
        The resulting matrix after scalar multiplication.

    Raises
    ------
    Exception
        If `scalar` is not of type int, float, or Basic.
    """
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
