import numpy as np

from Matrix import FloatMatrix, Matrix


def Transpose(matrix: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix:
    """
    Return the transpose of a matrix or vector.

    Parameters
    ----------
    matrix : Matrix | FloatMatrix | np.ndarray
        The matrix or vector to transpose.

    Returns
    -------
    FloatMatrix
        The transposed matrix or vector.
    """
    if isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)

    elif isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)

    else:

        matrix_ = FloatMatrix(matrix)

    return FloatMatrix(matrix_.matrix.T)
