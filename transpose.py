import numpy as np

from matrix import FloatMatrix, Matrix


def Transpose(matrix: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix:
    """Returns the transpose of a matrix or vector"""

    if isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)

    elif isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)

    else:

        matrix_ = FloatMatrix(matrix)

    return FloatMatrix(matrix_.matrix.T)
