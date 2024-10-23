from math import sqrt

import numpy as np

from matrix import FloatMatrix, Matrix


def Length(vector: Matrix | FloatMatrix | np.ndarray) -> float:
    """Computes the length of a given vector"""

    if isinstance(vector, Matrix):

        vector_ = FloatMatrix(vector)

    elif isinstance(vector, np.ndarray):

        vector_ = FloatMatrix(vector)

    else:

        vector_ = FloatMatrix(vector)

    if vector_.numRows != 1 and vector_.numCols != 1:

        raise Exception(
            f"Input vector is not a vector. Shape: {vector_.numRows} x {vector_.numCols}"
        )

    if vector_.numRows == 1:

        vector_.matrix = vector_.matrix.T

    squaredSum = 0

    for elem in vector_.flatten():

        squaredSum += elem * elem

    vectorLength = sqrt(squaredSum)

    return vectorLength
