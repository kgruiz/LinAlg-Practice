from math import sqrt

import numpy as np

from Matrix import FloatMatrix, Matrix


def Length(vector: Matrix | FloatMatrix | np.ndarray) -> float:
    """
    Calculate the Euclidean length of a vector.

    Parameters
    ----------
    vector : Matrix | FloatMatrix | np.ndarray
        The input vector whose length is to be calculated.

    Returns
    -------
    float
        The Euclidean length of the input vector.
    """
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
