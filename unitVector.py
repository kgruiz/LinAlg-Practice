import numpy as np

from matrix import FloatMatrix, Matrix
from vectorLength import Length


def UnitVector(vector: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix:
    """Computes the unit vector for a given vector"""

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

    vectorLength = Length(vector=vector_)

    originalShape = vector_.shape

    vector_ = vector_.flatten()

    if vectorLength != 0:

        for elemNum, elem in enumerate(vector_):

            vector_[elemNum] = elem / vectorLength

    np.reshape(vector_, newshape=originalShape)

    return vector_
