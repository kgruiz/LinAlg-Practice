import numpy as np

from Matrix import FloatMatrix, Matrix
from VectorLength import Length


def UnitVector(vector: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix:
    """
    Compute the unit vector of a given vector.

    Parameters
    ----------
    vector : Matrix | FloatMatrix | np.ndarray
        The input vector to be normalized.

    Returns
    -------
    FloatMatrix
        The unit vector in the direction of `vector`.

    Raises
    ------
    Exception
        If the input is not a vector.
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

    vectorLength = Length(vector=vector_)

    originalShape = vector.shape

    vector_ = vector_.flatten()

    if vectorLength != 0:

        for elemNum, elem in enumerate(vector_):

            vector_[elemNum] = elem / vectorLength

    vector_ = np.reshape(vector_, newshape=originalShape)

    return FloatMatrix(vector_)
