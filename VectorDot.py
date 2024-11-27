import numpy as np
from sympy import Symbol

from Matrix import FloatMatrix, Matrix


def VectorDot(
    matrixA: Matrix | FloatMatrix | np.ndarray,
    matrixB: Matrix | FloatMatrix | np.ndarray,
) -> Symbol:
    """
    Compute the dot product of two vectors.

    Parameters
    ----------
    matrixA : Matrix | FloatMatrix | np.ndarray
        The first vector.
    matrixB : Matrix | FloatMatrix | np.ndarray
        The second vector.

    Returns
    -------
    Symbol
        The dot product of `matrixA` and `matrixB`.

    Raises
    ------
    Exception
        If either input is not a vector or if the vectors have different lengths.
    """
    if isinstance(matrixA, np.ndarray):

        matrixA_ = FloatMatrix(matrixA)

    elif isinstance(matrixA, Matrix):

        matrixA_ = FloatMatrix(matrixA)

    else:

        matrixA_ = FloatMatrix(matrixA)

    if isinstance(matrixB, np.ndarray):

        matrixB_ = FloatMatrix(matrixB)

    elif isinstance(matrixB, Matrix):

        matrixB_ = FloatMatrix(matrixB)

    else:

        matrixB_ = FloatMatrix(matrixB)

    if not (
        (matrixA_.numCols == 1 or matrixA_.numRows == 1)
        and (matrixB_.numCols == 1 or matrixB_.numRows == 1)
    ):

        raise Exception(
            f"Cannot perform dot product. One of the input matrices is not a vector.\n\
                Shape A: {matrixA_.numRows} x {matrixA_.numCols}\n\
                    Shape B: {matrixB.numRows} x {matrixB.numCols}"
        )

    LengthA = max(matrixA_.numRows, matrixA_.numCols)
    LengthB = max(matrixB_.numRows, matrixB_.numCols)

    if LengthA != LengthB:

        raise Exception(
            f"Cannot perform dot product.\nMatrices A and B have different Lengths.\n\
                Length A: {LengthA}\nLength B: {LengthB}"
        )

    if matrixA_.shape[0] != 1:

        matrixA_ = matrixA_.T

    if matrixB_.shape[0] != 1:

        matrixB_ = matrixB_.T

    dotProduct = sum(a * b for a, b in zip(matrixA_[0], matrixB_[0]))

    return dotProduct
