import numpy as np

from matrix import FloatMatrix, Matrix


def VectorDot(
    matrixA: Matrix | FloatMatrix | np.ndarray,
    matrixB: Matrix | FloatMatrix | np.ndarray,
) -> int:

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

    lengthA = max(matrixA_.numRows, matrixA_.numCols)
    lengthB = max(matrixB_.numRows, matrixB_.numCols)

    if lengthA != lengthB:

        raise Exception(
            f"Cannot perform dot product.\nMatrices A and B have different lengths.\n\
                Length A: {lengthA}\nLength B: {lengthB}"
        )

    if matrixA_.shape[0] != 1:

        matrixA_ = matrixA_.T

    if matrixB_.shape[0] != 1:

        matrixB_ = matrixB_.T

    dotProduct = 0

    for elemA, elemB in zip(matrixA_[0], matrixB_[0]):

        dotProduct += elemA * elemB

    return dotProduct
