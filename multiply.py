import numpy as np

from Matrix import FloatMatrix, Matrix
from VectorDot import VectorDot


def Multiply(
    matrixA: Matrix | FloatMatrix | np.ndarray,
    matrixB: Matrix | FloatMatrix | np.ndarray,
) -> FloatMatrix:

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

    if matrixA_.numCols != matrixB_.numRows:

        raise Exception(
            f"Invalid shape for input matrices. Matrix A Columns: \
                {matrixA_.numCols} != Matrix B Rows: {matrixB_.numRows}"
        )

    resultMatrix = FloatMatrix(
        matrixA_.numRows,
        matrixB_.numCols,
    )

    for row in range(resultMatrix.numRows):

        for col in range(resultMatrix.numCols):

            newElem = VectorDot(
                matrixA=FloatMatrix(matrixA_[row, :]),
                matrixB=FloatMatrix(matrixB_[:, col]),
            )

            resultMatrix[row][col] = round(newElem, 12)

    return resultMatrix
