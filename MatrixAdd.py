from typing import Union

import numpy as np

from Matrix import FloatMatrix, Matrix


def MatrixAdd(
    matrixA: Union[Matrix, FloatMatrix, np.ndarray],
    matrixB: Union[Matrix, FloatMatrix, np.ndarray],
) -> FloatMatrix:
    """
    Add two matrices.

    Parameters
    ----------
    matrixA : Union[Matrix, FloatMatrix, np.ndarray]
        The first matrix.
    matrixB : Union[Matrix, FloatMatrix, np.ndarray]
        The second matrix.

    Returns
    -------
    FloatMatrix
        The result of matrix addition.

    Raises
    ------
    Exception
        If the shapes of `matrixA` and `matrixB` are not identical.
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

    if matrixA_.shape != matrixB_.shape:

        raise Exception(
            f"Shapes of input matrices are not identical.\n\
            Shape A: {matrixA_.numRows} x {matrixA_.numCols}\n\
                Shape B: {matrixB_.numRows} x {matrixB_.numCols}"
        )

    resultMatrix = FloatMatrix(matrixA_.numRows, matrixA_.numCols)

    for row in range(matrixA_.numRows):

        for col in range(matrixA_.numCols):

            resultMatrix[row][col] = matrixA_[row][col] + matrixB_[row][col]

    return resultMatrix
