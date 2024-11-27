from typing import Union

import numpy as np

from Matrix import FloatMatrix, Matrix
from VectorDot import VectorDot


defMatrixMultiply(
    matrixA: Union[Matrix, FloatMatrix, np.ndarray],
    matrixB: Union[Matrix, FloatMatrix, np.ndarray],
) -> FloatMatrix:
    """
    Multiply two matrices.

    Parameters
    ----------
    matrixA : Union[Matrix, FloatMatrix, np.ndarray]
        The first matrix.
    matrixB : Union[Matrix, FloatMatrix, np.ndarray]
        The second matrix.

    Returns
    -------
    FloatMatrix
        The product of `matrixA` and `matrixB`.

    Raises
    ------
    Exception
        If the number of columns in `matrixA` does not match the number of rows in `matrixB`.
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

    if matrixA_.numCols != matrixB_.numRows:

        raise Exception(
            f"Invalid shape for input matrices. Matrix A Columns: \
                {matrixA_.numCols} != Matrix B Rows: {matrixB_.numRows}"
        )

    resultArray = np.dot(matrixA_.matrix, matrixB_.matrix).astype(object)

    return FloatMatrix(resultArray)
