from typing import Union

import numpy as np
from sympy import Mod

from Base2VectorDot import Base2VectorDot
from Matrix import Base2Matrix, EnsureNoTwo


def Base2Multiply(
    matrixA: Union[Base2Matrix, np.ndarray],
    matrixB: Union[Base2Matrix, np.ndarray],
) -> Base2Matrix:
    """
    Multiply two matrices in base 2.

    Parameters
    ----------
    matrixA : Union[Base2Matrix, np.ndarray]
        The first matrix.
    matrixB : Union[Base2Matrix, np.ndarray]
        The second matrix.

    Returns
    -------
    Base2Matrix
        The result of multiplying the two matrices.

    Raises
    ------
    Exception
        If the number of columns in matrixA does not equal the number of rows in matrixB.
    """
    if isinstance(matrixA, np.ndarray):

        matrixA_ = Base2Matrix(matrixA)

    elif isinstance(matrixA, Base2Matrix):

        matrixA_ = Base2Matrix(matrixA)

    if isinstance(matrixB, np.ndarray):

        matrixB_ = Base2Matrix(matrixB)

    elif isinstance(matrixB, Base2Matrix):

        matrixB_ = Base2Matrix(matrixB)

    if matrixA_.numCols != matrixB_.numRows:

        raise Exception(
            f"Invalid shape for input matrices.\n\
                Matrix A Columns: {matrixA_.numCols} != Matrix B Rows: {matrixB_.numRows}"
        )

    EnsureNoTwo(matrix=matrixA_)
    EnsureNoTwo(matrix=matrixB_)

    resultMatrix = Base2Matrix(
        matrixA_.numRows,
        matrixB_.numCols,
    )

    EnsureNoTwo(matrix=matrixA_)
    EnsureNoTwo(matrix=matrixB_)
    EnsureNoTwo(matrix=resultMatrix)

    for row in range(resultMatrix.numRows):

        for col in range(resultMatrix.numCols):

            newElem = Base2VectorDot(
                matrixA=Base2Matrix(matrixA_[row, :]),
                matrixB=Base2Matrix(matrixB_[:, col]),
            )

            resultMatrix[row][col] = Mod(newElem, 2)

            EnsureNoTwo(matrix=matrixA_)
            EnsureNoTwo(matrix=matrixB_)
            EnsureNoTwo(matrix=resultMatrix)

    EnsureNoTwo(matrix=matrixA_)
    EnsureNoTwo(matrix=matrixB_)
    EnsureNoTwo(matrix=resultMatrix)

    return resultMatrix
