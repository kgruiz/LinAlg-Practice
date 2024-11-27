from typing import Union

import numpy as np
from sympy import Mod, Symbol

from Matrix import Base2Matrix, EnsureNoTwo


def Base2VectorDot(
    matrixA: Union[Base2Matrix, np.ndarray],
    matrixB: Union[Base2Matrix, np.ndarray],
) -> Symbol:
    """
    Compute the dot product of two vectors in base 2.

    Parameters
    ----------
    matrixA : Union[Base2Matrix, np.ndarray]
        The first vector.
    matrixB : Union[Base2Matrix, np.ndarray]
        The second vector.

    Returns
    -------
    Symbol
        The result of the dot product modulo 2.

    Raises
    ------
    Exception
        If input matrices are not vectors or have different lengths.
    """
    base = 2

    if isinstance(matrixA, np.ndarray):

        matrixA_ = Base2Matrix(matrixA)

    elif isinstance(matrixA, Base2Matrix):

        matrixA_ = Base2Matrix(matrixA)

    if isinstance(matrixB, np.ndarray):

        matrixB_ = Base2Matrix(matrixB)

    elif isinstance(matrixB, Base2Matrix):

        matrixB_ = Base2Matrix(matrixB)

    EnsureNoTwo(matrix=matrixA_)
    EnsureNoTwo(matrix=matrixB_)

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

    # Use symbolic modulo operation
    return Mod(dotProduct, 2)
