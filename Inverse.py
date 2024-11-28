import numpy as np

from Determinat import Determinat
from Matrix import FloatIdn, FloatMatrix, Matrix, MatrixAppend
from RREF import RREF


def Inverse(matrix: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix | None:
    """
    Compute the inverse of a matrix using Gaussian elimination.

    This function converts the input matrix into a `FloatMatrix` type if it's not already,
    verifies that the matrix is square, and checks if the determinant is non-zero.
    If invertible, it augments the matrix with an identity matrix and performs
    Gaussian elimination to obtain the inverse.

    Parameters
    ----------
    matrix : Matrix | FloatMatrix | np.ndarray
        The input matrix to invert. Supported types are `Matrix`, `FloatMatrix`, or `numpy.ndarray`.

    Returns
    -------
    FloatMatrix | None
        The inverse of the input matrix as a `FloatMatrix`. Returns `None` if the matrix is singular.

    Raises
    ------
    Exception
        If the input matrix is not square.
    """

    if isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)

    if isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)

    else:

        matrix_ = FloatMatrix(matrix)

    if matrix_.numRows != matrix_.numCols:

        raise Exception(
            f"Cannot compute the Inverse of a non-square matrix.\n\
            Input matrix has shape: {matrix_.numRows} x {matrix_.numCols}"
        )

    if Determinat(matrix=matrix_) == 0:

        return None

    idn = FloatIdn(dimension=matrix_.numRows)

    augmentedMatrix = MatrixAppend(matrixA=matrix_, matrixB=idn, horizontalStack=True)

    augmentedMatrix.SetNumAugmented(matrix_.numRows)

    augmentedRREF = RREF(matrix=augmentedMatrix, augmentedColCount=matrix_.numRows)

    InverseMatrix = FloatMatrix(augmentedRREF[:, matrix_.numCols :])

    return InverseMatrix
