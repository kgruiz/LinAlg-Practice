import numpy as np

from Determinat import Determinat
from Matrix import FloatIdn, FloatMatrix, Matrix, MatrixAppend
from RREF import RREF


def Inverse(matrix: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix | None:
    """
    Computes the Inverse of the input matrix using Gaussian elimination.

    This function first converts the input matrix into a `FloatMatrix` type if it's not already,
    then checks if the matrix is square (i.e., has an equal number of rows and columns). If the
    matrix is not square, an exception is raised. If the matrix is square but its determinant
    is zero, it returns `None` because a matrix with a determinant of zero is singular and
    non-invertible.

    If the matrix is invertible, the function augments it with an identity matrix of the same
    dimension, then performs Gaussian elimination to reduce it to row-reduced echelon form (RREF).
    The Inverse matrix is obtained from the right half of the augmented matrix.

    Args:
        matrix (Matrix | FloatMatrix | np.ndarray): The input matrix to invert. Can be either a
            `Matrix`, `FloatMatrix`, or a NumPy `ndarray`.

    Returns:
        FloatMatrix | None: The Inverse of the input matrix as a `FloatMatrix`. If the matrix is
        non-invertible, returns `None`.
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

    augmentedRREF = RREF(matrix=augmentedMatrix, augmentedColCount=matrix_.numRows)

    InverseMatrix = FloatMatrix(augmentedRREF[:, matrix_.numCols :])

    return InverseMatrix
