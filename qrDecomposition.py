import numpy as np

from Basis import Basis
from GramSchmidt import GramSchmidt
from Matrix import FloatMatrix, Matrix
from Multiply import Multiply
from Transpose import Transpose


def QRDecomposition(
    span: Matrix | FloatMatrix | np.ndarray,
) -> tuple[FloatMatrix, FloatMatrix]:
    """
    Peforms QR Decompositon on the input span

    Returns a tuple with (q, r)
    """

    def CleanNegZeros(matrix: FloatMatrix) -> FloatMatrix:

        matrix_ = matrix

        for rowNum, row in enumerate(matrix_):

            for colNum, elem in enumerate(row):

                if elem == -0:

                    matrix_[rowNum][colNum] = +0

        return matrix_

    if isinstance(span, Matrix):

        span_ = FloatMatrix(span)

    if isinstance(span, np.ndarray):

        span_ = FloatMatrix(span)

    else:

        span_ = FloatMatrix(span)

    basis = Basis(span=span_)

    orthonormalBasis = GramSchmidt(span=basis)

    q = FloatMatrix(orthonormalBasis)

    qT = Transpose(matrix=q)

    r = Multiply(matrixA=qT, matrixB=basis)

    q = CleanNegZeros(matrix=q)
    r = CleanNegZeros(matrix=r)

    return q, r
