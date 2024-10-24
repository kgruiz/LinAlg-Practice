import numpy as np

from basis import Basis
from gramSchmidt import GramSchmidt
from matrix import FloatMatrix, Matrix
from multiply import Multiply
from transpose import Transpose


def qrDecomposition(
    span: Matrix | FloatMatrix | np.ndarray,
) -> tuple[FloatMatrix, FloatMatrix]:
    """
    Peforms QR Decompositon on the input basis

    Returns a tuple with (q, r)
    """

    def cleanNegZeros(matrix: FloatMatrix) -> FloatMatrix:

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

    q = cleanNegZeros(matrix=q)
    r = cleanNegZeros(matrix=r)

    return q, r
