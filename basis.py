import numpy as np

from matrix import FloatMatrix, Matrix
from rref import RREF


def Basis(span: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix | Matrix:
    """
    Computes the basis of a subspace spanned by the given matrix.

    The function takes a matrix (which could be a custom Matrix type, a FloatMatrix, or a NumPy ndarray)
    that spans a subspace and returns the basis of this subspace. It first converts the input matrix to
    a FloatMatrix, then calculates its row-reduced echelon form (RREF). The function identifies the pivot
    columns in the RREF matrix and returns the corresponding columns from the original matrix as the basis.

    Args:
        span: A matrix or array that represents a set of vectors spanning a subspace. It can be a Matrix,
        FloatMatrix, or NumPy ndarray.

    Returns:
        A FloatMatrix representing the basis of the subspace.
    """

    if isinstance(span, Matrix):

        span_ = FloatMatrix(span)

    if isinstance(span, np.ndarray):

        span_ = FloatMatrix(span)

    else:

        span_ = FloatMatrix(span)

    spanRREF = RREF(matrix=span_)

    pivotCols = []

    for colNum in range(spanRREF.numCols):

        col = FloatMatrix(spanRREF[:, colNum])

        oneCount = 0
        oneRowIndex = None

        for i, elem in enumerate(col):

            if abs(elem - 1) < 0.00001:

                oneCount += 1

                oneRowIndex = i

        if oneRowIndex is not None and oneCount == 1:

            leadingOneIndex = None

            for colIndex, elem in enumerate(spanRREF[oneRowIndex]):

                if abs(elem - 1) < 0.00001:

                    leadingOneIndex = colIndex

                    break

            if leadingOneIndex == colNum:

                pivotCols.append(colNum)

    basis = span_[:, pivotCols]

    return FloatMatrix(basis)
