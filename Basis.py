from typing import List

import numpy as np

from Matrix import Base2Matrix, FloatMatrix, Matrix
from RREF import RREF


def Basis(span: Matrix | FloatMatrix | np.ndarray) -> FloatMatrix | Matrix:
    """
    Compute the basis of a subspace spanned by the given matrix.

    Parameters
    ----------
    span : Matrix | FloatMatrix | np.ndarray
        A matrix representing a set of vectors spanning a subspace.

    Returns
    -------
    FloatMatrix | Matrix
        The basis of the subspace.
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

    Basis = span_[:, pivotCols]

    return FloatMatrix(Basis)


class Basis:
    """
    Represents a basis for a vector space.

    Attributes
    ----------
    vectors : List[Base2Matrix]
        A list of basis vectors.
    """

    def __init__(self, vectors: List[Base2Matrix]):
        """
        Initialize a Basis instance with a list of vectors.

        Parameters
        ----------
        vectors : List[Base2Matrix]
            The basis vectors.
        """
        self.vectors = vectors

    def is_basis(self) -> bool:
        """
        Check if the current set of vectors forms a basis.

        Returns
        -------
        bool
            True if the vectors form a basis, False otherwise.
        """
        # ...existing code...
