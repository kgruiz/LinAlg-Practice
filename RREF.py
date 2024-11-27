import numpy as np
from sympy import Matrix as SympyMatrix

from Matrix import FloatMatrix, Matrix


def RREF(
    matrix: Matrix | FloatMatrix | np.ndarray, augmentedColCount: int = 0
) -> FloatMatrix:
    """
    Compute the Reduced Row Echelon Form (RREF) of a matrix.

    Parameters
    ----------
    matrix : Matrix | FloatMatrix | np.ndarray
        The input matrix to convert to RREF.
    augmentedColCount : int, optional
        The number of augmented columns, by default 0.

    Returns
    -------
    FloatMatrix
        The matrix in Reduced Row Echelon Form.
    """

    if isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)

    if isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)

    else:

        matrix_ = FloatMatrix(matrix)

    def ScaleRows(matrix: FloatMatrix) -> FloatMatrix:

        matrix_ = matrix

        for rowNum, row in enumerate(matrix_):

            firstNonZero = None

            for elem in row:

                firstNonZero = None

                if elem != 0:

                    firstNonZero = elem

                    break

            if not firstNonZero:

                continue

            for colNum, elem in enumerate(row):

                matrix_[rowNum][colNum] = round(elem / firstNonZero, 30)

        return matrix_

    def ScaleRow(matrix: FloatMatrix, rowNum: int) -> FloatMatrix:

        matrix_ = matrix

        for rowNum_, row in enumerate(matrix_):

            if rowNum_ != rowNum:

                continue

            firstNonZero = None

            for elem in row:

                firstNonZero = None

                if elem != 0:

                    firstNonZero = elem

                    break

            if not firstNonZero:

                continue

            for colNum, elem in enumerate(row):

                matrix_[rowNum][colNum] = round(elem / firstNonZero, 30)

        return matrix_

    def CleanNegZeros(matrix: FloatMatrix) -> FloatMatrix:

        matrix_ = matrix

        for rowNum, row in enumerate(matrix_):

            for colNum, elem in enumerate(row):

                if elem == -0:

                    matrix_[rowNum][colNum] = +0

        return matrix_

    def OrderRows(matrix: FloatMatrix) -> FloatMatrix:

        matrix_ = matrix

        def FirstNonZeroIndex(row):

            row_ = row

            nonZeroIndices = np.nonzero(row_)[0]

            return nonZeroIndices[0] if nonZeroIndices.size > 0 else len(row_)

        sortedMatrix = FloatMatrix(
            matrix_[np.argsort([FirstNonZeroIndex(row) for row in matrix_])]
        )

        return sortedMatrix

    pivotRowNum = 0

    FirstNonZeroIndex = None

    matrix_ = OrderRows(matrix=matrix_)

    while pivotRowNum < matrix_.numRows:

        ScaleRow(matrix=matrix_, rowNum=pivotRowNum)

        pivotRow = matrix_[pivotRowNum, :]

        for i, elem in enumerate(pivotRow):

            FirstNonZeroIndex = None

            if elem != 0:

                assert elem == 1

                FirstNonZeroIndex = i

                break

        if (
            FirstNonZeroIndex is not None
            and FirstNonZeroIndex + augmentedColCount >= matrix_.numCols
        ):

            break

        if FirstNonZeroIndex is None:

            pivotRowNum += 1

            continue

        for rowNum, row in enumerate(matrix_):

            if rowNum == pivotRowNum:

                continue

            if row[FirstNonZeroIndex] == 0:

                continue

            factor = row[FirstNonZeroIndex] / pivotRow[FirstNonZeroIndex]

            for colNum, elem in enumerate(row):

                matrix_[rowNum][colNum] = round(elem - (pivotRow[colNum] * factor), 12)

        pivotRowNum += 1

    matrix_ = ScaleRows(matrix=matrix_)

    matrix_ = CleanNegZeros(matrix=matrix_)

    matrix_ = OrderRows(matrix=matrix_)

    sympyMatrix = SympyMatrix(matrix_.matrix)

    rrefMatrix, _ = sympyMatrix.rref()

    resultArray = np.array(rrefMatrix).astype(object)

    return FloatMatrix(resultArray)
