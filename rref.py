import numpy as np

from matrix import FloatMatrix, Matrix


def RREF(
    matrix: Matrix | FloatMatrix | np.ndarray, augmentedColCount: int = 0
) -> FloatMatrix:

    if isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)

    if isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)

    else:

        matrix_ = FloatMatrix(matrix)

    def scaleRows(matrix: FloatMatrix) -> FloatMatrix:

        matrix_ = matrix

        for rowNum, row in enumerate(matrix_):

            firstNonZero = None

            for elem in row:

                firstNonZero = None

                if abs(elem - 0) > 0.0000001:

                    firstNonZero = elem

                    break

            if not firstNonZero:

                continue

            for colNum, elem in enumerate(row):

                matrix_[rowNum][colNum] = elem / firstNonZero

        return matrix_

    def scaleRow(matrix: FloatMatrix, rowNum: int) -> FloatMatrix:

        matrix_ = matrix

        for rowNum_, row in enumerate(matrix_):

            if rowNum_ != rowNum:

                continue

            firstNonZero = None

            for elem in row:

                firstNonZero = None

                if abs(elem - 0) > 0.0000001:

                    firstNonZero = elem

                    break

            if not firstNonZero:

                continue

            for colNum, elem in enumerate(row):

                matrix_[rowNum][colNum] = elem / firstNonZero

        return matrix_

    def cleanNegZeros(matrix: FloatMatrix) -> FloatMatrix:

        matrix_ = matrix

        for rowNum, row in enumerate(matrix_):

            for colNum, elem in enumerate(row):

                if elem == -0:

                    matrix_[rowNum][colNum] = +0

        return matrix_

    def orderRows(matrix: FloatMatrix) -> FloatMatrix:

        matrix_ = matrix

        def firstNonZeroIndex(row):

            row_ = row

            nonZeroIndices = np.nonzero(row_)[0]

            return nonZeroIndices[0] if nonZeroIndices.size > 0 else len(row_)

        sortedMatrix = FloatMatrix(
            matrix_[np.argsort([firstNonZeroIndex(row) for row in matrix_])]
        )

        return sortedMatrix

    pivotRowNum = 0

    firstNonZeroIndex = None

    matrix_ = orderRows(matrix=matrix_)

    while pivotRowNum < matrix_.numRows:

        scaleRow(matrix=matrix_, rowNum=pivotRowNum)

        pivotRow = matrix_[pivotRowNum, :]

        for i, elem in enumerate(pivotRow):

            firstNonZeroIndex = None

            if abs(elem - 0) > 0.0000001:

                assert elem == 1

                firstNonZeroIndex = i

                break

        if (
            firstNonZeroIndex is not None
            and firstNonZeroIndex + augmentedColCount >= matrix_.numCols
        ):

            break

        if firstNonZeroIndex is None:

            pivotRowNum += 1

            continue

        for rowNum, row in enumerate(matrix_):

            if rowNum == pivotRowNum:

                continue

            if row[firstNonZeroIndex] == 0:

                continue

            factor = row[firstNonZeroIndex] / pivotRow[firstNonZeroIndex]

            for colNum, elem in enumerate(row):

                matrix_[rowNum][colNum] = elem - (pivotRow[colNum] * factor)

        pivotRowNum += 1

    matrix_ = scaleRows(matrix=matrix_)

    matrix_ = cleanNegZeros(matrix=matrix_)

    matrix_ = orderRows(matrix=matrix_)

    return matrix_
