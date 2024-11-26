import numpy as np

from Matrix import Base2Matrix, EnsureNoTwo


def Base2RREF(
    matrix: Base2Matrix | np.ndarray,
    augmentedColCount: int = None,
    verbose: bool = False,
) -> Base2Matrix:

    base = 2

    if isinstance(matrix, Base2Matrix):

        matrix_ = Base2Matrix(matrix)

    if isinstance(matrix, np.ndarray):

        matrix_ = Base2Matrix(matrix)

    else:

        matrix_ = Base2Matrix(matrix)

    if augmentedColCount is None:

        augmentedColCount = matrix_.numAugmented

    elif augmentedColCount > matrix_.numAugmented:

        errorStr = (
            f"Parameter augmentedColCount value of {augmentedColCount} is greater than "
        )
        errorStr += f"matrix's augmented column count of {matrix_.numAugmented}"

        raise Exception(errorStr)

    def ScaleRows(matrix: Base2Matrix) -> Base2Matrix:

        matrix_ = Base2Matrix(matrix)

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

                matrix_[rowNum][colNum] = (elem / firstNonZero) % base

        return matrix_

    def ScaleRow(matrix: Base2Matrix, rowNum: int) -> Base2Matrix:

        matrix_ = Base2Matrix(matrix)

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

                matrix_[rowNum][colNum] = (elem / firstNonZero) % base

        return matrix_

    def CleanNegZeros(matrix: Base2Matrix) -> Base2Matrix:

        matrix_ = Base2Matrix(matrix)

        for rowNum, row in enumerate(matrix_):

            for colNum, elem in enumerate(row):

                if elem == -0:

                    matrix_[rowNum][colNum] = +0

        return matrix_

    def OrderRows(matrix: Base2Matrix) -> Base2Matrix:

        matrix_ = Base2Matrix(matrix)

        def FirstNonZeroIndex(row):

            row_ = row

            nonZeroIndices = np.nonzero(row_)[0]

            return nonZeroIndices[0] if nonZeroIndices.size > 0 else len(row_)

        sortedMatrix = Base2Matrix(
            matrix_[np.argsort([FirstNonZeroIndex(row) for row in matrix_])]
        )

        sortedMatrix.SetNumAugmented(numAugmented=matrix_.numAugmented)

        return sortedMatrix

    pivotRowNum = 0

    FirstNonZeroIndex = None

    if verbose:

        print(matrix_)

    matrix_ = OrderRows(matrix=matrix_)

    while pivotRowNum < matrix_.numRows:

        EnsureNoTwo(matrix=matrix_)

        ScaleRow(matrix=matrix_, rowNum=pivotRowNum)

        if verbose:

            print(matrix_)

        pivotRow = matrix_[pivotRowNum, :]

        EnsureNoTwo(matrix=matrix_)

        for i, elem in enumerate(pivotRow):

            EnsureNoTwo(matrix=matrix_)

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

            EnsureNoTwo(matrix=matrix_)

            if rowNum == pivotRowNum:

                continue

            if row[FirstNonZeroIndex] == 0:

                continue

            factor = row[FirstNonZeroIndex] / pivotRow[FirstNonZeroIndex]

            EnsureNoTwo(matrix=matrix_)

            for colNum, elem in enumerate(row):

                EnsureNoTwo(matrix=matrix_)

                matrix_[rowNum][colNum] = (elem - (pivotRow[colNum] * factor)) % base

        pivotRowNum += 1

        EnsureNoTwo(matrix=matrix_)

    matrix_ = ScaleRows(matrix=matrix_)

    matrix_ = CleanNegZeros(matrix=matrix_)

    matrix_ = OrderRows(matrix=matrix_)

    EnsureNoTwo(matrix=matrix_)

    return matrix_