import numpy as np
import sympy
from sympy import Matrix as SympyMatrix
from sympy import Symbol, sympify

from Matrix import FloatMatrix, Matrix


def GetPivotColumns(
    matrix: Matrix | FloatMatrix | np.ndarray, numAugmented: int = 0
) -> list:
    """
    Calculate which columns are pivot columns in the matrix.

    Parameters
    ----------
    matrix : Matrix | FloatMatrix | np.ndarray
        The matrix to analyze.
    numAugmented : int, optional
        The number of augmented columns, by default 0.

    Returns
    -------
    list
        A list of indices representing the pivot columns.
    """
    if isinstance(matrix, (Matrix, np.ndarray)):

        matrix_ = FloatMatrix(matrix)
    else:

        matrix_ = matrix

    originalNumAugmented = matrix_.numAugmented
    if numAugmented == originalNumAugmented:

        matrix_.SetNumAugmented(originalNumAugmented)
    else:

        raise Exception(
            f"Invalid augmented column count: {numAugmented}, expected: {originalNumAugmented}"
        )

    pivotColumns = []

    for colIndex in range(matrix_.numCols):

        if colIndex >= matrix_.numCols - numAugmented:

            break

        numNonZero = 0
        firstNonZeroRow = None

        for rowNum, elem in enumerate(matrix_[:, colIndex]):

            if elem != 0:

                numNonZero += 1

                if firstNonZeroRow is None:

                    firstNonZeroRow = rowNum

                if numNonZero >= 2:

                    break

        if numNonZero == 0 or numNonZero >= 2:

            continue

        firstNonZeroCol = None

        for colNum, elem in enumerate(matrix_[firstNonZeroRow, :]):

            if elem != 0:

                firstNonZeroCol = colNum

                break

        if firstNonZeroCol != colIndex:

            continue

        else:

            pivotColumns.append(colIndex)

    return pivotColumns


def GetFreeVariables(
    matrix: Matrix | FloatMatrix | np.ndarray, numAugmented: int = 0
) -> list:
    """
    Calculate which columns are free variables in the matrix.

    Parameters
    ----------
    matrix : Matrix | FloatMatrix | np.ndarray
        The matrix to analyze.
    numAugmented : int, optional
        The number of augmented columns, by default 0.

    Returns
    -------
    list
        A list of indices representing the free variable columns.

    """
    if isinstance(matrix, (Matrix, np.ndarray)):

        matrix_ = FloatMatrix(matrix)
    else:

        matrix_ = matrix

    originalNumAugmented = matrix_.numAugmented

    if numAugmented == originalNumAugmented:

        matrix_.SetNumAugmented(originalNumAugmented)
    else:

        raise Exception(
            f"Invalid augmented column count: {numAugmented}, expected: {originalNumAugmented}"
        )

    pivotColumns = GetPivotColumns(matrix_, numAugmented)
    freeVariables = [
        i for i in range(matrix_.numCols - numAugmented) if i not in pivotColumns
    ]
    return freeVariables


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

    originalNumAugmented = matrix.numAugmented

    if isinstance(matrix, Matrix):

        matrix_ = FloatMatrix(matrix)
    elif isinstance(matrix, np.ndarray):

        matrix_ = FloatMatrix(matrix)
    elif isinstance(matrix, FloatMatrix):

        matrix_ = matrix
    else:

        matrix_ = FloatMatrix(matrix)

    if augmentedColCount == originalNumAugmented:

        matrix_.SetNumAugmented(originalNumAugmented)

    else:

        raise Exception(
            f"Invalid augmented column count: {augmentedColCount}, expected: {originalNumAugmented}"
        )

    def ScaleRows(matrix: FloatMatrix) -> FloatMatrix:
        """
        Scale all rows of the matrix so that the first non-zero element in each row is 1.

        Parameters
        ----------
        matrix : FloatMatrix
            The matrix to scale.

        Returns
        -------
        FloatMatrix
            The scaled matrix.
        """

        matrix_ = matrix
        matrix_.SetNumAugmented(originalNumAugmented)

        for rowNum, row in enumerate(matrix_):

            firstNonZero = None
            for elem in row:

                if elem != 0:

                    firstNonZero = elem
                    break
            if not firstNonZero:

                continue
            for colNum, elem in enumerate(row):

                matrix_[rowNum][colNum] = sympify(elem) / sympify(firstNonZero)

        return matrix_

    def ScaleRow(matrix: FloatMatrix, rowNum: int) -> FloatMatrix:
        """
        Scale a specific row of the matrix so that the first non-zero element in the row is 1.

        Parameters
        ----------
        matrix : FloatMatrix
            The matrix to scale.
        rowNum : int
            The index of the row to scale.

        Returns
        -------
        FloatMatrix
            The matrix with the specified row scaled.
        """

        matrix_ = matrix
        matrix_.SetNumAugmented(originalNumAugmented)

        for rowNum_, row in enumerate(matrix_):

            if rowNum_ > rowNum:

                break

            if rowNum_ != rowNum:

                continue

            firstNonZero = None
            for elem in row:

                if elem != 0:

                    firstNonZero = elem
                    break
            if not firstNonZero:

                continue
            for colNum, elem in enumerate(row):

                if elem == firstNonZero:

                    matrix_[rowNum][colNum] = 1

                else:

                    matrix_[rowNum][colNum] = sympify(elem) / sympify(firstNonZero)

        return matrix_

    def CleanNegZeros(matrix: FloatMatrix) -> FloatMatrix:
        """
        Clean negative zeros in the matrix by converting them to positive zeros.

        Parameters
        ----------
        matrix : FloatMatrix
            The matrix to clean.

        Returns
        -------
        FloatMatrix
            The cleaned matrix.
        """

        matrix_ = matrix
        for rowNum, row in enumerate(matrix_):

            for colNum, elem in enumerate(row):

                if elem == -0:

                    matrix_[rowNum][colNum] = +0

        return matrix_

    def OrderRows(matrix: FloatMatrix) -> FloatMatrix:

        """
        Order the rows of the matrix based on the position of the first non-zero element in each row.

        Parameters
        ----------
        matrix : FloatMatrix
            The matrix to order.

        Returns
        -------
        FloatMatrix
            The ordered matrix.
        """

        matrix_ = matrix

        def FirstNonZeroIndex(row):

            row_ = row
            nonZeroIndices = np.nonzero(row_)[0]
            return nonZeroIndices[0] if nonZeroIndices.size > 0 else len(row_)

        sortedMatrix = FloatMatrix(
            matrix_[np.argsort([FirstNonZeroIndex(row) for row in matrix_])]
        )
        sortedMatrix.SetNumAugmented(originalNumAugmented)

        return sortedMatrix

    pivotRowNum = 0
    firstNonZeroIndex = None
    matrix_ = OrderRows(matrix=matrix_)

    while pivotRowNum < matrix_.numRows:

        ScaleRow(matrix=matrix_, rowNum=pivotRowNum)

        pivotRow = matrix_[pivotRowNum, :]
        firstNonZeroIndex = None

        for i, elem in enumerate(pivotRow):

            if elem != 0:

                assert elem == 1 or elem == sympy.Float(1)
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

            factor = sympify(row[firstNonZeroIndex]) / sympify(
                pivotRow[firstNonZeroIndex]
            )

            for colNum, elem in enumerate(row):

                matrix_[rowNum][colNum] = (
                    sympify(elem) - sympify(pivotRow[colNum]) * factor
                )

        pivotRowNum += 1

    matrix_ = ScaleRows(matrix=matrix_)
    matrix_ = CleanNegZeros(matrix=matrix_)
    matrix_ = OrderRows(matrix=matrix_)

    return matrix_
