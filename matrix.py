import random

import numpy as np


class Matrix:
    """
    Represents an integer matrix with various initialization options.
    Can be initialized with dimensions, a numpy array, or another Matrix object.
    """

    def __init__(self, *args):
        """
        Initializes the Matrix instance.

        Args:
            *args: A variable number of arguments. Acceptable formats are:
                - (rows, cols): Create a matrix with random integers.
                - (numpy array): Create a matrix from an existing numpy array.
                - (rows, cols, min, max): Create a matrix with specified value range.
                - (Matrix object): Create a new matrix as a copy of an existing Matrix.
        """
        if len(args) == 2:

            rows, cols = args
            self.matrix = np.random.randint(0, 100, (rows, cols), dtype=int)

            self.numRows = rows
            self.numCols = cols

        elif len(args) == 1 and isinstance(args[0], np.ndarray):

            if args[0].dtype != int:

                raise Exception(
                    f"np.ndarray has elements that are not int dtype. Dtype: {args[0].dtype}"
                )

            if len(args[0].shape) == 1:

                self.matrix = np.reshape(args[0], newshape=(args[0].shape[0], 1)).copy()

            else:

                self.matrix = args[0].copy()

            self.numRows = self.matrix.shape[0]
            self.numCols = self.matrix.shape[1]

        elif len(args) == 1 and isinstance(args[0], Matrix):
            # Copy constructor
            self.matrix = args[0].matrix.copy()
            self.numRows = args[0].numRows
            self.numCols = args[0].numCols

        elif len(args) >= 2 and all(isinstance(arg, int) for arg in args[:2]):

            numRows, numCols = args[0], args[1]

            minVal = args[2] if len(args) > 2 else -100
            maxVal = args[3] if len(args) > 3 else 100

            self.numRows = numRows
            self.numCols = numCols

            self.matrix = np.zeros((numRows, numCols), dtype=int)

            for row in range(numRows):

                for col in range(numCols):

                    self.matrix[row][col] = random.randint(minVal, maxVal)
        else:

            raise ValueError(
                "Invalid arguments. Pass either (rows, cols), a numpy array, or (rows, cols, min, max)."
            )

    def __str__(self):
        """
        Returns a string representation of the matrix.

        Returns:
            str: A formatted string representing the matrix.
        """
        return MatrixString(self.matrix)

    def __getitem__(self, index):
        """
        Allows matrix indexing using square brackets.

        Args:
            index (int or tuple): Index to access matrix elements.

        Returns:
            int: The element at the specified index.
        """
        return self.matrix[index]

    def __setitem__(self, index, value):
        """
        Allows setting matrix elements using square brackets.

        Args:
            index (int or tuple): Index to access matrix elements.
            value (int): Value to set at the specified index.
        """
        self.matrix[index] = value

    def __getattr__(self, attr):
        """
        Provides access to attributes of the matrix, such as number of rows or columns.

        Args:
            attr (str): The name of the attribute to retrieve.

        Returns:
            int or ndarray: The requested attribute value.
        """
        if attr == "numRows":

            return self.numRows

        elif attr == "numCols":

            return self.numCols

        elif attr == "matrix":

            return self.matrix

        else:

            return getattr(self.matrix, attr)


def MatrixString(matrix: np.ndarray) -> str:
    """
    Formats a numpy matrix as a string with proper alignment for display.

    Args:
        matrix (np.ndarray): The matrix to format.

    Returns:
        str: A formatted string representing the matrix.
    """
    colWidths = [
        max(len(str(int(val))) for val in matrix[:, col])
        for col in range(matrix.shape[1])
    ]

    formattedMatrix = ""

    for row in matrix:
        rowStr = " ".join(
            f"{int(val)}".rjust(colWidths[i]) for i, val in enumerate(row)
        )
        formattedMatrix += f"| {rowStr} |\n"
    return formattedMatrix


class FloatMatrix:
    """
    Represents a floating-point matrix with various initialization options.
    Can be initialized with dimensions, a numpy array, or another Matrix object.
    """

    def __init__(self, *args):
        """
        Initializes the FloatMatrix instance.

        Args:
            *args: A variable number of arguments. Acceptable formats are:
                - (rows, cols): Create a matrix with random floats.
                - (numpy array): Create a matrix from an existing numpy array.
                - (Matrix object): Create a new matrix as a copy of an existing Matrix.
                - (FloatMatrix object): Create a new matrix as a copy of an existing FloatMatrix.
                - (rows, cols, min, max): Create a matrix with specified value range.
        """
        if len(args) == 2:

            rows, cols = args
            self.matrix = np.random.uniform(0, 100, (rows, cols))
            self.matrix.dtype = float

            self.numRows = rows
            self.numCols = cols

        elif len(args) == 1:

            if isinstance(args[0], np.ndarray):

                if len(args[0].shape) == 1:

                    self.matrix = np.reshape(
                        args[0], newshape=(args[0].shape[0], 1)
                    ).copy()
                    self.matrix.dtype = float

                else:

                    self.matrix = args[0].copy()
                    self.matrix.dtype = float

                self.numRows = self.matrix.shape[0]
                self.numCols = self.matrix.shape[1]

            elif isinstance(args[0], Matrix):
                # Copy constructor from Matrix
                self.matrix = args[0].matrix.astype(float).copy()
                self.numRows = args[0].numRows
                self.numCols = args[0].numCols

            elif isinstance(args[0], FloatMatrix):
                # Copy constructor from FloatMatrix
                self.matrix = args[0].matrix.copy()
                self.numRows = args[0].numRows
                self.numCols = args[0].numCols

        elif len(args) >= 2 and all(isinstance(arg, int) for arg in args[:2]):

            numRows, numCols = args[0], args[1]

            minVal = args[2] if len(args) > 2 else -100
            maxVal = args[3] if len(args) > 3 else 100

            self.numRows = numRows
            self.numCols = numCols

            self.matrix = np.zeros((numRows, numCols), dtype=float)

            for row in range(numRows):

                for col in range(numCols):

                    self.matrix[row][col] = random.random() * random.randint(
                        minVal, maxVal
                    )
        else:

            raise ValueError(
                "Invalid arguments. Pass either (rows, cols), a numpy array, or (rows, cols, min, max)."
            )

    def __str__(self):
        """
        Returns a string representation of the float matrix.

        Returns:
            str: A formatted string representing the float matrix.
        """
        return FloatMatrixString(self.matrix)

    def __getitem__(self, index):
        """
        Allows matrix indexing using square brackets.

        Args:
            index (int or tuple): Index to access matrix elements.

        Returns:
            float: The element at the specified index.
        """
        return self.matrix[index]

    def __setitem__(self, index, value):
        """
        Allows setting matrix elements using square brackets.

        Args:
            index (int or tuple): Index to access matrix elements.
            value (float): Value to set at the specified index.
        """
        self.matrix[index] = value

    def __getattr__(self, attr):
        """
        Provides access to attributes of the matrix, such as number of rows or columns.

        Args:
            attr (str): The name of the attribute to retrieve.

        Returns:
            int or ndarray: The requested attribute value.
        """
        if attr == "numRows":

            return self.numRows

        elif attr == "numCols":

            return self.numCols

        elif attr == "matrix":

            return self.matrix

        else:

            return getattr(self.matrix, attr)


def FloatMatrixString(matrix: np.ndarray) -> str:
    """
    Formats a numpy float matrix as a string with proper alignment for display.

    Args:
        matrix (np.ndarray): The matrix to format.

    Returns:
        str: A formatted string representing the matrix.
    """
    colWidths = [
        max(len(f"{val:.2f}") for val in matrix[:, col])
        for col in range(matrix.shape[1])
    ]

    formattedMatrix = ""

    for row in matrix:

        rowStr = " ".join(f"{val:.2f}".rjust(colWidths[i]) for i, val in enumerate(row))
        formattedMatrix += f"| {rowStr} |\n"

    return formattedMatrix


class Idn(Matrix):
    """
    Represents an identity matrix with int values.
    Initializes a square identity matrix of given dimension.
    """

    def __init__(self, dimension: int):
        """
        Initializes the Idn instance as an int identity matrix.

        Args:
            dimension (int): The size of the identity matrix (number of rows and columns).
        """
        idn = np.zeros(shape=(dimension, dimension), dtype=int)

        for row in range(dimension):

            idn[row][row] = 1

        super().__init__(idn)


class FloatIdn(FloatMatrix):
    """
    Represents an identity matrix with floating-point values.
    Initializes a square identity matrix of given dimension.
    """

    def __init__(self, dimension: int):
        """
        Initializes the FloatIdn instance as a floating-point identity matrix.

        Args:
            dimension (int): The size of the identity matrix (number of rows and columns).
        """
        idn = np.zeros(shape=(dimension, dimension), dtype=float)

        for row in range(dimension):

            idn[row][row] = 1

        super().__init__(idn)


def Append(
    matrixA: Matrix | FloatMatrix | np.ndarray,
    matrixB: Matrix | FloatMatrix | np.ndarray,
    horizontalStack: bool = True,
) -> FloatMatrix | Matrix:
    """
    Appends two matrices (or arrays) either horizontally or vertically, and returns
    the result as a Matrix or FloatMatrix depending on the input types.

    Parameters:
    ----------
    matrixA : Matrix | FloatMatrix | np.ndarray
        The first matrix to append. Can be of type Matrix, FloatMatrix, or np.ndarray.

    matrixB : Matrix | FloatMatrix | np.ndarray
        The second matrix to append. Can be of type Matrix, FloatMatrix, or np.ndarray.

    horizontalStack : bool, optional, default=False
        If True, the matrices are appended horizontally (side by side).
        If False, the matrices are appended vertically (one on top of the other).

    Returns:
    -------
    FloatMatrix | Matrix
        A new Matrix or FloatMatrix that contains the result of the append operation.
        The result is a FloatMatrix if either input is a FloatMatrix, otherwise a Matrix.

    Raises:
    ------
    ValueError
        If the dimensions of the input matrices are incompatible for the selected stacking operation.

    Notes:
    ------
    The function copies the matrix data from the input Matrix or FloatMatrix objects before performing the append operation.
    If the inputs are `np.ndarray`, they will be treated as `Matrix` by default.
    """

    arrayA = matrixA.matrix.copy()
    arrayB = matrixB.matrix.copy()

    if horizontalStack:

        if matrixA.numRows != matrixB.numRows:

            raise Exception(
                f"Cannot horizontally append matrices with different number of rows.\n\
                MatrixA Shape: {matrixA.numRows} x {matrixA.numCols}\n\
                    MatrixB Shape: {matrixB.numRows} x {matrixB.numCols}"
            )

        resultArray = np.hstack((arrayA, arrayB))

    else:

        if matrixA.numCols != matrixB.numCols:

            raise Exception(
                f"Cannot vertically append matrices with different number of columns.\n\
                MatrixA Shape: {matrixA.numRows} x {matrixA.numCols}\n\
                    MatrixB Shape: {matrixB.numRows} x {matrixB.numCols}"
            )

        resultArray = np.vstack((arrayA, arrayB))

    if isinstance(matrixA, FloatMatrix) or isinstance(matrixB, FloatMatrix):

        resultMatrix = FloatMatrix(resultArray)

    else:

        resultMatrix = Matrix(resultArray)

    return resultMatrix
