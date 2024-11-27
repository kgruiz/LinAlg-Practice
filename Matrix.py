import random
from typing import Union

import numpy as np
from sympy import Basic, Symbol, pretty, symbols


class Matrix:
    """
    Represents an integer matrix with various initialization options.

    Can be initialized with dimensions, a numpy array, another Matrix object,
    or a FloatMatrix with integer values.
    """

    def __init__(self, *args):
        """
        Initialize the Matrix instance.

        Parameters
        ----------
        *args : various
            A variable number of arguments. Acceptable formats are:
            - (rows, cols): Create a matrix with random integers.
            - (numpy array): Create a matrix from an existing numpy array.
            - (rows, cols, min, max): Create a matrix with specified value range.
            - (Matrix object): Create a new matrix as a copy of an existing Matrix.
            - (FloatMatrix object): Create a Matrix from a FloatMatrix if all floats are integers.
            - (Symbol object): Create a 1x1 matrix with a Symbol.
            - (list or tuple of Symbols): Create a column matrix with Symbols.
        """
        self.numAugmented = 0
        if len(args) == 2:

            rows, cols = args
            self.matrix = np.array(
                [
                    [Symbol(f"a{row}{col}") for col in range(cols)]
                    for row in range(rows)
                ],
                dtype=object,
            )
            self.numRows = rows
            self.numCols = cols

        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self.matrix = args[0].astype(object).copy()
            self.numRows, self.numCols = self.matrix.shape

        elif len(args) == 1 and isinstance(args[0], Matrix):
            # Copy constructor
            self.matrix = args[0].matrix.copy()
            self.numRows = args[0].numRows
            self.numCols = args[0].numCols
            self.numAugmented = args[0].numAugmented

        elif len(args) == 1 and isinstance(args[0], FloatMatrix):
            # New constructor: Convert FloatMatrix to Matrix if all floats are integers
            self.matrix = args[0].matrix.astype(object).copy()
            self.numRows, self.numCols = self.matrix.shape

        elif len(args) == 1 and isinstance(args[0], Symbol):
            self.matrix = np.array([[args[0]]], dtype=object)
            self.numRows = 1
            self.numCols = 1

        elif (
            len(args) == 1
            and isinstance(args[0], (list, tuple))
            and all(isinstance(el, Symbol) for el in args[0])
        ):
            self.matrix = np.array(args[0], dtype=object).reshape(-1, 1)
            self.numRows = self.matrix.shape[0]
            self.numCols = self.matrix.shape[1]

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
                "Invalid arguments. Pass either (rows, cols), a numpy array, a FloatMatrix with integer values, or (rows, cols, min, max)."
            )

    def __str__(self):
        """
        Return a string representation of the matrix.

        Returns
        -------
        str
            A formatted string representing the matrix.
        """
        return MatrixString(self.matrix, self.numAugmented)

    def __getitem__(self, index):
        """
        Access matrix elements using square brackets.

        Parameters
        ----------
        index : int or tuple
            Index to access matrix elements.

        Returns
        -------
        int
            The element at the specified index.
        """
        return self.matrix[index]

    def __setitem__(self, index, value):
        """
        Set matrix elements using square brackets.

        Parameters
        ----------
        index : int or tuple
            Index to access matrix elements.
        value : int
            Value to set at the specified index.
        """
        self.matrix[index] = value

    def __len__(self):
        """
        Get the number of rows in the matrix.

        Returns
        -------
        int
            The number of rows in the matrix.
        """
        return self.numRows

    def SetNumAugmented(self, numAugmented):
        """
        Set the number of augmented columns.

        Parameters
        ----------
        numAugmented : int
            The number of augmented columns.

        Raises
        ------
        ValueError
            If `numAugmented` is negative or exceeds the number of columns minus one.
        """
        if numAugmented < 0 or numAugmented > self.numCols - 1:

            raise ValueError("numAugmented must be between 0 and numCols - 1.")
        self.numAugmented = numAugmented

    def __getattr__(self, attr):
        """
        Access matrix attributes such as number of rows or columns.

        Parameters
        ----------
        attr : str
            The name of the attribute to retrieve.

        Returns
        -------
        int or ndarray
            The requested attribute value.

        Raises
        ------
        AttributeError
            If the attribute does not exist.
        """
        if attr in object.__getattribute__(self, "__dict__"):

            return object.__getattribute__(self, attr)

        if attr == "numRows":

            return object.__getattribute__(self, "numRows")

        elif attr == "numCols":

            return object.__getattribute__(self, "numCols")

        elif attr == "shape":

            return self.matrix.shape

        elif attr == "matrix":

            return object.__getattribute__(self, "matrix")

        else:

            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            )


def MatrixString(matrix, numAugmented=0):
    """
    Formats a numpy matrix as a string with proper alignment for display, including augmented column separation.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to format.
    numAugmented : int
        The number of augmented columns.

    Returns
    -------
    str
        A formatted string representing the matrix.
    """
    colWidths = []
    displayVals = []

    for row in matrix:

        displayRow = []
        for val in row:

            if isinstance(val, Basic):

                displayVal = pretty(val)

            else:

                displayVal = str(val)

            displayRow.append(displayVal)
        displayVals.append(displayRow)

    for col in range(matrix.shape[1]):

        maxWidth = max(len(displayVals[row][col]) for row in range(matrix.shape[0]))
        colWidths.append(maxWidth)

    lines = []
    for row in range(matrix.shape[0]):

        rowStr = "|"  # Add starting pipe
        for col in range(matrix.shape[1]):

            displayVal = displayVals[row][col]
            rowStr += " " + displayVal.rjust(
                colWidths[col] + 1
            )  # Add space before each column
            if (
                col < matrix.shape[1] - 1
            ):  # Add space between columns except the last one
                rowStr += " "
            if numAugmented != 0 and col == matrix.shape[1] - numAugmented - 1:

                rowStr += " :"
        rowStr += " |"  # Add ending pipe
        lines.append(rowStr)

    return "\n".join(lines)


class FloatMatrix:
    """
    Represents a floating-point matrix with various initialization options.

    Can be initialized with dimensions, a numpy array, or another Matrix object.
    """

    def __init__(self, *args):
        """
        Initialize the FloatMatrix instance.

        Parameters
        ----------
        *args : various
            A variable number of arguments. Acceptable formats are:
            - (rows, cols): Create a matrix with random floats.
            - (numpy array): Create a matrix from an existing numpy array.
            - (Matrix object): Create a new matrix as a copy of an existing Matrix.
            - (FloatMatrix object): Create a new matrix as a copy of an existing FloatMatrix.
            - (rows, cols, min, max): Create a matrix with specified value range.
            - (Symbol object): Create a 1x1 matrix with a Symbol.
            - (list or tuple of Symbols): Create a column matrix with Symbols.
        """

        self.numAugmented = 0
        if len(args) == 2 and all(isinstance(arg, int) for arg in args):

            rows, cols = args
            self.matrix = np.array(
                [
                    [Symbol(f"a{row}{col}") for col in range(cols)]
                    for row in range(rows)
                ],
                dtype=object,
            )
            self.numRows = rows
            self.numCols = cols

        elif len(args) == 1:

            if isinstance(args[0], np.ndarray):

                if len(args[0].shape) == 1:

                    # If it's a 1D array, reshape it to a column matrix
                    self.matrix = np.reshape(
                        args[0], newshape=(args[0].shape[0], 1)
                    ).astype(object)
                else:

                    # Copying the matrix and ensuring dtype is float
                    self.matrix = args[0].astype(object).copy()

                self.numRows = self.matrix.shape[0]
                self.numCols = self.matrix.shape[1]

            elif isinstance(args[0], FloatMatrix):
                # Copy constructor from FloatMatrix
                self.matrix = args[0].matrix.copy()
                self.numRows = args[0].numRows
                self.numCols = args[0].numCols
                self.numAugmented = args[0].numAugmented

            elif isinstance(args[0], Matrix):
                # Copy constructor from Matrix (assuming Matrix has a matrix attribute)
                self.matrix = args[0].matrix.astype(object).copy()
                self.numRows = args[0].numRows
                self.numCols = args[0].numCols
                self.numAugmented = args[0].numAugmented

            elif isinstance(args[0], Symbol):
                self.matrix = np.array([[args[0]]], dtype=object)
                self.numRows = 1
                self.numCols = 1

            elif isinstance(args[0], (list, tuple)) and all(
                isinstance(el, Symbol) for el in args[0]
            ):
                self.matrix = np.array(args[0], dtype=object).reshape(-1, 1)
                self.numRows = self.matrix.shape[0]
                self.numCols = self.matrix.shape[1]

        elif len(args) >= 2 and all(isinstance(arg, int) for arg in args[:2]):

            numRows, numCols = args[0], args[1]

            minVal = args[2] if len(args) > 2 else -100
            maxVal = args[3] if len(args) > 3 else 100

            self.numRows = numRows
            self.numCols = numCols

            # Initialize the matrix with random values between minVal and maxVal
            self.matrix = np.random.uniform(minVal, maxVal, (numRows, numCols)).astype(
                float
            )

        else:

            raise ValueError(
                "Invalid arguments. Pass either (rows, cols), a numpy array, or (rows, cols, min, max)."
            )

    def __str__(self):
        """
        Return a string representation of the float matrix.

        Returns
        -------
        str
            A formatted string representing the float matrix.
        """
        return FloatMatrixString(self.matrix, self.numAugmented)

    def __getitem__(self, index):
        """
        Access float matrix elements using square brackets.

        Parameters
        ----------
        index : int or tuple
            Index to access matrix elements.

        Returns
        -------
        float
            The element at the specified index.
        """
        return self.matrix[index]

    def __setitem__(self, index, value):
        """
        Set float matrix elements using square brackets.

        Parameters
        ----------
        index : int or tuple
            Index to access matrix elements.
        value : float
            Value to set at the specified index.
        """
        self.matrix[index] = value

    def __len__(self):
        """
        Get the number of rows in the float matrix.

        Returns
        -------
        int
            The number of rows in the float matrix.
        """
        return self.numRows

    def SetNumAugmented(self, numAugmented):
        """
        Set the number of augmented columns in the float matrix.

        Parameters
        ----------
        numAugmented : int
            The number of augmented columns.

        Raises
        ------
        ValueError
            If `numAugmented` is negative or exceeds the number of columns minus one.
        """
        if numAugmented < 0 or numAugmented > self.numCols - 1:

            raise ValueError("numAugmented must be between 0 and numCols - 1.")
        self.numAugmented = numAugmented

    def __getattr__(self, attr):
        """
        Access float matrix attributes such as number of rows or columns.

        Parameters
        ----------
        attr : str
            The name of the attribute to retrieve.

        Returns
        -------
        int or ndarray
            The requested attribute value.

        Raises
        ------
        AttributeError
            If the attribute does not exist.
        """
        if attr in object.__getattribute__(self, "__dict__"):

            return object.__getattribute__(self, attr)

        if attr == "numRows":

            return object.__getattribute__(self, "numRows")

        elif attr == "numCols":

            return object.__getattribute__(self, "numCols")

        elif attr == "shape":

            return self.matrix.shape

        elif attr == "matrix":

            return object.__getattribute__(self, "matrix")

        else:

            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            )


def FloatMatrixString(matrix: np.ndarray, numAugmented: int) -> str:
    """
    Formats a numpy float matrix as a string with proper alignment for display, including augmented column separation.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to format.
    numAugmented : int
        The number of augmented columns.

    Returns
    -------
    str
        A formatted string representing the float matrix.
    """
    colWidths = []
    displayVals = []

    for row in matrix:

        displayRow = []
        for val in row:

            if isinstance(val, Basic):

                displayVal = pretty(val)

            else:

                displayVal = str(val)

            displayRow.append(displayVal)
        displayVals.append(displayRow)

    for col in range(matrix.shape[1]):

        maxWidth = max(len(displayVals[row][col]) for row in range(matrix.shape[0]))
        colWidths.append(maxWidth)

    lines = []
    for row in range(matrix.shape[0]):

        rowStr = "|"  # Add starting pipe
        for i in range(matrix.shape[1]):

            displayVal = displayVals[row][i]
            rowStr += " " + displayVal.rjust(
                colWidths[i] + 1
            )  # Add space before each column
            if i < matrix.shape[1] - 1:  # Add space between columns except the last one
                rowStr += " "
            if numAugmented != 0 and i == matrix.shape[1] - numAugmented - 1:

                rowStr += " :"
        rowStr += " |"  # Add ending pipe
        lines.append(rowStr)

    return "\n".join(lines)


class Idn(Matrix):
    """
    Represents an identity matrix with integer values.

    Initializes a square identity matrix of the given dimension.
    """

    def __init__(self, dimension: int):
        """
        Initialize the integer identity matrix.

        Parameters
        ----------
        dimension : int
            The size of the identity matrix (number of rows and columns).
        """
        self.numAugmented = 0
        idn = np.zeros(shape=(dimension, dimension), dtype=int)

        for row in range(dimension):

            idn[row][row] = 1

        super().__init__(idn)


class FloatIdn(FloatMatrix):
    """
    Represents an identity matrix with floating-point values.

    Initializes a square identity matrix of the given dimension.
    """

    def __init__(self, dimension: int):
        """
        Initialize the floating-point identity matrix.

        Parameters
        ----------
        dimension : int
            The size of the identity matrix (number of rows and columns).
        """
        self.numAugmented = 0
        idn = np.zeros(shape=(dimension, dimension), dtype=float)

        for row in range(dimension):

            idn[row][row] = 1

        super().__init__(idn)


def MatrixAppend(
    matrixA: Union["Matrix", "FloatMatrix", np.ndarray],
    matrixB: Union["Matrix", "FloatMatrix", np.ndarray],
    horizontalStack: bool = True,
) -> Union["FloatMatrix", "Matrix"]:
    """
    Append two matrices either horizontally or vertically.

    Parameters
    ----------
    matrixA : Matrix | FloatMatrix | np.ndarray
        The first matrix to append.
    matrixB : Matrix | FloatMatrix | np.ndarray
        The second matrix to append.
    horizontalStack : bool, optional
        If True, append horizontally (default is True).

    Returns
    -------
    FloatMatrix | Matrix
        The resulting appended matrix.

    Raises
    ------
    ValueError
        If the dimensions are incompatible for the chosen append operation.
    """
    arrayA = matrixA.matrix.copy()
    arrayB = matrixB.matrix.copy()

    if horizontalStack:

        if matrixA.numRows != matrixB.numRows:

            raise Exception(
                f"Cannot horizontally MatrixAppend matrices with different number of rows.\n                MatrixA Shape: {matrixA.numRows} x {matrixA.numCols}\n                    MatrixB Shape: {matrixB.numRows} x {matrixB.numCols}"
            )

        resultArray = np.hstack((arrayA, arrayB))

    else:

        if matrixA.numCols != matrixB.numCols:

            raise Exception(
                f"Cannot vertically MatrixAppend matrices with different number of columns.\n                MatrixA Shape: {matrixA.numRows} x {matrixA.numCols}\n                    MatrixB Shape: {matrixB.numRows} x {matrixB.numCols}"
            )

        resultArray = np.vstack((arrayA, arrayB))

    if isinstance(matrixA, FloatMatrix) or isinstance(matrixB, FloatMatrix):

        resultMatrix = FloatMatrix(resultArray)

    else:

        resultMatrix = Matrix(resultArray)

    return resultMatrix


class Base2Matrix:
    """
    Represents an integer matrix with various initialization options.
    Can be initialized with dimensions, a numpy array, another Matrix object, or a FloatMatrix with integer values.
    """

    def __init__(
        self,
        *args: Union[
            int, np.ndarray, "Base2Matrix", "Matrix", "FloatMatrix", Symbol, list, tuple
        ],
    ):
        """
        Initializes the Matrix instance.

        Parameters
        ----------
        *args : various
            A variable number of arguments. Acceptable formats are:
                - (rows, cols): Create a matrix with random integers.
                - (numpy array): Create a matrix from an existing numpy array.
                - (rows, cols, min, max): Create a matrix with specified value range.
                - (Matrix object): Create a new matrix as a copy of an existing Matrix.
                - (FloatMatrix object): Create a Matrix from a FloatMatrix if all floats are integers.
                - (Symbol object): Create a 1x1 matrix with a Symbol.
                - (list or tuple of Symbols): Create a column matrix with Symbols.
        """
        self.numAugmented = 0
        if len(args) == 2:

            rows, cols = args
            self.matrix = np.array(
                [
                    [Symbol(f"a{row}{col}") % 2 for col in range(cols)]
                    for row in range(rows)
                ],
                dtype=object,
            )
            self.numRows = rows
            self.numCols = cols

        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self.matrix = args[0].astype(object).copy()
            self.numRows, self.numCols = self.matrix.shape

        elif len(args) == 1 and isinstance(args[0], Base2Matrix):
            # Copy constructor
            self.matrix = args[0].matrix.copy()
            self.numRows = args[0].numRows
            self.numCols = args[0].numCols
            self.numAugmented = args[0].numAugmented

        elif len(args) == 1 and isinstance(args[0], Matrix):

            self.matrix = args[0].matrix.copy()
            self.numRows = args[0].numRows
            self.numCols = args[0].numCols
            self.numAugmented = args[0].numAugmented

        elif len(args) == 1 and isinstance(args[0], FloatMatrix):
            # New constructor: Convert FloatMatrix to Matrix if all floats are integers
            floatMatrix = args[0].matrix
            if not np.all(floatMatrix == np.round(floatMatrix)):

                raise ValueError(
                    "FloatMatrix contains non-integer values and cannot be converted to Matrix."
                )
            intMatrix = floatMatrix.astype(int)
            self.matrix = intMatrix

            for row in range(intMatrix.shape[0]):

                for col in range(intMatrix.shape[1]):

                    if intMatrix[row][col] == 2:

                        raise Exception(
                            f"TWO HAPPENS {row}|{col} == {intMatrix[row][col]}"
                        )

    def __str__(self) -> str:
        """
        Returns a string representation of the matrix.

        Returns
        -------
        str
            A formatted string representing the matrix.
        """
        EnsureNoTwo(matrix=self.matrix)
        return MatrixString(self.matrix, self.numAugmented)

    def __getitem__(self, index: Union[int, tuple]) -> Union[int, float, Symbol]:
        """
        Gets the item at the specified index.

        Parameters
        ----------
        index : int or tuple
            The index of the item to retrieve.

        Returns
        -------
        Union[int, float, Symbol]
            The element at the specified index.
        """
        return self.matrix[index]

    def __setitem__(
        self, index: Union[int, tuple], value: Union[int, float, Symbol]
    ) -> None:
        """
        Sets the item at the specified index to the given value.

        Parameters
        ----------
        index : int or tuple
            The index of the item to set.
        value : Union[int, float, Symbol]
            The value to set at the specified index.
        """
        self.matrix[index] = value % 2 if isinstance(value, (int, float)) else value
        EnsureNoTwo(matrix=self.matrix)

    def __len__(self) -> int:
        """
        Returns the number of rows in the matrix.

        Returns
        -------
        int
            The number of rows in the matrix.
        """
        return self.numRows

    def SetNumAugmented(self, numAugmented: int) -> None:
        """
        Sets the number of augmented columns in the matrix.

        Parameters
        ----------
        numAugmented : int
            The number of augmented columns.
        """
        if numAugmented < 0 or numAugmented > self.numCols - 1:

            raise ValueError("numAugmented must be between 0 and numCols - 1.")
        self.numAugmented = numAugmented

    def __getattr__(self, attr: str) -> Union[int, tuple, np.ndarray]:
        """
        Gets the attribute with the specified name.

        Parameters
        ----------
        attr : str
            The name of the attribute to retrieve.

        Returns
        -------
        Union[int, tuple, np.ndarray]
            The requested attribute value.
        """
        EnsureNoTwo(matrix=self.matrix)
        if attr in object.__getattribute__(self, "__dict__"):

            return object.__getattribute__(self, attr)

        if attr == "numRows":

            return object.__getattribute__(self, "numRows")

        elif attr == "numCols":

            return object.__getattribute__(self, "numCols")

        elif attr == "shape":

            return self.matrix.shape

        elif attr == "matrix":

            return object.__getattribute__(self, "matrix")

        else:

            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            )


def EnsureNoTwo(matrix: Union[Base2Matrix, np.ndarray]) -> None:
    """
    Ensure that there are no duplicate elements in the matrix.

    Parameters
    ----------
    matrix : Base2Matrix | np.ndarray
        The matrix to check for duplicates.

    Raises
    ------
    ValueError
        If duplicate elements are found in the matrix.
    """
    if isinstance(matrix, Base2Matrix):

        matrix_ = matrix.matrix
    else:

        matrix_ = matrix

    for row in range(matrix_.shape[0]):

        for col in range(matrix_.shape[1]):

            elem = matrix_[row, col]
            if isinstance(elem, (int, float)) and elem == 2:

                raise Exception(f"TWO HAPPENS {row}|{col} == {elem}")

        for col in range(matrix_.shape[1]):

            elem = matrix_[row, col]
            if isinstance(elem, (int, float)) and elem == 2:

                raise Exception(f"TWO HAPPENS {row}|{col} == {elem}")
