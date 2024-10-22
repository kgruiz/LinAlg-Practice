import random

import numpy as np


class Matrix:

    def __init__(self, *args):

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

                self.matrix = np.reshape(args[0], newshape=(args[0].shape[0], 1))

            else:

                self.matrix = args[0]

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
                "Invalid arguments. Pass either (rows, cols), a numpy array, or (rows, cols, min, max)."
            )

    def __str__(self):

        return MatrixString(self.matrix)

    def __getitem__(self, index):

        return self.matrix[index]

    def __setitem__(self, index, value):

        self.matrix[index] = value

    def __getattr__(self, attr):

        if attr == "numRows":

            return self.numRows

        elif attr == "numCols":

            return self.numCols

        elif attr == "matrix":

            return self.matrix

        else:

            return getattr(self.matrix, attr)


def MatrixString(matrix: np.ndarray) -> str:

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

    def __init__(self, *args):

        if len(args) == 2:

            rows, cols = args
            self.matrix = np.random.uniform(0, 100, (rows, cols))
            self.matrix.dtype = float

            self.numRows = rows
            self.numCols = cols

        elif len(args) == 1:

            if isinstance(args[0], np.ndarray):

                if len(args[0].shape) == 1:

                    self.matrix = np.reshape(args[0], newshape=(args[0].shape[0], 1))
                    self.matrix.dtype = float

                else:

                    self.matrix = args[0]
                    self.matrix.dtype = float

                self.numRows = self.matrix.shape[0]
                self.numCols = self.matrix.shape[1]

            elif isinstance(args[0], Matrix):

                self.matrix = args[0].matrix.astype(float)
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

        return FloatMatrixString(self.matrix)

    def __getitem__(self, index):

        return self.matrix[index]

    def __setitem__(self, index, value):

        self.matrix[index] = value

    def __getattr__(self, attr):

        if attr == "numRows":

            return self.numRows

        elif attr == "numCols":

            return self.numCols

        elif attr == "matrix":

            return self.matrix

        else:

            return getattr(self.matrix, attr)


def FloatMatrixString(matrix: np.ndarray) -> str:

    colWidths = [
        max(len(f"{val:.2f}") for val in matrix[:, col])
        for col in range(matrix.shape[1])
    ]

    formattedMatrix = ""

    for row in matrix:

        rowStr = " ".join(f"{val:.2f}".rjust(colWidths[i]) for i, val in enumerate(row))
        formattedMatrix += f"| {rowStr} |\n"

    return formattedMatrix
