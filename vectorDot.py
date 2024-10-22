import numpy as np

from matrix import Matrix


def VectorDot(matrixA: Matrix, matrixB: Matrix) -> int | str:

    if not (
        (matrixA.numCols == 1 or matrixA.numRows == 1)
        and (matrixB.numCols == 1 or matrixB.numRows == 1)
    ):

        raise Exception(
            f"Cannot perform dot product. One of the input matrices is not a vector.\nShape A: {matrixA.numRows} x {matrixA.numCols}\nShape B: {matrixB.numRows} x {matrixB.numCols}"
        )

    lengthA = max(matrixA.numRows, matrixA.numCols)
    lengthB = max(matrixB.numRows, matrixB.numCols)

    if lengthA != lengthB:

        raise Exception(
            f"Cannot perform dot product.\nMatrices A and B have different lengths.\nLength A: {lengthA}\nLength B: {lengthB}"
        )

    if matrixA.shape[0] != 1:

        matrixA = matrixA.T

    if matrixB.shape[0] != 1:

        matrixB = matrixB.T

    dotProduct = 0

    for elemA, elemB in zip(matrixA[0], matrixB[0]):

        dotProduct += elemA * elemB

    return dotProduct
