from matrix import Matrix
from vectorDot import VectorDot


def Multiply(matrixA: Matrix, matrixB: Matrix) -> Matrix | str:

    if matrixA.numCols != matrixB.numRows:

        return f"Invalid shape for input matrices. Matrix A Columns: {matrixA.numCols} != Matrix B Rows: {matrixB.numRows}"

    resultMatrix = Matrix(
        matrixA.numRows,
        matrixB.numCols,
    )

    for row in range(resultMatrix.numRows):

        for col in range(resultMatrix.numCols):

            newElem = VectorDot(
                matrixA=Matrix(matrixA[row, :]), matrixB=Matrix(matrixB[:, col])
            )

            resultMatrix[row][col] = newElem

    return resultMatrix
