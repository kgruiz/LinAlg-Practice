import random
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix as symMatrix
from tqdm import tqdm

from basis import Basis
from determinat import Determinat
from gramSchmidt import GramSchmidt
from inverse import Inverse
from matrix import FloatMatrix, Matrix
from matrixAdd import MatrixAdd
from matrixSubtract import MatrixSubtract
from multiply import Multiply
from qrDecomposition import qrDecomposition
from rref import RREF
from scalarMultiply import ScalarMultiply
from transpose import Transpose
from unitVector import UnitVector
from vectorDot import VectorDot
from vectorLength import Length

# ============================
# Core Test Functions and Corresponding Random Test Functions
# ============================


# --- testMatrix ---
def testMatrix(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 3,
    Bcols: int = 1,
    min_: int = 0,
    max_: int = 100,
    verbose: bool = False,
):
    """
    Tests the creation of two matrices A and B with specified dimensions and value ranges.

    Args:
        Arows (int): Number of rows in matrix A.
        Acols (int): Number of columns in matrix A.
        Brows (int): Number of rows in matrix B.
        Bcols (int): Number of columns in matrix B.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrices.

    Returns:
        None
    """
    if verbose:
        print(f"Making A and B\n")

    a = Matrix(Arows, Acols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"B:\n{b}")


def randomTestMatrix(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testMatrix function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Always returns True as testMatrix does not perform validations.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = random.randint(0, maxSize)
    Bcols = random.randint(0, maxSize)

    try:
        testMatrix(
            Arows=Arows,
            Acols=Acols,
            Brows=Brows,
            Bcols=Bcols,
            min_=minSize,
            max_=maxSize,
            verbose=verbose,
        )
        return True
    except Exception as e:
        if verbose:
            print(f"Exception in randomTestMatrix: {e}")
        return False


# --- testVectorDot ---
def testVectorDot(
    Arows: int = 4,
    Acols: int = 1,
    Brows: int = 1,
    Bcols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the VectorDot operation between two vectors A and B.

    Args:
        Arows (int): Number of rows in vector A.
        Acols (int): Number of columns in vector A.
        Brows (int): Number of rows in vector B.
        Bcols (int): Number of columns in vector B.
        min_ (int): Minimum value for vector elements.
        max_ (int): Maximum value for vector elements.
        verbose (bool): If True, prints the vectors and results.

    Returns:
        bool: True if the dot product is correct, False otherwise.
    """
    a = Matrix(Arows, Acols, min_, max_)
    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"A:\n{a}")
        print(f"B:\n{b}")
        print(f"Calculating Dot Product of A and B\n")

    try:
        c = VectorDot(a, b)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:
        print(f"C:\n{c}")

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    d = np.dot(a_flat, b_flat)

    correct = c == d

    if not correct:
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestVectorDot(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testVectorDot function.

    Args:
        minSize (int): Minimum size for the vector dimensions and element values.
        maxSize (int): Maximum size for the vector dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testVectorDot function.
    """
    vectorElemCount = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:
        Arows = vectorElemCount
        Acols = 1
    else:
        Arows = 1
        Acols = vectorElemCount

    if random.randrange(0, 2) == 0:
        Brows = vectorElemCount
        Bcols = 1
    else:
        Brows = 1
        Bcols = vectorElemCount

    testResult = testVectorDot(
        Arows=Arows,
        Acols=Acols,
        Brows=Brows,
        Bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testMultiply ---
def testMultiply(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 3,
    Bcols: int = 2,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Multiply operation between two matrices A and B.

    Args:
        Arows (int): Number of rows in matrix A.
        Acols (int): Number of columns in matrix A.
        Brows (int): Number of rows in matrix B.
        Bcols (int): Number of columns in matrix B.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrices and results.

    Returns:
        bool: True if the multiplication is correct, False otherwise.
    """
    a = Matrix(Arows, Acols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"B:\n{b}")
        print(f"Multiplying A and B\n")

    try:
        c = Multiply(matrixA=a, matrixB=b)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:
        print(f"c:\n{c}")

    d = np.matmul(a.matrix, b.matrix)
    correct = c.shape == d.shape

    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if not correct:
                break
            correct = c[row][col] == d[row][col]

    if not correct:
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        print(f"c:\n{c}")
        print(f"d:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestMultiply(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testMultiply function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testMultiply function.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Acols  # Ensures multiplication is possible
    Bcols = random.randint(0, maxSize)

    testResult = testMultiply(
        Arows=Arows,
        Acols=Acols,
        Brows=Brows,
        Bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testMatrixAdd ---
def testMatrixAdd(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 4,
    Bcols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the MatrixAdd operation between two matrices A and B.

    Args:
        Arows (int): Number of rows in matrix A.
        Acols (int): Number of columns in matrix A.
        Brows (int): Number of rows in matrix B.
        Bcols (int): Number of columns in matrix B.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrices and results.

    Returns:
        bool: True if the addition is correct, False otherwise.
    """
    a = Matrix(Arows, Acols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"B:\n{b}")

    try:
        c = MatrixAdd(matrixA=a, matrixB=b)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:
        print(f"C:\n{c}")

    d = a.matrix + b.matrix
    correct = c.shape == d.shape

    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if not correct:
                break
            correct = c[row][col] == d[row][col]

    if not correct:
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestMatrixAdd(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testMatrixAdd function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testMatrixAdd function.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Arows  # Ensures addition is possible
    Bcols = Acols

    testResult = testMatrixAdd(
        Arows=Arows,
        Acols=Acols,
        Brows=Brows,
        Bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testMatrixSubtract ---
def testMatrixSubtract(
    Arows: int = 4,
    Acols: int = 3,
    Brows: int = 4,
    Bcols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the MatrixSubtract operation between two matrices A and B.

    Args:
        Arows (int): Number of rows in matrix A.
        Acols (int): Number of columns in matrix A.
        Brows (int): Number of rows in matrix B.
        Bcols (int): Number of columns in matrix B.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrices and results.

    Returns:
        bool: True if the subtraction is correct, False otherwise.
    """
    a = Matrix(Arows, Acols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    b = Matrix(Brows, Bcols, min_, max_)

    if verbose:
        print(f"B:\n{b}")

    try:
        c = MatrixSubtract(matrixA=a, matrixB=b)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:
        print(f"C:\n{c}")

    d = a.matrix - b.matrix
    correct = c.shape == d.shape

    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if not correct:
                break
            correct = c[row][col] == d[row][col]

    if not correct:
        print(f"{Arows=}\n{Acols=}\n{Brows=}\n{Bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestMatrixSubtract(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testMatrixSubtract function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testMatrixSubtract function.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Arows  # Ensures subtraction is possible
    Bcols = Acols

    testResult = testMatrixSubtract(
        Arows=Arows,
        Acols=Acols,
        Brows=Brows,
        Bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testVectorLength ---
def testVectorLength(
    rows: int = 1,
    cols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Length operation for a vector.

    Args:
        rows (int): Number of rows in the vector.
        cols (int): Number of columns in the vector.
        min_ (int): Minimum value for vector elements.
        max_ (int): Maximum value for vector elements.
        verbose (bool): If True, prints the vector and results.

    Returns:
        bool: True if the length is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    try:
        c = Length(vector=a)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        return False

    if verbose:
        print(f"c:\n{c}")

    d = np.linalg.norm(a.matrix)

    correct = c == d

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"c:\n{c}")
        print(f"d:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestVectorLength(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testVectorLength function.

    Args:
        minSize (int): Minimum size for the vector dimensions and element values.
        maxSize (int): Maximum size for the vector dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testVectorLength function.
    """
    vectorElemCount = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:
        rows = vectorElemCount
        cols = 1
    else:
        rows = 1
        cols = vectorElemCount

    testResult = testVectorLength(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testTranspose ---
def testTranspose(
    rows: int = 2,
    cols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """ """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    try:

        transA = Transpose(matrix=a)

    except Exception:

        print(f"A:\n{a}\n")
        print(f"transA:\n{transA}\n")
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        return False

    if verbose:
        print(f"transA:\n{transA}")

    c = a.matrix.T

    correct = transA.shape == c.shape

    if not correct:

        print(f"\n\nError causing components:\n")
        print(f"Shape transA: {transA.shape}")
        print(f"Shape C: {c.shape}\n")

    for row in range(transA.shape[0]):
        for col in range(transA.shape[1]):

            if not correct:

                break

            correct = abs(transA[row][col] - c[row][col]) < 0.00001

            if not correct:

                print(f"\n\nError causing components:\n")
                print(f"transA[{row}][{col}]: {transA[row][col]}")
                print(f"c[{row}][{col}]: {c[row][col]}")

                break

    if not correct:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")

        print(f"transA:\n{transA}\n")
        print(f"C:\n{c}")

        return False

    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestTranspose(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """ """

    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)

    testResult = testTranspose(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testUnitVector ---
def testUnitVector(
    rows: int = 1,
    cols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the UnitVector operation for a vector.

    Args:
        rows (int): Number of rows in the vector.
        cols (int): Number of columns in the vector.
        min_ (int): Minimum value for vector elements.
        max_ (int): Maximum value for vector elements.
        verbose (bool): If True, prints the vector and results.

    Returns:
        bool: True if the unit vector is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    try:

        c = UnitVector(vector=a)

    except Exception:

        print(f"C:\n{c}\n")
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        return False

    if verbose:

        print(f"C:\n{c}")

    if np.linalg.norm(a.matrix) == 0:

        d = a.matrix

    else:

        d = a.matrix / np.linalg.norm(a.matrix)

    correct = c.shape == d.shape

    if not correct:

        print(f"\n\nIncorrect:\n")

        print(f"c.shape: {c.shape}")
        print(f"d.shape: {d.shape}\n\n")

    if correct:

        c_flat = c.flatten()
        d_flat = d.flatten()

        for row in range(c_flat.shape[0]):

            correct = abs(c_flat[row] - d_flat[row]) < 0.0001

            if not correct:

                print(f"\n\nIncorrect Elements:\n")

                print(f"c_flat[{row}][0]: {c_flat[row][0]}")
                print(f"d_flat[{row}][0]: {d_flat[row][0]}\n\n")

                break

    if not correct:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"a:\n{a}")
        print(f"ShapeA: {a.shape}")
        print(f"LengthA Own: {Length(vector=a)}")
        print(f"LengthA numpy: {np.linalg.norm(a.matrix)}\n")
        print(f"c:\n{c}")
        print(f"d:\n{d}")

        return False

    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestUnitVector(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testUnitVector function.

    Args:
        minSize (int): Minimum size for the vector dimensions and element values.
        maxSize (int): Maximum size for the vector dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testUnitVector function.
    """
    vectorElemCount = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:
        rows = vectorElemCount
        cols = 1
    else:
        rows = 1
        cols = vectorElemCount

    testResult = testUnitVector(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testScalarMultiply ---
def testScalarMultiply(
    scalar: int | float = 5,
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the ScalarMultiply operation on a matrix.

    Args:
        scalar (int | float): Scalar value to multiply with the matrix.
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the scalar multiplication is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    try:
        c = ScalarMultiply(scalar=scalar, matrix=a)
    except Exception:
        print(f"C:\n{c}\n")
        print(f"{scalar=}\n{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        return False

    if verbose:
        print(f"C:\n{c}")

    d = scalar * a.matrix
    correct = c.shape == d.shape

    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if not correct:
                break
            correct = abs(c[row][col] - d[row][col]) < 0.00001
            if not correct:
                print(c[row][col])
                print(d[row][col])
                break

    if not correct:
        print(f"{scalar=}\n{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"A:\n{a}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestScalarMultiply(
    minSize: int = -100,
    maxSize: int = 100,
    verbose: bool = False,
) -> bool:
    """
    Generates random dimensions and scalar values to test the testScalarMultiply function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and scalar values.
        maxSize (int): Maximum size for the matrix dimensions and scalar values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testScalarMultiply function.
    """
    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)

    scalar = random.randint(minSize, maxSize)

    testResult = testScalarMultiply(
        scalar=scalar,
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testRREF ---
def testRREF(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Reduced Row Echelon Form (RREF) operation on a matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the RREF is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    rrefA = RREF(matrix=a)

    sympyRREF = symMatrix(a.matrix)
    sympyRREF = sympyRREF.rref()[0]
    sympyRREF = FloatMatrix(np.array(sympyRREF).astype(float))

    if verbose:
        print(f"rrefA:\n{rrefA}")
        print(f"sympyRREF:\n{sympyRREF}")

    correct = rrefA.shape == sympyRREF.shape

    for row in range(rrefA.shape[0]):
        for col in range(rrefA.shape[1]):
            if not correct:
                break
            correct = abs(rrefA[row][col] - sympyRREF[row][col]) < 0.0001
            if not correct:
                print(rrefA[row][col])
                print(sympyRREF[row][col])
                break

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"rrefA:\n{rrefA}")
        print(f"sympyRREF:\n{sympyRREF}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestRREF(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testRREF function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testRREF function.
    """
    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)
    augmented = 0  # Unused parameter, kept for interface consistency

    testResult = testRREF(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testBasis ---
def testBasis(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Basis operation to find a basis for the column space of a matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the basis is correct, False otherwise.
    """

    def basisOfVectors(vectors):
        """
        Returns a set of linearly independent vectors forming the basis of the input matrix.

        Parameters:
        vectors (numpy.ndarray): A 2D array where each column represents a vector.

        Returns:
        numpy.ndarray: A matrix containing the linearly independent vectors as columns.
        """
        matrix = vectors.astype(float)
        n_rows, n_cols = matrix.shape
        pivot_columns = []
        row = 0

        for col in range(n_cols):
            if row >= n_rows:
                break
            if matrix[row, col] == 0:
                for r in range(row + 1, n_rows):
                    if matrix[r, col] != 0:
                        matrix[[row, r]] = matrix[[r, row]]
                        break
            if matrix[row, col] != 0:
                pivot_columns.append(col)
                matrix[row] = matrix[row] / matrix[row, col]
                for r in range(n_rows):
                    if r != row:
                        matrix[r] -= matrix[r, col] * matrix[row]
                row += 1

        independent_vectors = vectors[:, pivot_columns]

        return independent_vectors

    a = Matrix(rows, cols, min_, max_)

    a = FloatMatrix(np.array([[-6, -88, -31, -7], [-6, -88, -12, -13]]))

    if verbose:
        print(f"A:\n{a}")
    basisA = Basis(span=a)
    c = basisOfVectors(a.matrix)

    if verbose:

        print(f"basisA:\n{basisA}")
        print(f"C:\n{c}")

    correct = basisA.shape == c.shape

    if (
        basisA.shape[0] == 1
        and basisA.shape[1] == 1
        and c.shape[0] == 1
        and c.shape[1] == 1
    ):

        if c[0][0] == 0 and a[0][0] == 0:

            return True

    if not correct:

        print(f"\n\nError causing components:\n")
        print(f"Shape basisA: {basisA.shape}")
        print(f"Shape C: {c.shape}\n")

    for row in range(basisA.shape[0]):
        for col in range(basisA.shape[1]):

            if not correct:

                break

            correct = abs(basisA[row][col] - c[row][col]) < 0.00001

            if not correct:

                print(f"\n\nError causing components:\n")
                print(f"basisA: {basisA[row][col]}")
                print(f"C: {c[row][col]}")

                break

    if not correct:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")

        print(f"basisA:\n{basisA}\n")
        print(f"C:\n{c}")

        return False

    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestBasis(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testBasis function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testBasis function.
    """
    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = testBasis(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testQRDecomposition ---
def testQRDecomposition(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """ """

    a = Matrix(rows, cols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

    q, r = qrDecomposition(span=a)

    basisA = Basis(span=a)

    q2, r2 = np.linalg.qr(basisA)

    try:

        if verbose:

            print(f"basisA:\n{basisA}\n")

            print(f"q:\n{q}\n")
            print(f"q2:\n{FloatMatrix(q2)}")
            print(f"r:\n{r}\n")
            print(f"r2:\n{FloatMatrix(r2)}")

        # check q

        correct = q.shape == q2.shape

        if not correct:

            print(f"\n\nError causing components:\n")
            print(f"Shape r: {r.shape}")
            print(f"Shape r2: {r2.shape}\n")

            print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

            print(f"A:\n{a}\n")

            print(f"basisA:\n{basisA}\n")

            print(f"q:\n{q}\n")
            print(f"q2:\n{FloatMatrix(q2)}")
            print(f"r:\n{r}\n")
            print(f"r2:\n{FloatMatrix(r2)}")

            return False

        for colNum in range(q.numCols):

            firstNonZeroIndex = None

            for row in range(q.numRows):

                if abs(q[row][colNum] - 0) > 0.000001:

                    firstNonZeroIndex = row

                    break

            if (
                q[firstNonZeroIndex][colNum] > 0.000001
                and q2[firstNonZeroIndex][colNum] < -0.000001
            ) or (
                q[firstNonZeroIndex][colNum] < -0.000001
                and q2[firstNonZeroIndex][colNum] > 0.000001
            ):

                for row in range(q.numRows):

                    if abs(q[row][colNum] - 0) > 0.000001:

                        q[row][colNum] = -q[row][colNum]

        for row in range(q.shape[0]):
            for col in range(q.shape[1]):

                if not correct:

                    break

                correct = abs(q[row][col] - q2[row][col]) < 0.00001

                if not correct:

                    print(f"\n\nError causing components:\n")
                    print(f"q[{row}][{col}]: {q[row][col]}")
                    print(f"q2[{row}][{col}]: {q2[row][col]}")

                    break

        if not correct:

            print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

            print(f"A:\n{a}\n")

            print(f"basisA:\n{basisA}\n")

            print(f"q:\n{q}\n")
            print(f"q2:\n{FloatMatrix(q2)}")
            print(f"r:\n{r}\n")
            print(f"r2:\n{FloatMatrix(r2)}")

            return False

        # check r

        correct = r.shape == r2.shape

        if not correct:

            print(f"\n\nError causing components:\n")
            print(f"Shape r: {r.shape}")
            print(f"Shape r2: {r2.shape}\n")

            print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

            print(f"A:\n{a}\n")

            print(f"basisA:\n{basisA}\n")

            print(f"q:\n{q}\n")
            print(f"q2:\n{FloatMatrix(q2)}")
            print(f"r:\n{r}\n")
            print(f"r2:\n{FloatMatrix(r2)}")

            return False

        for rowNum in range(r.numRows):

            firstNonZeroIndex = None

            for col in range(r.numCols):

                if abs(r[rowNum][col] - 0) > 0.000001:

                    firstNonZeroIndex = col

                    break

            if (
                r[rowNum][firstNonZeroIndex] > 0.000001
                and r2[rowNum][firstNonZeroIndex] < -0.000001
            ) or (
                r[rowNum][firstNonZeroIndex] < -0.000001
                and r2[rowNum][firstNonZeroIndex] > 0.000001
            ):

                for col in range(r.numCols):

                    if abs(r[rowNum][col] - 0) > 0.000001:

                        r[rowNum][col] = -r[rowNum][col]

        for row in range(r.shape[0]):
            for col in range(r.shape[1]):

                if not correct:

                    break

                correct = abs(r[row][col] - r2[row][col]) < 0.00001

                if not correct:

                    print(f"\n\nError causing components:\n")
                    print(f"r[{row}][{col}]: {r[row][col]}")
                    print(f"r2[{row}][{col}]: {r2[row][col]}")

                    break

        if not correct:

            print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

            print(f"A:\n{a}\n")

            print(f"basisA:\n{basisA}\n")

            print(f"q:\n{q}\n")
            print(f"q2:\n{FloatMatrix(q2)}")
            print(f"r:\n{r}\n")
            print(f"r2:\n{FloatMatrix(r2)}")

            return False

        else:
            if verbose:
                print(f"Correct Shape\nCorrect Elements")
            return True

    except Exception:

        print()
        print()
        traceback.print_exc()
        print()

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")

        print(f"basisA:\n{basisA}\n")

        print(f"q:\n{q}\n")
        print(f"q2:\n{FloatMatrix(q2)}")
        print(f"r:\n{r}\n")
        print(f"r2:\n{FloatMatrix(r2)}")

        return False


def randomTestQRDecomposition(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """ """
    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = testQRDecomposition(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testGramSchmidtRandomSpan ---
def testGramSchmidtRandomSpan(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Gram Schmidt operation to find an orthonormal basis from an input basis.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the orthonormal basis is correct, False otherwise.
    """

    a = Matrix(rows, cols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

    orthonormalA = GramSchmidt(span=a)
    c, R = np.linalg.qr(a)

    if verbose:

        print(f"A:\n{a}")
        print(f"C:\n{c}")

    correct = orthonormalA.shape == c.shape

    if not correct:

        print(f"\n\nError causing components:\n")
        print(f"Shape orthnormalA: {orthonormalA.shape}")
        print(f"Shape C: {c.shape}\n")

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")

        print(f"orthonormalA:\n{orthonormalA}\n")
        print(f"FloatMatrix(c):\n{FloatMatrix(c)}")
        print(f"C:\n{c}")

        return False

    for colNum in range(orthonormalA.numCols):

        firstNonZeroIndex = None

        for row in range(orthonormalA.numRows):

            if abs(orthonormalA[row][colNum] - 0) > 0.0001:

                firstNonZeroIndex = row

                break

        if (
            orthonormalA[firstNonZeroIndex][colNum] > 0.0001
            and c[firstNonZeroIndex][colNum] < -0.0001
        ) or (
            orthonormalA[firstNonZeroIndex][colNum] < -0.0001
            and c[firstNonZeroIndex][colNum] > 0.0001
        ):

            for row in range(orthonormalA.numRows):

                orthonormalA[row][colNum] = -orthonormalA[row][colNum]

    for row in range(orthonormalA.shape[0]):
        for col in range(orthonormalA.shape[1]):

            if not correct:

                break

            correct = abs(orthonormalA[row][col] - c[row][col]) < 0.00001

            if not correct:

                print(f"\n\nError causing components:\n")
                print(f"orthonormalA[{row}][{col}]: {orthonormalA[row][col]}")
                print(f"c[{row}][{col}]: {c[row][col]}")

                break

    if not correct:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")

        print(f"orthonormalA:\n{orthonormalA}\n")
        print(f"FloatMatrix(c):\n{FloatMatrix(c)}")
        print(f"C:\n{c}")

        return False

    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestGramSchmidtRandomSpan(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testGramSchmidt function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testBasis function.
    """
    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = testGramSchmidt(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testGramSchmidt ---
def testGramSchmidt(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Gram Schmidt operation to find an orthonormal basis from an input basis.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the orthonormal basis is correct, False otherwise.
    """

    a = Matrix(rows, cols, min_, max_)

    while np.count_nonzero(a.matrix) == 0:

        a = Matrix(rows, cols, min_, max_)

    basisA = Basis(span=a)

    if verbose:

        print(f"A:\n{a}")

    try:

        orthonormalA = GramSchmidt(span=basisA)

    except IndexError:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")
        print(f"BasisA:\n{basisA}\n")

        return False

    c, R = np.linalg.qr(basisA)

    if verbose:

        print(f"basisA:\n{basisA}")
        print(f"C:\n{c}")

    correct = orthonormalA.shape == c.shape

    if not correct:

        print(f"\n\nError causing components:\n")
        print(f"Shape orthnormalA: {orthonormalA.shape}")
        print(f"Shape C: {c.shape}\n")

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")
        print(f"BasisA:\n{basisA}\n")

        print(f"orthonormalA:\n{orthonormalA}\n")
        print(f"FloatMatrix(c):\n{FloatMatrix(c)}")
        print(f"C:\n{c}")

        return False

    for colNum in range(orthonormalA.numCols):

        firstNonZeroIndex = None

        for row in range(orthonormalA.numRows):

            if abs(orthonormalA[row][colNum] - 0) > 0.0001:

                firstNonZeroIndex = row

                break

        if (
            orthonormalA[firstNonZeroIndex][colNum] > 0.0001
            and c[firstNonZeroIndex][colNum] < -0.0001
        ) or (
            orthonormalA[firstNonZeroIndex][colNum] < -0.0001
            and c[firstNonZeroIndex][colNum] > 0.0001
        ):

            for row in range(orthonormalA.numRows):

                orthonormalA[row][colNum] = -orthonormalA[row][colNum]

    for row in range(orthonormalA.shape[0]):
        for col in range(orthonormalA.shape[1]):

            if not correct:

                break

            correct = abs(orthonormalA[row][col] - c[row][col]) < 0.00001

            if not correct:

                print(f"\n\nError causing components:\n")
                print(f"orthonormalA[{row}][{col}]: {orthonormalA[row][col]}")
                print(f"c[{row}][{col}]: {c[row][col]}")

                break

    if not correct:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")
        print(f"BasisA:\n{basisA}\n")

        print(f"orthonormalA:\n{orthonormalA}\n")
        print(f"FloatMatrix(c):\n{FloatMatrix(c)}")
        print(f"C:\n{c}")

        return False

    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")

        return True


def randomTestGramSchmidt(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random dimensions and values to test the testGramSchmidt function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testBasis function.
    """
    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = testGramSchmidt(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- testDeterminant ---
def testDeterminant(
    rows: int = 4,
    cols: int = 4,
    min_: int = -10,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Determinant operation on a square matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the determinant is correct, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    detA = Determinat(matrix=a)
    b = np.linalg.det(a.matrix)

    if verbose:
        print(f"detA:\n{detA}")
        print(f"B:\n{b}")

    if abs(detA) > 100000000:
        correct = abs((detA - b) / b) < 0.000001
    else:
        correct = abs(detA - b) < 0.00001

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"A:\n{a}")
        print(f"detA:\n{detA}")
        print(f"B:\n{b}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestDeterminant(
    minSize: int = -100, maxSize: int = 12, maxVal: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random square dimensions and values to test the testDeterminant function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions.
        maxVal (int): Maximum absolute value for matrix elements.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testDeterminant function.
    """
    rows = random.randint(0, maxSize)
    cols = rows  # Ensures the matrix is square

    testResult = testDeterminant(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxVal,
        verbose=verbose,
    )

    return testResult


# --- testInverse ---
def testInverse(
    rows: int = 4,
    cols: int = 4,
    min_: int = -10,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Inverse operation on a square matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the inverse is correct or None if the matrix is singular, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    if verbose:
        print(f"A:\n{a}")

    invA = Inverse(matrix=a)

    if Determinat(a) == 0:
        return invA is None

    try:
        c = np.linalg.inv(a.matrix)
    except np.linalg.LinAlgError:
        if verbose:
            print("Matrix is singular and cannot be inverted.")
        return invA is None

    if verbose:
        print(f"invA:\n{invA}")
        print(f"C:\n{c}")

    correct = invA.shape == c.shape

    for row in range(invA.shape[0]):
        for col in range(invA.shape[1]):
            if not correct:
                break
            correct = abs(invA[row][col] - c[row][col]) < 0.00001
            if not correct:
                print(invA[row][col])
                print(c[row][col])
                break

    if not correct:
        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"invA:\n{invA}")
        print(f"C:\n{c}")
        return False
    else:
        if verbose:
            print(f"Correct Shape\nCorrect Elements")
        return True


def randomTestInverse(
    minSize: int = -100, maxSize: int = 100, maxVal: int = 100, verbose: bool = False
) -> bool:
    """
    Generates random square dimensions and values to test the testInverse function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions.
        maxVal (int): Maximum absolute value for matrix elements.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the testInverse function.
    """
    rows = random.randint(0, maxSize)
    cols = rows  # Ensures the matrix is square

    testResult = testInverse(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxVal,
        verbose=verbose,
    )

    return testResult


# ============================
# Utility Functions
# ============================


def calculateTimeComplexity(
    func,
    minSize: int = -100,
    maxSize: int = 100,
    maxVal: int = 100,
    verbose: bool = True,
):
    """
    Measures the execution time of a given function for different input sizes.

    Args:
        func: The function to be tested, which accepts rows, cols, min_, and max_ as parameters.
        minSize (int): Minimum size for the input (default -100).
        maxSize (int): Maximum size for the input (default 100).
        maxVal (int): Maximum absolute value for the input values (default 100).
        verbose (bool): If True, prints execution details and plots results (default True).

    Returns:
        None
    """
    runs = []
    if verbose:
        print(f"Function: {func.__name__}")

    for size in range(maxSize):
        startTime = time.time()

        if (
            func.__name__ == "randomTestInverse"
            or func.__name__ == "randomTestDeterminant"
        ):
            testResult = func(minSize=-maxVal, maxSize=maxVal, verbose=False)
        else:
            testResult = func(minSize=minSize, maxSize=maxSize, verbose=False)

        endTime = time.time()

        if not testResult:
            print(f"{func.__name__}")
            break

        runs.append((size, endTime - startTime))

        if verbose:
            print(f"\nSize: {size}\n\tTime: {endTime - startTime} s")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sizes, times = zip(*runs) if runs else ([], [])
    ax.plot(sizes, times)
    ax.set_xlabel("Input Size")
    ax.set_ylabel("Execution Time (s)")

    plt.savefig(f"{func.__name__}Complexity.png")

    if verbose and runs:
        plt.show()


# ============================
# Main Execution Block
# ============================

if __name__ == "__main__":

    minSize = -100
    maxSize = 100

    noErrors = True

    # Test qrDecomposition
    if noErrors:

        for _ in tqdm(range(1000), desc="qrDecomposition"):
            testResult = randomTestQRDecomposition(
                minSize=minSize, maxSize=15, verbose=False
            )

            if not testResult:

                print("qrDecomposition")
                noErrors = False

                break

    # Test Basis
    if noErrors:

        for _ in tqdm(range(1000), desc="testBasis"):
            testResult = randomTestBasis(minSize=minSize, maxSize=15, verbose=False)

            if not testResult:

                print("Basis")
                noErrors = False

                break

    # Test Transpose
    if noErrors:

        for _ in tqdm(range(1000), desc="testTranspose"):
            testResult = randomTestTranspose(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("Transpose")
                noErrors = False
                break

    # Test GramSchmidt
    if noErrors:

        for _ in tqdm(range(1000), desc="testGramSchmidt"):
            testResult = randomTestGramSchmidt(
                minSize=minSize, maxSize=15, verbose=False
            )
            if not testResult:
                print("GramSchmidt")
                noErrors = False
                break

    # Test GramSchmidtRandomSpan
    if noErrors:

        for _ in tqdm(range(1000), desc="testGramSchmidtRandomSpan"):
            testResult = randomTestGramSchmidtRandomSpan(
                minSize=minSize, maxSize=15, verbose=False
            )
            if not testResult:
                print("GramSchmidtRandomSpan")
                noErrors = False
                break

    # Test UnitVector
    if noErrors:

        for _ in tqdm(range(10000), desc="testUnitVector"):
            testResult = randomTestUnitVector(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("UnitVector")
                noErrors = False
                break

    # Test Inverse
    if noErrors:

        for _ in tqdm(range(50), desc="testInverse"):
            testResult = randomTestInverse(
                minSize=minSize, maxSize=8, maxVal=maxSize, verbose=False
            )
            if not testResult:
                print("Inverse")
                noErrors = False
                break

    # Test Determinant
    if noErrors:

        for _ in tqdm(range(100), desc="testDeterminant"):
            testResult = randomTestDeterminant(
                minSize=minSize, maxSize=8, maxVal=maxSize, verbose=False
            )
            if not testResult:
                print("Determinant")
                noErrors = False
                break

    # Test VectorDot
    if noErrors:

        for _ in tqdm(range(1000), desc="testVectorDot"):
            testResult = randomTestVectorDot(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("VectorDot")
                noErrors = False
                break

    # Test Multiply
    if noErrors:

        for _ in tqdm(range(10), desc="testMultiply"):
            testResult = randomTestMultiply(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("Multiply")
                noErrors = False
                break

    # Test MatrixAdd
    if noErrors:

        for _ in tqdm(range(100), desc="testMatrixAdd"):
            testResult = randomTestMatrixAdd(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("MatrixAdd")
                noErrors = False
                break

    # Test MatrixSubtract
    if noErrors:

        for _ in tqdm(range(100), desc="testMatrixSubtract"):
            testResult = randomTestMatrixSubtract(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("MatrixSubtract")
                noErrors = False
                break

    # Test ScalarMultiply
    if noErrors:

        for _ in tqdm(range(100), desc="testScalarMultiply"):
            testResult = randomTestScalarMultiply(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("ScalarMultiply")
                noErrors = False
                break

    # Test VectorLength
    if noErrors:

        for _ in tqdm(range(1000), desc="testVectorLength"):
            testResult = randomTestVectorLength(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:
                print("VectorLength")
                noErrors = False
                break

    # Test RREF
    if noErrors:

        for _ in tqdm(range(25), desc="testRREF"):
            testResult = randomTestRREF(minSize=minSize, maxSize=maxSize, verbose=False)
            if not testResult:
                print("RREF")
                noErrors = False
                break
