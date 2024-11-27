import random
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix as symMatrix
from sympy import symbols
from tqdm import tqdm

from Base2RREF import Base2RREF
from Basis import Basis
from Determinat import Determinat
from GramSchmidt import GramSchmidt
from Inverse import Inverse
from Matrix import FloatMatrix, Matrix
from MatrixAdd import MatrixAdd
from MatrixMultiply import MatrixMultiply
from MatrixSubtract import MatrixSubtract
from QRDecomposition import QRDecomposition
from RREF import RREF
from ScalarMultiply import ScalarMultiply
from Transpose import Transpose
from UnitVector import UnitVector
from VectorDot import VectorDot
from VectorLength import Length

# ============================
# Core Test Functions and Corresponding Random Test Functions
# ============================


# --- TestMatrix ---
def TestMatrix(
    arows: int = 4,
    acols: int = 3,
    brows: int = 3,
    bcols: int = 1,
    min_: int = 0,
    max_: int = 100,
    verbose: bool = False,
) -> None:
    """
    Test matrix operations with specified dimensions and value ranges.

    Parameters
    ----------
    arows : int, optional
        Number of rows for matrix A. Default is 4.
    acols : int, optional
        Number of columns for matrix A. Default is 3.
    brows : int, optional
        Number of rows for matrix B. Default is 3.
    bcols : int, optional
        Number of columns for matrix B. Default is 1.
    min_ : int, optional
        Minimum value for matrix entries. Default is 0.
    max_ : int, optional
        Maximum value for matrix entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    None
    """
    if verbose:

        print(f"Making A and B\n")

    a = Matrix(arows, acols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

    b = Matrix(brows, bcols, min_, max_)

    if verbose:

        print(f"B:\n{b}")


def RandomTestMatrix(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Perform a random test on matrix operations within specified size ranges.

    Parameters
    ----------
    minSize : int, optional
        Minimum size for matrix entries. Default is -100.
    maxSize : int, optional
        Maximum size for matrix entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = random.randint(0, maxSize)
    Bcols = random.randint(0, maxSize)

    try:

        TestMatrix(
            arows=Arows,
            acols=Acols,
            brows=Brows,
            bcols=Bcols,
            min_=minSize,
            max_=maxSize,
            verbose=verbose,
        )
        return True

    except Exception as e:

        if verbose:

            print(f"Exception in RandomTestMatrix: {e}")
        return False


# --- TestVectorDot ---
def TestVectorDot(
    arows: int = 4,
    acols: int = 1,
    brows: int = 1,
    bcols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Test the dot product of vectors with specified dimensions and value ranges.

    Parameters
    ----------
    arows : int, optional
        Number of rows for vector A. Default is 4.
    acols : int, optional
        Number of columns for vector A. Default is 1.
    brows : int, optional
        Number of rows for vector B. Default is 1.
    bcols : int, optional
        Number of columns for vector B. Default is 4.
    min_ : int, optional
        Minimum value for vector entries. Default is 0.
    max_ : int, optional
        Maximum value for vector entries. Default is 10.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    a = Matrix(arows, acols, min_, max_)
    b = Matrix(brows, bcols, min_, max_)

    if verbose:

        print(f"A:\n{a}")
        print(f"B:\n{b}")
        print(f"Calculating Dot Product of A and B\n")

    try:

        c = VectorDot(a, b)
    except Exception:

        print(f"C:\n{c}\n")
        print(f"{arows=}\n{acols=}\n{brows=}\n{bcols=}\n{min_=}\n{max_=}")
        return False

    if verbose:

        print(f"C:\n{c}")

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    d = np.dot(a_flat, b_flat)

    correct = c == d

    if not correct:

        print(f"{arows=}\n{acols=}\n{brows=}\n{bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")
        return True


def RandomTestVectorDot(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Perform a random test on vector dot products within specified size ranges.

    Parameters
    ----------
    minSize : int, optional
        Minimum size for vector entries. Default is -100.
    maxSize : int, optional
        Maximum size for vector entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
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

    testResult = TestVectorDot(
        arows=Arows,
        acols=Acols,
        brows=Brows,
        bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestMultiply ---
def TestMultiply(
    arows: int = 4,
    acols: int = 3,
    brows: int = 3,
    bcols: int = 2,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Test matrix multiplication with specified dimensions and value ranges.

    Parameters
    ----------
    arows : int, optional
        Number of rows for matrix A. Default is 4.
    acols : int, optional
        Number of columns for matrix A. Default is 3.
    brows : int, optional
        Number of rows for matrix B. Default is 3.
    bcols : int, optional
        Number of columns for matrix B. Default is 2.
    min_ : int, optional
        Minimum value for matrix entries. Default is 0.
    max_ : int, optional
        Maximum value for matrix entries. Default is 10.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    a = Matrix(arows, acols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

    b = Matrix(brows, bcols, min_, max_)

    if verbose:

        print(f"B:\n{b}")
        print(f"Multiplying A and B\n")

    try:

        c = MatrixMultiply(matrixA=a, matrixB=b)
    except Exception:

        print(f"C:\n{c}\n")
        print(f"{arows=}\n{acols=}\n{brows=}\n{bcols=}\n{min_=}\n{max_=}")
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

        print(f"{arows=}\n{acols=}\n{brows=}\n{bcols=}\n{min_=}\n{max_=}")
        print(f"c:\n{c}")
        print(f"d:\n{d}")
        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")
        return True


def RandomTestMultiply(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Perform a random test on matrix multiplication within specified size ranges.

    Parameters
    ----------
    minSize : int, optional
        Minimum size for matrix entries. Default is -100.
    maxSize : int, optional
        Maximum size for matrix entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Acols  # Ensures multiplication is possible
    Bcols = random.randint(0, maxSize)

    testResult = TestMultiply(
        arows=Arows,
        acols=Acols,
        brows=Brows,
        bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestMatrixAdd ---
def TestMatrixAdd(
    arows: int = 4,
    acols: int = 3,
    brows: int = 4,
    bcols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Test matrix addition with specified dimensions and value ranges.

    Parameters
    ----------
    arows : int, optional
        Number of rows for matrix A. Default is 4.
    acols : int, optional
        Number of columns for matrix A. Default is 3.
    brows : int, optional
        Number of rows for matrix B. Default is 4.
    bcols : int, optional
        Number of columns for matrix B. Default is 3.
    min_ : int, optional
        Minimum value for matrix entries. Default is 0.
    max_ : int, optional
        Maximum value for matrix entries. Default is 10.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    a = Matrix(arows, acols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

    b = Matrix(brows, bcols, min_, max_)

    if verbose:

        print(f"B:\n{b}")

    try:

        c = MatrixAdd(matrixA=a, matrixB=b)
    except Exception:

        print(f"C:\n{c}\n")
        print(f"{arows=}\n{acols=}\n{brows=}\n{bcols=}\n{min_=}\n{max_=}")
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

        print(f"{arows=}\n{acols=}\n{brows=}\n{bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")
        return True


def RandomTestMatrixAdd(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Perform a random test on matrix addition within specified size ranges.

    Parameters
    ----------
    minSize : int, optional
        Minimum size for matrix entries. Default is -100.
    maxSize : int, optional
        Maximum size for matrix entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Arows  # Ensures addition is possible
    Bcols = Acols

    testResult = TestMatrixAdd(
        arows=Arows,
        acols=Acols,
        brows=Brows,
        bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestMatrixSubtract ---
def TestMatrixSubtract(
    arows: int = 4,
    acols: int = 3,
    brows: int = 4,
    bcols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Test matrix subtraction with specified dimensions and value ranges.

    Parameters
    ----------
    arows : int, optional
        Number of rows for matrix A. Default is 4.
    acols : int, optional
        Number of columns for matrix A. Default is 3.
    brows : int, optional
        Number of rows for matrix B. Default is 4.
    bcols : int, optional
        Number of columns for matrix B. Default is 3.
    min_ : int, optional
        Minimum value for matrix entries. Default is 0.
    max_ : int, optional
        Maximum value for matrix entries. Default is 10.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    a = Matrix(arows, acols, min_, max_)

    if verbose:

        print(f"A:\n{a}")

    b = Matrix(brows, bcols, min_, max_)

    if verbose:

        print(f"B:\n{b}")

    try:

        c = MatrixSubtract(matrixA=a, matrixB=b)
    except Exception:

        print(f"C:\n{c}\n")
        print(f"{arows=}\n{acols=}\n{brows=}\n{bcols=}\n{min_=}\n{max_=}")
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

        print(f"{arows=}\n{acols=}\n{brows=}\n{bcols=}\n{min_=}\n{max_=}")
        print(f"C:\n{c}")
        print(f"D:\n{d}")
        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")
        return True


def RandomTestMatrixSubtract(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Perform a random test on matrix subtraction within specified size ranges.

    Parameters
    ----------
    minSize : int, optional
        Minimum size for matrix entries. Default is -100.
    maxSize : int, optional
        Maximum size for matrix entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    Arows = random.randint(0, maxSize)
    Acols = random.randint(0, maxSize)
    Brows = Arows  # Ensures subtraction is possible
    Bcols = Acols

    testResult = TestMatrixSubtract(
        arows=Arows,
        acols=Acols,
        brows=Brows,
        bcols=Bcols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestVectorLength ---
def TestVectorLength(
    rows: int = 1,
    cols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Test the computation of vector lengths with specified dimensions and value ranges.

    Parameters
    ----------
    rows : int, optional
        Number of rows for the vector. Default is 1.
    cols : int, optional
        Number of columns for the vector. Default is 4.
    min_ : int, optional
        Minimum value for vector entries. Default is 0.
    max_ : int, optional
        Maximum value for vector entries. Default is 10.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
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


def RandomTestVectorLength(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Perform a random test on vector length computations within specified size ranges.

    Parameters
    ----------
    minSize : int, optional
        Minimum size for vector entries. Default is -100.
    maxSize : int, optional
        Maximum size for vector entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    vectorElemCount = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:

        rows = vectorElemCount
        cols = 1
    else:

        rows = 1
        cols = vectorElemCount

    testResult = TestVectorLength(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestTranspose ---
def TestTranspose(
    rows: int = 2,
    cols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Test matrix transpose with specified dimensions and value ranges.

    Parameters
    ----------
    rows : int, optional
        Number of rows for the matrix. Default is 2.
    cols : int, optional
        Number of columns for the matrix. Default is 4.
    min_ : int, optional
        Minimum value for matrix entries. Default is 0.
    max_ : int, optional
        Maximum value for matrix entries. Default is 10.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
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


def RandomTestTranspose(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Perform a random test on matrix transpositions within specified size ranges.

    Parameters
    ----------
    minSize : int, optional
        Minimum size for matrix entries. Default is -100.
    maxSize : int, optional
        Maximum size for matrix entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)

    testResult = TestTranspose(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestUnitVector ---
def TestUnitVector(
    rows: int = 1,
    cols: int = 4,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Test the creation of unit vectors with specified dimensions and value ranges.

    Parameters
    ----------
    rows : int, optional
        Number of rows for the vector. Default is 1.
    cols : int, optional
        Number of columns for the vector. Default is 4.
    min_ : int, optional
        Minimum value for vector entries. Default is 0.
    max_ : int, optional
        Maximum value for vector entries. Default is 10.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
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


def RandomTestUnitVector(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Perform a random test on unit vector creation within specified size ranges.

    Parameters
    ----------
    minSize : int, optional
        Minimum size for vector entries. Default is -100.
    maxSize : int, optional
        Maximum size for vector entries. Default is 100.
    verbose : bool, optional
        If True, prints detailed information. Default is False.

    Returns
    -------
    bool
        True if the test passes, False otherwise.
    """
    vectorElemCount = random.randint(0, maxSize)

    if random.randrange(0, 2) == 0:

        rows = vectorElemCount
        cols = 1
    else:

        rows = 1
        cols = vectorElemCount

    testResult = TestUnitVector(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestScalarMultiply ---
def TestScalarMultiply(
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
        scalar (int | float): Scalar value to Multiply with the matrix.
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


def RandomTestScalarMultiply(
    minSize: int = -100,
    maxSize: int = 100,
    verbose: bool = False,
) -> bool:
    """
    Generates Random dimensions and scalar values to test the TestScalarMultiply function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and scalar values.
        maxSize (int): Maximum size for the matrix dimensions and scalar values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the TestScalarMultiply function.
    """
    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)

    scalar = random.randint(minSize, maxSize)

    testResult = TestScalarMultiply(
        scalar=scalar,
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestRREF ---
def TestRREF(
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


def RandomTestRREF(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates Random dimensions and values to test the TestRREF function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the TestRREF function.
    """
    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)
    augmented = 0  # Unused parameter, kept for interface consistency

    testResult = TestRREF(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestBase2RREF ---
def TestBase2RREF(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 1,
    verbose: bool = False,
) -> bool:
    """
    Tests the Base2RREF function.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value in the matrix.
        max_ (int): Maximum value in the matrix.
        verbose (bool): If True, prints detailed information.

    Returns:
        bool: True if the test passes, False otherwise.
    """
    a = Matrix(rows, cols, min_, max_)

    print(f"A:\n{a}")

    if verbose:

        print(f"A:\n{a}")

    Base2RREFA = Base2RREF(matrix=a)

    print(f"Base2RREFA:\n{Base2RREFA}")

    print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

    sympyRREF = symMatrix(a.matrix)
    sympyRREF = sympyRREF.rref()[0]
    sympyRREF = FloatMatrix(np.array(sympyRREF).astype(float))

    if verbose:

        print(f"rrefA:\n{Base2RREFA}")
        print(f"sympyRREF:\n{sympyRREF}")

    correct = Base2RREFA.shape == sympyRREF.shape

    for row in range(Base2RREFA.shape[0]):

        for col in range(Base2RREFA.shape[1]):

            if not correct:

                break

            correct = abs(Base2RREFA[row][col] - sympyRREF[row][col]) < 0.0001
            if not correct:

                print(Base2RREFA[row][col])
                print(sympyRREF[row][col])
                break

    if not correct:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")
        print(f"rrefA:\n{Base2RREFA}")
        print(f"sympyRREF:\n{sympyRREF}")
        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")
        return True


def RandomTestBase2RREF(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Performs a random test for Base2RREF.

    Args:
        minSize (int): Minimum size of the matrix.
        maxSize (int): Maximum size of the matrix.
        verbose (bool): If True, prints detailed information.

    Returns:
        bool: True if the test passes, False otherwise.
    """
    rows = random.randint(0, maxSize)
    cols = random.randint(0, maxSize)
    augmented = 0  # Unused parameter, kept for interface consistency

    testResult = TestBase2RREF(
        rows=rows,
        cols=cols,
        min_=0,
        max_=1,
        verbose=verbose,
    )

    return testResult


# --- TestBasis ---
def TestBasis(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Basis operation to find a Basis for the column space of a matrix.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the Basis is correct, False otherwise.
    """

    def BasisOfVectors(vectors):
        """
        Returns a set of linearly independent vectors forming the Basis of the input matrix.

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
    BasisA = Basis(span=a)
    c = BasisOfVectors(a.matrix)

    if verbose:

        print(f"BasisA:\n{BasisA}")
        print(f"C:\n{c}")

    correct = BasisA.shape == c.shape

    if (
        BasisA.shape[0] == 1
        and BasisA.shape[1] == 1
        and c.shape[0] == 1
        and c.shape[1] == 1
    ):

        if c[0][0] == 0 and a[0][0] == 0:

            return True

    if not correct:

        print(f"\n\nError causing components:\n")
        print(f"Shape BasisA: {BasisA.shape}")
        print(f"Shape C: {c.shape}\n")

    for row in range(BasisA.shape[0]):

        for col in range(BasisA.shape[1]):

            if not correct:

                break

            correct = abs(BasisA[row][col] - c[row][col]) < 0.00001

            if not correct:

                print(f"\n\nError causing components:\n")
                print(f"BasisA: {BasisA[row][col]}")
                print(f"C: {c[row][col]}")

                break

    if not correct:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")

        print(f"BasisA:\n{BasisA}\n")
        print(f"C:\n{c}")

        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")
        return True


def RandomTestBasis(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates Random dimensions and values to test the TestBasis function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the TestBasis function.
    """
    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = TestBasis(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestQRDecomposition ---
def TestQRDecomposition(
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

    q, r = QRDecomposition(span=a)

    BasisA = Basis(span=a)

    q2, r2 = np.linalg.qr(BasisA)

    try:

        if verbose:

            print(f"BasisA:\n{BasisA}\n")

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

            print(f"BasisA:\n{BasisA}\n")

            print(f"q:\n{q}\n")
            print(f"q2:\n{FloatMatrix(q2)}")
            print(f"r:\n{r}\n")
            print(f"r2:\n{FloatMatrix(r2)}")

            return False

        for colNum in range(q.numCols):

            FirstNonZeroIndex = None

            for row in range(q.numRows):

                if abs(q[row][colNum] - 0) > 0.000001:

                    FirstNonZeroIndex = row

                    break

            if (
                q[FirstNonZeroIndex][colNum] > 0.000001
                and q2[FirstNonZeroIndex][colNum] < -0.000001
            ) or (
                q[FirstNonZeroIndex][colNum] < -0.000001
                and q2[FirstNonZeroIndex][colNum] > 0.000001
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

            print(f"BasisA:\n{BasisA}\n")

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

            print(f"BasisA:\n{BasisA}\n")

            print(f"q:\n{q}\n")
            print(f"q2:\n{FloatMatrix(q2)}")
            print(f"r:\n{r}\n")
            print(f"r2:\n{FloatMatrix(r2)}")

            return False

        for rowNum in range(r.numRows):

            FirstNonZeroIndex = None

            for col in range(r.numCols):

                if abs(r[rowNum][col] - 0) > 0.000001:

                    FirstNonZeroIndex = col

                    break

            if (
                r[rowNum][FirstNonZeroIndex] > 0.000001
                and r2[rowNum][FirstNonZeroIndex] < -0.000001
            ) or (
                r[rowNum][FirstNonZeroIndex] < -0.000001
                and r2[rowNum][FirstNonZeroIndex] > 0.000001
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

            print(f"BasisA:\n{BasisA}\n")

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

        print(f"BasisA:\n{BasisA}\n")

        print(f"q:\n{q}\n")
        print(f"q2:\n{FloatMatrix(q2)}")
        print(f"r:\n{r}\n")
        print(f"r2:\n{FloatMatrix(r2)}")

        return False


def RandomTestQRDecomposition(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """ """

    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = TestQRDecomposition(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestGramSchmidtRandomSpan ---
def TestGramSchmidtRandomSpan(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Gram Schmidt operation to find an orthonormal Basis from an input Basis.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the orthonormal Basis is correct, False otherwise.
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

        FirstNonZeroIndex = None

        for row in range(orthonormalA.numRows):

            if abs(orthonormalA[row][colNum] - 0) > 0.0001:

                FirstNonZeroIndex = row

                break

        if (
            orthonormalA[FirstNonZeroIndex][colNum] > 0.0001
            and c[FirstNonZeroIndex][colNum] < -0.0001
        ) or (
            orthonormalA[FirstNonZeroIndex][colNum] < -0.0001
            and c[FirstNonZeroIndex][colNum] > 0.0001
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


def RandomTestGramSchmidtRandomSpan(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates Random dimensions and values to test the TestGramSchmidt function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the TestBasis function.
    """
    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = TestGramSchmidt(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestGramSchmidt ---
def TestGramSchmidt(
    rows: int = 4,
    cols: int = 3,
    min_: int = 0,
    max_: int = 10,
    verbose: bool = False,
) -> bool:
    """
    Tests the Gram Schmidt operation to find an orthonormal Basis from an input Basis.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        min_ (int): Minimum value for matrix elements.
        max_ (int): Maximum value for matrix elements.
        verbose (bool): If True, prints the matrix and results.

    Returns:
        bool: True if the orthonormal Basis is correct, False otherwise.
    """

    a = Matrix(rows, cols, min_, max_)

    while np.count_nonzero(a.matrix) == 0:

        a = Matrix(rows, cols, min_, max_)

    BasisA = Basis(span=a)

    if verbose:

        print(f"A:\n{a}")

    try:

        orthonormalA = GramSchmidt(span=BasisA)

    except IndexError:

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")
        print(f"BasisA:\n{BasisA}\n")

        return False

    c, R = np.linalg.qr(BasisA)

    if verbose:

        print(f"BasisA:\n{BasisA}")
        print(f"C:\n{c}")

    correct = orthonormalA.shape == c.shape

    if not correct:

        print(f"\n\nError causing components:\n")
        print(f"Shape orthnormalA: {orthonormalA.shape}")
        print(f"Shape C: {c.shape}\n")

        print(f"{rows=}\n{cols=}\n{min_=}\n{max_=}\n{verbose=}\n")

        print(f"A:\n{a}\n")
        print(f"BasisA:\n{BasisA}\n")

        print(f"orthonormalA:\n{orthonormalA}\n")
        print(f"FloatMatrix(c):\n{FloatMatrix(c)}")
        print(f"C:\n{c}")

        return False

    for colNum in range(orthonormalA.numCols):

        FirstNonZeroIndex = None

        for row in range(orthonormalA.numRows):

            if abs(orthonormalA[row][colNum] - 0) > 0.0001:

                FirstNonZeroIndex = row

                break

        if (
            orthonormalA[FirstNonZeroIndex][colNum] > 0.0001
            and c[FirstNonZeroIndex][colNum] < -0.0001
        ) or (
            orthonormalA[FirstNonZeroIndex][colNum] < -0.0001
            and c[FirstNonZeroIndex][colNum] > 0.0001
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
        print(f"BasisA:\n{BasisA}\n")

        print(f"orthonormalA:\n{orthonormalA}\n")
        print(f"FloatMatrix(c):\n{FloatMatrix(c)}")
        print(f"C:\n{c}")

        return False

    else:

        if verbose:

            print(f"Correct Shape\nCorrect Elements")

        return True


def RandomTestGramSchmidt(
    minSize: int = -100, maxSize: int = 100, verbose: bool = False
) -> bool:
    """
    Generates Random dimensions and values to test the TestGramSchmidt function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions and element values.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the TestBasis function.
    """
    rows = random.randint(1, maxSize)
    cols = random.randint(1, maxSize)

    testResult = TestGramSchmidt(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxSize,
        verbose=verbose,
    )

    return testResult


# --- TestDeterminant ---
def TestDeterminant(
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


def RandomTestDeterminant(
    minSize: int = -100, maxSize: int = 12, maxVal: int = 100, verbose: bool = False
) -> bool:
    """
    Generates Random square dimensions and values to test the TestDeterminant function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions.
        maxVal (int): Maximum absolute value for matrix elements.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the TestDeterminant function.
    """
    rows = random.randint(0, maxSize)
    cols = rows  # Ensures the matrix is square

    testResult = TestDeterminant(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxVal,
        verbose=verbose,
    )

    return testResult


# --- TestInverse ---
def TestInverse(
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
        bool: True if the Inverse is correct or None if the matrix is singular, False otherwise.
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


def RandomTestInverse(
    minSize: int = -100, maxSize: int = 100, maxVal: int = 100, verbose: bool = False
) -> bool:
    """
    Generates Random square dimensions and values to test the TestInverse function.

    Args:
        minSize (int): Minimum size for the matrix dimensions and element values.
        maxSize (int): Maximum size for the matrix dimensions.
        maxVal (int): Maximum absolute value for matrix elements.
        verbose (bool): If True, prints the testing details.

    Returns:
        bool: Result of the TestInverse function.
    """
    rows = random.randint(0, maxSize)
    cols = rows  # Ensures the matrix is square

    testResult = TestInverse(
        rows=rows,
        cols=cols,
        min_=minSize,
        max_=maxVal,
        verbose=verbose,
    )

    return testResult


def TestMatrixWithSymbols(verbose: bool = False) -> bool:
    """
    Tests matrix operations involving symbols.

    Args:
        verbose: If True, prints detailed information.

    Returns:
        bool: True if the test passes, False otherwise.
    """
    x, y, z = symbols("x y z")
    a = Matrix([[x, y], [y, z]])
    b = Matrix([[1, 0], [0, 1]])
    try:

        result = MatrixMultiply(a, b)
        if verbose:

            print(f"Result:\n{result}")
        return True

    except Exception as e:

        if verbose:

            print(f"Error: {e}")
        return False


# ============================
# Utility Functions
# ============================


def CalculateTimeComplexity(
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
            func.__name__ == "RandomTestInverse"
            or func.__name__ == "RandomTestDeterminant"
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

    # Test Base2RREF
    # if noErrors:

    #     for _ in tqdm(range(1000), desc="Base2RREF"):

    #         testResult = RandomTestBase2RREF(
    #             minSize=minSize, maxSize=maxSize, verbose=False
    #         )

    #         if not testResult:

    #             print("BASE2RREF")
    #             noErrors = False

    #             break

    # Test qrDecomposition
    if noErrors:

        for _ in tqdm(range(1000), desc="qrDecomposition"):

            testResult = RandomTestQRDecomposition(
                minSize=minSize, maxSize=15, verbose=False
            )

            if not testResult:

                print("qrDecomposition")
                noErrors = False

                break

    # Test Basis
    if noErrors:

        for _ in tqdm(range(1000), desc="TestBasis"):

            testResult = RandomTestBasis(minSize=minSize, maxSize=15, verbose=False)

            if not testResult:

                print("Basis")
                noErrors = False

                break

    # Test Transpose
    if noErrors:

        for _ in tqdm(range(1000), desc="TestTranspose"):

            testResult = RandomTestTranspose(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:

                print("Transpose")
                noErrors = False
                break

    # Test GramSchmidt
    if noErrors:

        for _ in tqdm(range(1000), desc="TestGramSchmidt"):

            testResult = RandomTestGramSchmidt(
                minSize=minSize, maxSize=15, verbose=False
            )
            if not testResult:

                print("GramSchmidt")
                noErrors = False
                break

    # Test GramSchmidtRandomSpan
    if noErrors:

        for _ in tqdm(range(1000), desc="TestGramSchmidtRandomSpan"):

            testResult = RandomTestGramSchmidtRandomSpan(
                minSize=minSize, maxSize=15, verbose=False
            )
            if not testResult:

                print("GramSchmidtRandomSpan")
                noErrors = False
                break

    # Test UnitVector
    if noErrors:

        for _ in tqdm(range(10000), desc="TestUnitVector"):

            testResult = RandomTestUnitVector(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:

                print("UnitVector")
                noErrors = False
                break

    # Test Inverse
    if noErrors:

        for _ in tqdm(range(50), desc="TestInverse"):

            testResult = RandomTestInverse(
                minSize=minSize, maxSize=8, maxVal=maxSize, verbose=False
            )
            if not testResult:

                print("Inverse")
                noErrors = False
                break

    # Test Determinant
    if noErrors:

        for _ in tqdm(range(100), desc="TestDeterminant"):

            testResult = RandomTestDeterminant(
                minSize=minSize, maxSize=8, maxVal=maxSize, verbose=False
            )
            if not testResult:

                print("Determinant")
                noErrors = False
                break

    # Test VectorDot
    if noErrors:

        for _ in tqdm(range(1000), desc="TestVectorDot"):

            testResult = RandomTestVectorDot(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:

                print("VectorDot")
                noErrors = False
                break

    # Test Multiply
    if noErrors:

        for _ in tqdm(range(10), desc="TestMultiply"):

            testResult = RandomTestMultiply(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:

                print("Multiply")
                noErrors = False
                break

    # Test MatrixAdd
    if noErrors:

        for _ in tqdm(range(100), desc="TestMatrixAdd"):

            testResult = RandomTestMatrixAdd(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:

                print("MatrixAdd")
                noErrors = False
                break

    # Test MatrixSubtract
    if noErrors:

        for _ in tqdm(range(100), desc="TestMatrixSubtract"):

            testResult = RandomTestMatrixSubtract(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:

                print("MatrixSubtract")
                noErrors = False
                break

    # Test ScalarMultiply
    if noErrors:

        for _ in tqdm(range(100), desc="TestScalarMultiply"):

            testResult = RandomTestScalarMultiply(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:

                print("ScalarMultiply")
                noErrors = False
                break

    # Test VectorLength
    if noErrors:

        for _ in tqdm(range(1000), desc="TestVectorLength"):

            testResult = RandomTestVectorLength(
                minSize=minSize, maxSize=maxSize, verbose=False
            )
            if not testResult:

                print("VectorLength")
                noErrors = False
                break

    # Test RREF
    if noErrors:

        for _ in tqdm(range(25), desc="TestRREF"):

            testResult = RandomTestRREF(minSize=minSize, maxSize=maxSize, verbose=False)
            if not testResult:

                print("RREF")

                noErrors = False
                break

    # Test Matrix with Symbols
    if noErrors:

        for _ in tqdm(range(100), desc="TestMatrixWithSymbols"):

            testResult = TestMatrixWithSymbols(verbose=False)
            if not testResult:

                print("MatrixWithSymbols")
                noErrors = False
                break


# ...add similar test functions for other operations involving Symbols...
